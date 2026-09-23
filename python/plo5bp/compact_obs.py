"""Compact storage for rollout observations — training STORAGE only.

A PLO observation row is mostly exact 0/1 flags: in the minimal layout 704 of
the 796 columns (card multi-hots, one-hots, seat masks, each history slot's
seat/gate/street one-hots — `encoding.FLAG_MASK_MINIMAL`), yet dense storage
spends a float32 on each. The batched rollout instead keeps those columns as
bits (numpy `packbits` order) and every other column verbatim as float32:

    minimal (796):  88 B of bits + 92 x 4 B  =   456 B/row  (dense 3,184 B)
    full   (1171):  88 B of bits + 467 x 4 B = 1,956 B/row  (dense 4,684 B)

Unpacking reproduces every value BIT-EXACTLY (flags come back as 0.0/1.0, real
columns are copied), so the networks, the losses and the PPO update see exactly
what dense storage would give them. Only memory shrinks — the host staging
buffers, the end-of-rollout host->GPU copy, the GPU-resident batch — plus the
per-row host copies, which is what lets `--rollout-length` grow (CLAUDE.md
"Longer rollouts are always better").

Pieces:
- `layout_for(variant, obs_mode)` — the layout for a collection (None = keep
  dense: NLH's layout is not mapped).
- Rust `_engine.pack_obs_rows` packs rows straight into the caller's buffers and
  REJECTS any flag value that is not exactly 0.0/1.0, so an encoder change can
  never silently corrupt stored rows; `pack_rows_np` is the numpy reference it
  is tested against.
- `PackedObs` stands in for the (N, obs_dim) float32 tensor in `Batch.obs`:
  `.shape`, `.device`, `.to()` and row indexing, which returns the DENSE float32
  rows — that is how `rollout.iter_minibatches` unpacks each PPO minibatch on
  the learner device. Anything else (feeding it to a network, `torch.equal`, …)
  fails loudly; tests/diagnostics use `as_dense()` — never on a full production
  batch (it materializes the whole dense matrix).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .encoding import FLAG_MASK_FULL, FLAG_MASK_MINIMAL

try:
    from ._engine import pack_obs_rows as _rust_pack_obs_rows
    from ._engine import unpack_obs_rows as _rust_unpack_obs_rows
except ImportError:  # an engine built before compact storage existed
    _rust_pack_obs_rows = None
    _rust_unpack_obs_rows = None

# False only for a stale engine binary; the rollout then stores dense rows and
# says so once (rollout._resolve_obs_layout).
RUST_PACKER_AVAILABLE: bool = _rust_pack_obs_rows is not None

# Bit pattern of 1.0f32 — with 0 the only values a flag column may hold.
_F32_ONE_BITS = np.uint32(0x3F80_0000)

# Rows unpacked per chunk: bounds the uint8 / float32 temporaries of `unpack`
# (a PPO minibatch can be millions of rows on the pod).
_UNPACK_CHUNK_ROWS = 262_144

_PLO_VARIANTS = frozenset({"plo4_double_bomb", "plo5_double_bomb", "plo6_double_bomb"})


@dataclass(frozen=True, eq=False)
class CompactObsLayout:
    """Which columns of an `obs_dim`-wide observation are stored as bits
    (`flag_cols`) and which verbatim as float32 (`real_cols`). Both ascending;
    together they partition range(obs_dim)."""

    name: str
    obs_dim: int
    flag_cols: np.ndarray  # (F,) int64
    real_cols: np.ndarray  # (R,) int64
    _torch_cache: dict = field(default_factory=dict, repr=False)

    @property
    def n_flag(self) -> int:
        return int(self.flag_cols.size)

    @property
    def n_real(self) -> int:
        return int(self.real_cols.size)

    @property
    def n_bytes(self) -> int:
        return (self.n_flag + 7) // 8

    @property
    def row_bytes(self) -> int:
        """Stored bytes per observation row (dense: 4 * obs_dim)."""
        return self.n_bytes + 4 * self.n_real

    def torch_consts(
        self, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(flag column index, real column index, byte -> 8 float bits lookup
        table (256, 8), MSB first) on `device`, cached — `unpack` runs once
        per PPO minibatch."""
        key = str(device)
        consts = self._torch_cache.get(key)
        if consts is None:
            byte = torch.arange(256, dtype=torch.int64)[:, None]
            lut = ((byte >> torch.arange(7, -1, -1)) & 1).to(torch.float32)
            consts = (
                torch.from_numpy(self.flag_cols).to(device),
                torch.from_numpy(self.real_cols).to(device),
                lut.to(device),
            )
            self._torch_cache[key] = consts
        return consts


def _layout_from_mask(name: str, flag_mask: np.ndarray) -> CompactObsLayout:
    mask = np.asarray(flag_mask, dtype=bool)
    return CompactObsLayout(
        name=name,
        obs_dim=int(mask.size),
        flag_cols=np.nonzero(mask)[0].astype(np.int64),
        real_cols=np.nonzero(~mask)[0].astype(np.int64),
    )


MINIMAL_LAYOUT = _layout_from_mask("minimal", FLAG_MASK_MINIMAL)
FULL_LAYOUT = _layout_from_mask("full", FLAG_MASK_FULL)


def layout_for(variant: str, obs_mode: str) -> CompactObsLayout | None:
    """The compact layout for a (variant, obs_mode) collection, or None to
    keep dense storage (NLH: its 995-wide layout is not mapped — the NLH PPO
    lineage is retired)."""
    if variant not in _PLO_VARIANTS:
        return None
    if obs_mode == "minimal":
        return MINIMAL_LAYOUT
    if obs_mode == "full":
        return FULL_LAYOUT
    raise ValueError(f"unknown obs_mode {obs_mode!r} (expected 'full' or 'minimal')")


def pack_rows_into(
    obs: np.ndarray,
    rows: np.ndarray,
    layout: CompactObsLayout,
    out_bits: np.ndarray,
    out_real: np.ndarray,
    offset: int,
) -> None:
    """Pack `obs[rows]` into `out_bits` / `out_real` at row `offset` (Rust,
    parallel; raises ValueError on a non-0/1 flag)."""
    _rust_pack_obs_rows(
        obs,
        np.asarray(rows, dtype=np.int64),
        layout.flag_cols,
        layout.real_cols,
        out_bits,
        out_real,
        int(offset),
    )


def pack_rows_np(
    obs: np.ndarray, rows: np.ndarray, layout: CompactObsLayout
) -> tuple[np.ndarray, np.ndarray]:
    """numpy reference for the Rust packer (tests): (bits, reals) of `obs[rows]`."""
    rows = np.asarray(rows, dtype=np.int64)
    x = np.asarray(obs, dtype=np.float32)[rows]
    flags = np.ascontiguousarray(x[:, layout.flag_cols])
    u = flags.view(np.uint32)
    ok = (u == 0) | (u == _F32_ONE_BITS)
    if not ok.all():
        r, c = (int(v) for v in np.argwhere(~ok)[0])
        raise ValueError(
            f"obs row {int(rows[r])} column {int(layout.flag_cols[c])} holds "
            f"{flags[r, c]!r}, not a 0/1 flag"
        )
    return np.packbits(u != 0, axis=1), np.ascontiguousarray(x[:, layout.real_cols])


def unpack(
    bits: torch.Tensor, real: torch.Tensor, layout: CompactObsLayout
) -> torch.Tensor:
    """Dense (k, obs_dim) float32 rows from their packed form, on `bits`'
    device. Bit-exact inverse of the packers. CPU tensors go through the Rust
    unpacker (parallel; faster than gathering dense rows, since it reads ~7x
    fewer bytes); other devices through a byte->bits lookup table and two
    column scatters, in row chunks that bound the temporaries."""
    k = int(bits.shape[0])
    out = torch.empty((k, layout.obs_dim), dtype=torch.float32, device=bits.device)
    if bits.device.type == "cpu" and _rust_unpack_obs_rows is not None:
        if k:
            _rust_unpack_obs_rows(
                bits.contiguous().numpy(),
                real.contiguous().numpy(),
                layout.flag_cols,
                layout.real_cols,
                out.numpy(),
            )
        return out
    flag_idx, real_idx, lut = layout.torch_consts(bits.device)
    for start in range(0, k, _UNPACK_CHUNK_ROWS):
        stop = min(k, start + _UNPACK_CHUNK_ROWS)
        byte_idx = bits[start:stop].reshape(-1).to(torch.int64)
        flags = lut.index_select(0, byte_idx).reshape(stop - start, -1)
        dst = out[start:stop]
        dst.index_copy_(1, flag_idx, flags[:, : layout.n_flag])
        dst.index_copy_(1, real_idx, real[start:stop])
    return out


class PackedObs:
    """Stand-in for a (N, obs_dim) float32 observation tensor held in compact
    form (see the module docstring). Row indexing returns DENSE rows."""

    __slots__ = ("bits", "real", "layout")
    dtype = torch.float32  # the dtype of the rows it yields

    def __init__(
        self, bits: torch.Tensor, real: torch.Tensor, layout: CompactObsLayout
    ) -> None:
        if bits.dtype != torch.uint8 or bits.dim() != 2 or bits.shape[1] != layout.n_bytes:
            raise ValueError(
                f"PackedObs bits must be (N, {layout.n_bytes}) uint8, got "
                f"{tuple(bits.shape)} {bits.dtype}"
            )
        if real.dtype != torch.float32 or real.dim() != 2 or real.shape[1] != layout.n_real:
            raise ValueError(
                f"PackedObs real must be (N, {layout.n_real}) float32, got "
                f"{tuple(real.shape)} {real.dtype}"
            )
        if bits.shape[0] != real.shape[0] or bits.device != real.device:
            raise ValueError("PackedObs bits/real row counts or devices differ")
        self.bits = bits
        self.real = real
        self.layout = layout

    @property
    def shape(self) -> torch.Size:
        return torch.Size((int(self.bits.shape[0]), self.layout.obs_dim))

    @property
    def device(self) -> torch.device:
        return self.bits.device

    @property
    def nbytes(self) -> int:
        return int(self.bits.numel()) + 4 * int(self.real.numel())

    def __len__(self) -> int:
        return int(self.bits.shape[0])

    def size(self, dim: int | None = None):
        return self.shape if dim is None else self.shape[dim]

    def to(self, device, non_blocking: bool = False) -> "PackedObs":
        return PackedObs(
            self.bits.to(device, non_blocking=non_blocking),
            self.real.to(device, non_blocking=non_blocking),
            self.layout,
        )

    def __getitem__(self, rows) -> torch.Tensor:
        """DENSE float32 rows (`obs[index_tensor]`, `obs[a:b]`, `obs[i]`)."""
        if isinstance(rows, tuple):
            raise TypeError(
                "PackedObs supports row indexing only (obs[rows]); index the "
                "columns of the dense result"
            )
        bits, real = self.bits[rows], self.real[rows]
        if bits.dim() == 1:  # a single integer row
            return unpack(bits.unsqueeze(0), real.unsqueeze(0), self.layout)[0]
        return unpack(bits, real, self.layout)

    def dense(self) -> torch.Tensor:
        """The whole dense matrix — tests/diagnostics only (full size!)."""
        return unpack(self.bits, self.real, self.layout)

    @staticmethod
    def cat(parts: Sequence["PackedObs"]) -> "PackedObs":
        layout = parts[0].layout
        if any(p.layout is not layout for p in parts):
            raise ValueError("PackedObs.cat: mixed layouts")
        return PackedObs(
            torch.cat([p.bits for p in parts], dim=0),
            torch.cat([p.real for p in parts], dim=0),
            layout,
        )

    def __repr__(self) -> str:
        return (
            f"PackedObs({len(self)} x {self.layout.obs_dim} {self.layout.name}, "
            f"{self.nbytes / 1e6:.1f} MB, device={self.device})"
        )


def as_dense(obs: "torch.Tensor | PackedObs") -> torch.Tensor:
    """A dense tensor for either storage — tests/diagnostics (materializes a
    PackedObs in full)."""
    return obs.dense() if isinstance(obs, PackedObs) else obs
