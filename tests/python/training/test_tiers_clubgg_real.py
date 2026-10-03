"""The "clubgg_real" table tier (2026-10-03): the owner's own ClubGG tables, measured from
their hand histories (953 hands at $10/$20, 3bb ante, 6-max), as a training / evaluation
tier — `--mix-tiers clubgg_real`, `--stack-dist clubgg_real`.

Pinned: the tier draws tables that look like the measured ones (seat counts, stack
percentiles, how often short and deep stacks share a table, stack-to-pot on the flop), it
keeps to --num-seats-range / --stack-range, and every EXISTING tier still draws exactly
what it drew before the tier was added (training stays bit-exact unless it is asked for)."""

from __future__ import annotations

import collections
import hashlib

import numpy as np
import pytest

from plo5bp.train import tiers
from plo5bp.train.cli import _parse_mix_tiers

SEATS = (2, 3, 4, 5, 6)
BB, ANTE = 10_000, 30_000


def _tables(n, seed=1, seats=SEATS, lo=1.0, hi=300.0):
    rng = np.random.default_rng(seed)
    return [tiers._sample_game_config(seats, lo, hi, BB, ANTE, rng, stack_dist="clubgg_real")[0]
            for _ in range(n)]


def test_every_existing_tier_draws_exactly_what_it_drew_before():
    # (the digest of 300 draws of every old (stack, seats) distribution, computed with the
    # code before "clubgg_real" existed — plus where the generator ended up)
    h = hashlib.sha256()
    for dist in ("uniform", "clubgg", "clubgg_deep", "clubgg_mix", "deep", "full_mix"):
        for seats_dist in ("uniform", "clubgg"):
            rng = np.random.default_rng(1234)
            for _ in range(300):
                cfg, eff = tiers._sample_game_config(SEATS, 1.0, 300.0, BB, ANTE, rng,
                                                     stack_dist=dist, seats_dist=seats_dist)
                h.update(repr((cfg.num_seats, cfg.starting_stacks, eff)).encode())
            h.update(rng.integers(0, 2**63 - 1).tobytes())
    assert h.hexdigest() == "ee2bab9688ad1d2a29a6845e7a08d39f4d7c7f06a5a19188894e71ed3d0b992d"


def test_the_real_tier_looks_like_the_owner_s_tables():
    cfgs = _tables(20_000)
    n = collections.Counter(c.num_seats for c in cfgs)
    measured_seats = {2: 1.2, 3: 5.2, 4: 20.9, 5: 31.2, 6: 41.6}
    for k, want in measured_seats.items():
        assert abs(100 * n[k] / len(cfgs) - want) < 1.5, (k, n[k])
    stacks = np.array([s / BB for c in cfgs for s in c.starting_stacks])
    measured = {10: 19, 25: 21, 50: 43, 75: 90, 90: 161}  # (the real seats' percentiles, bb)
    for q, want in measured.items():
        got = float(np.percentile(stacks, q))
        assert abs(got - want) <= max(2.0, 0.08 * want), (q, got, want)
    assert 0.12 < float(np.mean((stacks >= 19.5) & (stacks <= 20.5))) < 0.16  # (the 20bb buy-in)
    mixed = np.mean([min(c.starting_stacks) <= 25 * BB and max(c.starting_stacks) >= 100 * BB for c in cfgs])
    assert 0.53 < mixed < 0.62  # (real: 58% — the old three-tier mix: 10%)
    spr = [(min(c.starting_stacks[0], max(c.starting_stacks[1:])) - ANTE) / (ANTE * c.num_seats) for c in cfgs]
    assert 2.3 < float(np.median(spr)) < 2.7  # (real: 2.47 — the old mix: 5.3)


def test_the_real_tier_keeps_to_the_seat_and_stack_ranges():
    cfgs = _tables(4000, seed=2, seats=(2, 3))
    n = collections.Counter(c.num_seats for c in cfgs)
    assert set(n) == {2, 3} and 0.75 < n[3] / len(cfgs) < 0.88  # (5.2 : 1.2, renormalized)
    deep = _tables(3000, seed=3, hi=100.0)
    assert max(s for c in deep for s in c.starting_stacks) <= 100 * BB
    # (every table still has two seats that can act after the ante)
    assert all(sum(s > ANTE for s in c.starting_stacks) >= 2 for c in _tables(5000, seed=4))


def test_the_tier_is_one_the_command_line_takes():
    assert _parse_mix_tiers("clubgg_real") == ["clubgg_real"]
    assert _parse_mix_tiers("clubgg,clubgg_deep,deep,clubgg_real")[-1] == "clubgg_real"
    with pytest.raises(SystemExit):
        _parse_mix_tiers("clubg_real")
