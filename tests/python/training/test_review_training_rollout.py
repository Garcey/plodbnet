"""Regression tests for the 2026-09-20 code review, training workstream —
rollout.py items:

- A1  a config that can never deal a live hand RAISES (it used to spin the
      batched collector forever / trip a bare assert in the serial one);
      done-at-deal hands in an otherwise playable config are just re-dealt.
- A6  drain_inflight: every hand started before the row target is in the
      batch; with it off the collectors are the pre-fix ones.
- A10 no runt (num_minibatches+1)-th minibatch.
- A16 serial-path critic inputs use the variant's hole width.
"""

from __future__ import annotations

import threading

import numpy as np
import pytest
import torch

import plo5bp.rollout as rollout_mod
from plo5bp.config import (
    VARIANT_NLH,
    VARIANT_PLO4,
    VARIANT_PLO6,
    GameConfig,
    TrainingConfig,
)
from plo5bp.compact_obs import as_dense
from plo5bp.encoding_nlh import OBS_DIM_NLH
from plo5bp.env_batched import BatchedBombPotEnv
from plo5bp.network import (
    ActorCriticV2,
    ActorCriticV4,
    CentralCritic,
    opp_holes_multihot,
)
from plo5bp.rollout import (
    _BATCH_TENSOR_FIELDS,
    _minibatch_bounds,
    _resolve_drain_inflight,
    collect_rollout,
    collect_rollout_batched,
    collect_rollout_multiconfig,
    iter_minibatches,
)
from plo5bp.selfplay import OpponentPool
from plo5bp.sizing import NLH_ANCHOR_SPEC

BB = 10_000
ANTE = 30_000


def _with_deadline(fn, seconds: float = 120.0):
    """Run `fn` on a daemon thread; FAIL (instead of hanging the suite) when
    it does not return — the A1 bug was an infinite loop."""
    box: dict = {}

    def target() -> None:
        try:
            box["out"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller
            box["err"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        pytest.fail(f"collector still running after {seconds}s (hang)")
    if "err" in box:
        raise box["err"]
    return box["out"]


def _plo_model():
    torch.manual_seed(0)
    return ActorCriticV2(hidden_dim=32).eval()


def _tc(**kw) -> TrainingConfig:
    base = dict(num_envs=8, rollout_length=160, hidden_dim=32)
    base.update(kw)
    return TrainingConfig(**base)


COLLECTORS = [
    pytest.param(collect_rollout_batched, id="batched"),
    pytest.param(collect_rollout, id="serial"),
]


# --------------------------------------------------------------------- A1
@pytest.mark.parametrize("collector", COLLECTORS)
def test_a1_all_stacks_below_ante_raises_instead_of_hanging(collector):
    # agent_bindings/hang_repro.py: HU 2.5bb / 1.8bb vs a 3bb ante — both
    # seats are all-in on the ante, every hand is over AT DEAL.
    cfg = GameConfig(num_seats=2, starting_stacks=(25_000, 18_000))
    with pytest.raises(RuntimeError, match="terminal AT DEAL") as ei:
        _with_deadline(
            lambda: collector(
                _plo_model(), OpponentPool(capacity=4), cfg, _tc(),
                np.random.default_rng(0),
            ),
            seconds=60.0,
        )
    # The message names the offending config so the log is actionable.
    assert "starting_stacks=(25000, 18000)" in str(ei.value)


@pytest.mark.parametrize("collector", COLLECTORS)
def test_a1_one_short_seat_three_handed_still_collects(collector):
    # Seat 0 is all-in on the ante; seats 1-2 play a normal hand around it.
    cfg = GameConfig(
        num_seats=3, starting_stacks=(20_000, 60 * BB, 45 * BB)
    )
    tc = _tc()
    batch = _with_deadline(
        lambda: collector(
            _plo_model(), OpponentPool(capacity=4), cfg, tc,
            np.random.default_rng(1),
        )
    )
    assert batch.obs.shape[0] >= tc.rollout_length
    assert torch.isfinite(batch.returns).all()
    assert torch.isfinite(batch.advantages).all()


@pytest.mark.parametrize("collector", COLLECTORS)
def test_a1_done_at_deal_hands_are_redealt(collector):
    """NLH 4-handed where the BUTTON decides whether anyone can act: seats 1-2
    are all-in on the ante, seats 0/3 hold exactly ante+sb (all-in as soon as
    they post a blind). Button 0 -> two seats can act; button 2 -> nobody can
    (terminal at deal). A done-at-deal hand has no decision: it must be
    re-dealt, not stall the loop and not abort a playable config."""
    ante, sb = 5_000, 5_000
    cfg = GameConfig(
        num_seats=4, ante=ante, bb=BB, sb=sb, variant=VARIANT_NLH,
        starting_stacks=(ante + sb, ante, ante, ante + sb),
    )
    torch.manual_seed(0)
    model = ActorCriticV4(
        hidden_dim=32, obs_dim=OBS_DIM_NLH, num_layers=2,
        anchor_spec=NLH_ANCHOR_SPEC,
    ).eval()
    tc = _tc(num_envs=16, rollout_length=96)
    batch = _with_deadline(
        lambda: collector(
            model, OpponentPool(capacity=4), cfg, tc, np.random.default_rng(3)
        )
    )
    assert batch.obs.shape[0] >= tc.rollout_length
    # Every stored row is a real decision (a legal gate was taken).
    gm = batch.gate_masks.numpy()
    ga = batch.gate_actions.numpy()
    assert gm[np.arange(ga.shape[0]), ga].all()


def test_a1_run_match_survives_done_at_deal_hands():
    from plo5bp.eval import always_call_policy, run_match

    ante, sb = 5_000, 5_000
    cfg = GameConfig(
        num_seats=4, ante=ante, bb=BB, sb=sb, variant=VARIANT_NLH,
        starting_stacks=(ante + sb, ante, ante, ante + sb),
    )
    # reset() reports terminal=False even for a hand that is over at deal, so
    # the match loop used to ask a policy to act on an empty gate mask.
    stats = run_match(always_call_policy(), always_call_policy(), cfg, 40, seed=0)
    assert 0 < stats.num_hands < 40  # button-2 deals had nothing to play
    assert np.isfinite(stats.hero_reward_mean)
    dead = GameConfig(num_seats=2, starting_stacks=(25_000, 18_000))
    with pytest.raises(RuntimeError, match="terminal at deal"):
        run_match(always_call_policy(), always_call_policy(), dead, 10, seed=0)


# --------------------------------------------------------------------- A6
def test_a6_resolve_drain_default_and_override():
    tc = _tc()
    # getattr-based: works whether or not TrainingConfig carries the field.
    assert _resolve_drain_inflight(tc, None) is bool(
        getattr(tc, "drain_inflight", True)
    )
    assert _resolve_drain_inflight(tc, False) is False
    assert _resolve_drain_inflight(tc, True) is True

    class _Legacy:  # a config object with the flag switched off
        drain_inflight = False

    assert _resolve_drain_inflight(_Legacy(), None) is False
    assert _resolve_drain_inflight(_Legacy(), True) is True


def _collect(collector, drain, seed=5, env=None, **cfg_kw):
    torch.manual_seed(seed)
    model = _plo_model()
    torch.manual_seed(seed)
    kw = {"drain_inflight": drain}
    if env is not None:
        kw["env"] = env
    return collector(
        model, OpponentPool(capacity=1), GameConfig(num_seats=5),
        _tc(**cfg_kw), np.random.default_rng(seed), **kw,
    )


def test_a6_batched_drain_leaves_no_live_hand_and_drops_nothing():
    tc = _tc()
    env = BatchedBombPotEnv(
        tc.num_envs, GameConfig(num_seats=5),
        opp_outcome_mc=rollout_mod.TRAIN_OPP_OUTCOME_MC,
    )
    batch = _collect(collect_rollout_batched, True, env=env)
    # No env is mid-hand at exit ...
    assert env._dones.all()
    # ... and every learner decision ever taken is a row of the batch
    # (aggr_steps_total counts decisions as they are TAKEN, flushed or not).
    assert batch.aggr_steps_total == batch.obs.shape[0]
    # One terminal row per flushed (env, seat, hand) trajectory, all present.
    assert int(batch.is_terminal.sum()) > 0


def test_a6_serial_and_multiconfig_drain_drop_nothing():
    batch = _collect(collect_rollout, True)
    assert batch.aggr_steps_total == batch.obs.shape[0]

    torch.manual_seed(0)
    model = _plo_model()
    for legacy in (False, True):
        torch.manual_seed(11)
        mb = collect_rollout_multiconfig(
            model, OpponentPool(capacity=1),
            [GameConfig(num_seats=3), GameConfig(num_seats=6)],
            _tc(num_envs=8, rollout_length=200), np.random.default_rng(11),
            _legacy_staging=legacy, drain_inflight=True,
        )
        assert mb.aggr_steps_total == mb.obs.shape[0]


@pytest.mark.parametrize("collector", COLLECTORS)
def test_a6_drain_off_is_the_legacy_collector(collector):
    """drain off == the pre-fix loop: it exits at the first flush that reaches
    the target and drops the in-flight hands. Pinned structurally (same seed,
    same process): until the target both modes are the SAME run, so the legacy
    batch must be exactly the leading rows of the drained one, in order.
    (Byte-identity with the pre-fix module itself — rows, order, numpy + torch
    RNG state — was verified against `git show HEAD:python/plo5bp/rollout.py`
    when this landed.)"""
    off = _collect(collector, False)
    on = _collect(collector, True)
    n_off, n_on = off.obs.shape[0], on.obs.shape[0]
    target = _tc().rollout_length
    assert target <= n_off < n_on
    # legacy: decisions of the abandoned in-flight hands were taken but dropped
    assert off.aggr_steps_total > n_off
    for f in _BATCH_TENSOR_FIELDS:
        if f == "advantages":  # normalized over the whole batch -> differs
            continue
        assert torch.equal(as_dense(getattr(off, f)), as_dense(getattr(on, f))[:n_off]), f
    # and a second legacy run with the same seed is bit-identical
    again = _collect(collector, False)
    for f in _BATCH_TENSOR_FIELDS:
        assert torch.equal(as_dense(getattr(off, f)), as_dense(getattr(again, f))), f


def test_a6_slabs_and_pool_grow_instead_of_overflowing(monkeypatch):
    """The drained tail can exceed any fixed per-env slack; pool, own slabs
    and the multiconfig shared staging must GROW with identical results."""
    torch.manual_seed(0)
    model = _plo_model()
    cfgs = [GameConfig(num_seats=4), GameConfig(num_seats=6)]

    def run_single():
        torch.manual_seed(21)
        return collect_rollout_batched(
            model, OpponentPool(capacity=1), cfgs[1], _tc(),
            np.random.default_rng(21), drain_inflight=True,
        )

    def run_multi():
        torch.manual_seed(22)
        return collect_rollout_multiconfig(
            model, OpponentPool(capacity=1), cfgs,
            _tc(num_envs=8, rollout_length=240), np.random.default_rng(22),
            drain_inflight=True,
        )

    monkeypatch.setattr(rollout_mod, "_observed_slack_per_env", 0)
    roomy_single, roomy_multi = run_single(), run_multi()

    grown = {"n": 0}
    real_grow = rollout_mod._SlabAllocator.grow

    def counting_grow(self, slabs, used_rows, new_cap):
        grown["n"] += 1
        return real_grow(self, slabs, used_rows, new_cap)

    monkeypatch.setattr(rollout_mod._SlabAllocator, "grow", counting_grow)
    monkeypatch.setattr(rollout_mod, "_slack_per_env", lambda: 0)  # zero slack
    tight_single, tight_multi = run_single(), run_multi()
    assert grown["n"] >= 2  # own slabs AND the shared staging buffer grew
    for a, b in ((roomy_single, tight_single), (roomy_multi, tight_multi)):
        for f in _BATCH_TENSOR_FIELDS:
            assert torch.equal(as_dense(getattr(a, f)), as_dense(getattr(b, f))), f
        assert torch.equal(a.is_terminal, b.is_terminal)


# -------------------------------------------------------------------- A10
def test_a10_runt_tail_is_folded_into_full_minibatches():
    # 16 minibatches derived from a 96,000-row target; the collector
    # overshoots to 96,297 rows -> used to be 16 x 6000 + a 297-row 17th.
    bounds = _minibatch_bounds(96_297, 6_000)
    assert len(bounds) == 16
    sizes = [b - a for a, b in bounds]
    assert sum(sizes) == 96_297 and max(sizes) - min(sizes) <= 1
    assert max(sizes) <= 6_000 + 297  # spread, not stacked on one minibatch
    assert bounds[0][0] == 0 and bounds[-1][1] == 96_297
    assert all(bounds[i][1] == bounds[i + 1][0] for i in range(15))


@pytest.mark.parametrize(
    "n,bs,expected",
    [
        (96_000, 6_000, [6_000] * 16),              # exact multiple: unchanged
        (99_000, 6_000, [6_000] * 16 + [3_000]),    # tail == half: kept as-is
        (100_000, 6_000, [6_000] * 16 + [4_000]),   # tail > half: kept as-is
        (500, 6_000, [500]),                        # shorter than one batch
        (13, 4, [5, 4, 4]),                         # 1-row tail folded evenly
    ],
)
def test_a10_bounds_cases(n, bs, expected):
    assert [b - a for a, b in _minibatch_bounds(n, bs)] == expected


def test_a10_iter_minibatches_covers_every_row_once():
    n = 1_037
    batch = rollout_mod.Batch(
        obs=torch.arange(n, dtype=torch.float32)[:, None],
        gate_masks=torch.ones(n, 3, dtype=torch.bool),
        gate_actions=torch.zeros(n, dtype=torch.long),
        raise_chips=torch.zeros(n, dtype=torch.long),
        sizing=torch.zeros(n, 4, dtype=torch.long),
        anchor_actions=torch.zeros(n, dtype=torch.long),
        refine_u=torch.zeros(n),
        opp_holes=torch.zeros(n, 5, 5, dtype=torch.uint8),
        log_probs=torch.zeros(n),
        values=torch.zeros(n),
        returns=torch.zeros(n),
        advantages=torch.zeros(n),
        old_gate_logp=torch.zeros(n),
        old_anchor_logp=torch.zeros(n),
    )
    mbs = list(iter_minibatches(batch, 256, np.random.default_rng(0)))
    assert len(mbs) == 4  # 1037 = 4 x 256 + 13 -> no 13-row 5th minibatch
    seen = torch.cat([mb.obs[:, 0] for mb in mbs]).long()
    assert sorted(seen.tolist()) == list(range(n))


# -------------------------------------------------------------------- A16
@pytest.mark.parametrize(
    "variant,hole_w",
    [(VARIANT_NLH, 2), (VARIANT_PLO4, 4), (VARIANT_PLO6, 6)],
)
def test_a16_serial_critic_inputs_use_variant_hole_width(variant, hole_w):
    if variant == VARIANT_NLH:
        cfg = GameConfig.nlh_default(num_seats=3)
        torch.manual_seed(0)
        model = ActorCriticV4(
            hidden_dim=32, obs_dim=OBS_DIM_NLH, num_layers=2,
            anchor_spec=NLH_ANCHOR_SPEC,
        ).eval()
        obs_dim = OBS_DIM_NLH
    else:
        cfg = GameConfig(num_seats=3, variant=variant)
        model = _plo_model()
        obs_dim = model.torso[0].in_features
    torch.manual_seed(1)
    critic = CentralCritic(obs_dim=obs_dim, hidden_dim=32, num_blocks=1).eval()
    tc = _tc(num_envs=6, rollout_length=120)
    torch.manual_seed(2)
    batch = collect_rollout(
        model, OpponentPool(capacity=1), cfg, tc, np.random.default_rng(2),
        critic=critic,
    )
    oh = batch.opp_holes
    # PLO6 used to raise here; NLH/PLO4 used to come back padded to width 5.
    assert oh.shape == (batch.obs.shape[0], 5, hole_w) and oh.dtype == torch.uint8
    oh_np = oh.numpy()
    # 3-handed: slots 0-1 are real opponents (no pad inside), slots 2-4 empty.
    assert (oh_np[:, :2] < 52).all()
    assert (oh_np[:, 2:] == 255).all()
    mh = opp_holes_multihot(oh)
    assert torch.equal(mh.sum(-1), torch.full((oh.shape[0],), 2.0 * hole_w))
    # The stored values ARE the critic's read of (obs, stored opp block).
    with torch.no_grad():
        v = critic(as_dense(batch.obs), mh)
    assert torch.allclose(v, batch.values, atol=1e-5)
