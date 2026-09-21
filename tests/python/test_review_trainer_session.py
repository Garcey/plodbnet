"""Review 2026-09-20 — TrainerSession fixes (H2, H3, H4, F5/F7, F9, F10 +
latent items). Scoring (H1) lives in test_review_trainer_scoring.py and the
range grid (H5) in test_review_trainer_ranges.py.
"""

from __future__ import annotations

import importlib
import json
import threading
import time
from typing import Any

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE
from plo5bp.config import VARIANT_NLH, VARIANT_PLO5
from plo5bp.env import BombPotEnv
from plo5bp.network import ActorCriticV2
from plo5bp.sizing import sizing_from_info

CPU = torch.device("cpu")
SLUG = {GATE_FOLD: "fold", GATE_CHECK_CALL: "check_call", GATE_RAISE: "raise"}


@pytest.fixture()
def T():
    """The CURRENT `plo5bp.ui.trainer` module, resolved at test RUN time.

    The public-build test modules purge and re-import the ui modules
    (conftest `ui_purge`), and conftest's `trainer_factory` imports
    `TrainerSession` lazily - so after a purge it builds sessions from a NEW
    module object. A handle captured at collection time would then point at a
    stale copy: monkeypatching its `BombPotEnv` / `_snapshot_rng_state`, or
    taking its `_TORCH_RNG_LOCK`, would silently miss the code under test."""
    return importlib.import_module("plo5bp.ui.trainer")


def _deviate(ts) -> tuple[str, int | None]:
    """A legal hero action whose GATE differs from the recommendation (so the
    MC EV-loss estimate actually runs); falls back to the rec when the node
    offers no alternative."""
    h = ts.hand
    info = h.last_info
    dist = ts._node_dist(h.last_obs, info)
    for g in (GATE_CHECK_CALL, GATE_FOLD, GATE_RAISE):
        if g != dist["rec_gate"] and bool(info.gate_mask[g]):
            if g == GATE_RAISE:
                lo = int(info.min_raise_chips) or int(info.max_raise_chips)
                return "raise", lo
            return SLUG[g], None
    g = dist["rec_gate"]
    return SLUG[g], (dist["rec_chips"] if g == GATE_RAISE else None)


def _ev_numbers(node: Any, path: str = "") -> list[tuple[str, Any]]:
    """Every non-null value stored under an `ev_*` key anywhere in a payload
    (`ev_loss_hidden` is the flag that SAYS a number is withheld)."""
    out: list[tuple[str, Any]] = []
    if isinstance(node, dict):
        for k, v in node.items():
            p = f"{path}.{k}"
            if str(k).startswith("ev_") and k != "ev_loss_hidden":
                if v is not None:
                    out.append((p, v))
            else:
                out.extend(_ev_numbers(v, p))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            out.extend(_ev_numbers(v, f"{path}[{i}]"))
    return out


# --- H2: EV loss is withheld while the hand is live ---------------------------------


def test_ev_loss_hidden_mid_hand_and_revealed_at_terminal(trainer_factory):
    saw_hidden = saw_revealed = 0
    for seed in range(6):
        ts = trainer_factory(
            rng_seed=seed, stats_name=f"h2_{seed}.json", model_cls=ActorCriticV2,
            seats_mode="fixed", seats_fixed=3, mc_rollouts=4, stack_bb=100.0,
        )
        ts.new_hand()
        h = ts.hand
        steps = 0
        while not h.terminal and steps < 40:
            stats_before = ts.project_state()["trainer"]["stats"]
            frames = ts.act(*_deviate(ts))
            d = h.decisions[-1]
            for f in frames + [ts.project_state()]:
                tr = f["trainer"]
                fb = tr["feedback"]
                assert fb["decision_idx"] == d.decision_idx
                if tr["hand_active"]:
                    # LIVE: no EV number anywhere, review closed, and the
                    # stats blocks must not have moved (their delta would
                    # leak the same number).
                    assert fb["ev_loss_bb"] is None
                    assert fb["ev_loss_hidden"] is (d.ev_loss_bb is not None)
                    assert tr["review"] is None
                    assert _ev_numbers(tr["feedback"]) == []
                    assert tr["stats"] == stats_before
                    saw_hidden += bool(fb["ev_loss_hidden"])
                    # score / category are observation-only: always shown.
                    assert fb["category"] == d.category
                    assert fb["score"] == round(d.score, 1)
                else:
                    assert fb["ev_loss_hidden"] is False
                    if d.ev_loss_bb is not None:
                        assert fb["ev_loss_bb"] == round(d.ev_loss_bb, 3)
                        saw_revealed += 1
            steps += 1
        assert h.terminal
        # Terminal: review + stats carry every decision's estimate.
        s = ts.project_state()
        pills = s["trainer"]["review"]["decisions"]
        assert [p["ev_loss_bb"] for p in pills] == [
            round(d.ev_loss_bb, 3) if d.ev_loss_bb is not None else None
            for d in h.decisions
        ]
        assert s["trainer"]["stats"]["session"]["moves"] == len(h.decisions)
    assert saw_hidden > 0, "never exercised a live frame with a withheld estimate"
    assert saw_revealed > 0, "never exercised a terminal reveal"


def test_feedback_hidden_flag_false_when_ev_disabled(trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    for _ in range(5):
        ts.new_hand()
        if not ts.hand.terminal:
            break
    if ts.hand.terminal:
        pytest.skip("never reached a live hero node")
    frames = ts.act(*_deviate(ts))
    fb = frames[0]["trainer"]["feedback"]
    assert fb["ev_loss_bb"] is None and fb["ev_loss_hidden"] is False


# --- H3: MC continuations come from the ACTIVE backend's network --------------------


def _nlh_nets():
    from plo5bp.encoding_nlh import OBS_DIM_NLH
    from plo5bp.network import ActorCriticV4
    from plo5bp.sizing import NLH_ANCHOR_SPEC

    def make(seed):
        torch.manual_seed(seed)
        m = ActorCriticV4(
            hidden_dim=32, obs_dim=OBS_DIM_NLH, num_layers=1,
            anchor_spec=NLH_ANCHOR_SPEC,
        ).eval()
        for p in m.parameters():
            p.requires_grad_(False)
        return m

    return make(1), make(2)


def _count_act(model, calls, key):
    orig = model.act

    def act(*a, **k):
        calls[key] += 1
        return orig(*a, **k)

    model.act = act


def _nlh_session(T, tmp_path, ppo_nlh, backend, seats=3, mc=4, name="s.json"):
    torch.manual_seed(0)
    plo = ActorCriticV2(hidden_dim=32, num_layers=1).eval()
    ts = T.TrainerSession(plo, CPU, stats_path=tmp_path / name)
    # Exactly what the router's set_format does when a GTO host is loaded.
    ts.set_format(VARIANT_NLH, ppo_nlh, None, backend=backend)
    ts.set_settings(T.TrainerSettings(**{
        **ts.settings.model_dump(),
        "seats_mode": "fixed", "seats_fixed": seats, "mc_rollouts": mc,
    }))
    return ts, plo


def _hero_node_on_street(ts, want_postflop: bool, seeds=range(60)):
    """Deal (hero check/calls) until hero faces a live decision preflop /
    postflop; returns the HandRecord or skips."""
    for seed in seeds:
        ts.rng = np.random.default_rng(seed)
        ts.new_hand()
        h = ts.hand
        steps = 0
        while not h.terminal and steps < 30:
            street = int(h.last_info.raw_obs["street"])
            if (street >= 1) == want_postflop:
                return h
            s = ts.project_state()
            ts.act("check_call" if s["legal"]["check_call"] else "fold", None)
            steps += 1
    pytest.skip("never reached the wanted hero node")


@pytest.mark.parametrize("obs_form", ["live", "canonical"])
def test_gto_host_mc_goes_through_the_host_where_it_has_coverage(
    T, tmp_path, obs_form
):
    """H3, refined by coverage (the PRE-coverage version of this test pinned
    "every MC sample comes from the GTO net"; `PolicyNetHost.supports` now
    answers from recorded training coverage, so that is only true for the
    nodes the teacher covers)."""
    from plo5bp.gto.policy_host import PolicyNetHost

    ppo_nlh, gto_net = _nlh_nets()
    calls = {"ppo": 0, "gto": 0}
    _count_act(ppo_nlh, calls, "ppo")
    _count_act(gto_net, calls, "gto")
    # A teacher trained on 3-handed POSTFLOP nodes only. "canonical" makes the
    # real host re-encode every served node (canonical_serve_obs) inside the MC.
    host = PolicyNetHost(model=gto_net, device=CPU, meta={
        "coverage": {"streets": [1, 2, 3], "seats": [3], "streets_exact": [1, 2, 3]},
        "obs_forms": {obs_form: 1},
    })
    assert host.supports(seats=3, street=2) and not host.supports(seats=3, street=0)
    assert host.serves_canonical_obs is (obs_form == "canonical")

    ts, plo = _nlh_session(T, tmp_path, ppo_nlh, host)
    assert ts.backend is host
    assert ts.model is gto_net, "session model mirrors the active backend"
    assert ts._ppo_host.model is ppo_nlh, "fallback = the FORMAT's PPO actor"

    hosted: list[dict] = []
    real_act = host.act

    def spy_act(obs, info, *, deterministic=False, rng_seed=None):
        hosted.append({
            "street": int(info.raw_obs["street"]), "seed": rng_seed,
            "det": deterministic, "locked": T._TORCH_RNG_LOCK.locked(),
        })
        return real_act(obs, info, deterministic=deterministic, rng_seed=rng_seed)

    host.act = spy_act  # type: ignore[method-assign]

    # --- a POSTFLOP prefix: every continuation node is covered -> host only
    h = _hero_node_on_street(ts, want_postflop=True)
    calls["ppo"] = calls["gto"] = 0
    hosted.clear()
    a = ts._rollout_ev(h, list(h.action_log), GATE_CHECK_CALL, 0, 4, 123)
    assert hosted, "MC never went through the GTO host"
    assert calls["gto"] == len(hosted) and calls["ppo"] == 0
    for c in hosted:
        assert c["street"] >= 1
        assert c["seed"] is None and c["det"] is False  # continues OUR stream
        assert c["locked"] is True                       # seed->sample region
    assert ts._rollout_ev(h, list(h.action_log), GATE_CHECK_CALL, 0, 4, 123) == a

    # --- a PREFLOP prefix: the teacher never drives the uncovered nodes
    h = _hero_node_on_street(ts, want_postflop=False)
    calls["ppo"] = calls["gto"] = 0
    hosted.clear()
    ts._rollout_ev(h, list(h.action_log), GATE_CHECK_CALL, 0, 4, 123)
    assert calls["ppo"] > 0, "uncovered preflop nodes must use the PPO fallback"
    assert all(c["street"] >= 1 for c in hosted)

    # Re-applying the same format + host is a no-op (must not drop the hand).
    ts.set_format(VARIANT_NLH, ppo_nlh, None, backend=host)
    assert ts.hand is h

    # Leaving NLH falls back to a PPO host on the new format's model.
    ts.set_format(VARIANT_PLO5, plo, None)
    assert ts.model is plo and ts.backend.model is plo
    assert ts._ppo_host is ts.backend


class _FakeHost:
    """Minimal NON-PPO StrategyBackend: serves `model` through an inner PPO
    host, covers only what `covered(seats, street)` says."""

    name = "fake_gto"
    mode = "policy_net"

    def __init__(self, T, model, covered):
        from plo5bp.gto.backend import make_ppo_host

        self.T = T
        self.model = model
        self.device = CPU
        self.covered = covered
        self._inner = make_ppo_host(model, CPU)
        self.acts: list[dict] = []

    def supports(self, *, seats: int, street: int) -> bool:
        return bool(self.covered(int(seats), int(street)))

    def act(self, obs, info, *, deterministic=False, rng_seed=None):
        self.acts.append({
            "street": int(info.raw_obs["street"]), "seed": rng_seed,
            "det": deterministic, "locked": self.T._TORCH_RNG_LOCK.locked(),
        })
        return self._inner.act(
            obs, info, deterministic=deterministic, rng_seed=rng_seed
        )

    def node_distribution(self, obs, info):
        nd = self._inner.node_distribution(obs, info)
        nd.backend_name = self.name
        return nd

    def coverage_badge(self):
        return {"backend": self.name, "mode": self.mode, "label": "fake", "is_gto": False}


def _reference_hosted_rollout_ev(T, ts, host, h, prefix, gate, chips, n, node_seed):
    """One uninterrupted seeded stream, every continuation through
    `host.act(rng_seed=None)`, lock held throughout — the hosted twin of
    `_reference_rollout_ev`."""
    with T._TORCH_RNG_LOCK:
        torch.manual_seed(node_seed)
        total = 0.0
        live = []
        for _ in range(n):
            env = BombPotEnv(h.config, ev_runout_samples=32)
            obs, info = env.reset(h.seed, h.button)
            for a in prefix:
                obs, _, _, info = env.step_hybrid(a["gate"], a["chips"])
            obs, rewards, done, info = env.step_hybrid(gate, chips)
            if done:
                total += float(rewards[h.hero_seat])
            else:
                live.append([env, obs, info])
        while live:
            acts = [host.act(x[1], x[2], deterministic=False, rng_seed=None)
                    for x in live]
            nxt = []
            for x, (g, c) in zip(live, acts):
                obs2, rewards, done, info2 = x[0].step_hybrid(int(g), int(c))
                if done:
                    total += float(rewards[h.hero_seat])
                else:
                    nxt.append([x[0], obs2, info2])
            live = nxt
        return total / n / h.config.bb


def test_mc_through_a_covering_host_is_deterministic_and_stream_exact(
    T, tmp_path, monkeypatch
):
    ppo_nlh, gto_net = _nlh_nets()
    calls = {"ppo": 0, "gto": 0}
    _count_act(ppo_nlh, calls, "ppo")
    _count_act(gto_net, calls, "gto")
    fake = _FakeHost(T, gto_net, covered=lambda seats, street: True)
    ts, _ = _nlh_session(T, tmp_path, ppo_nlh, fake)
    h = _hero_node_on_street(ts, want_postflop=False)
    prefix = list(h.action_log)

    want = _reference_hosted_rollout_ev(
        T, ts, fake, h, prefix, GATE_CHECK_CALL, 0, 5, 2024
    )
    fake.acts.clear()
    calls["ppo"] = calls["gto"] = 0

    # reseed the global RNG in every gap where the lock is released
    real_snapshot = T._snapshot_rng_state
    depths = {"n": 0}

    def snapshot_then_clobber(device):
        state = real_snapshot(device)
        depths["n"] += 1
        torch.manual_seed(55 + depths["n"])
        torch.rand(3)
        return state

    monkeypatch.setattr(T, "_snapshot_rng_state", snapshot_then_clobber)
    got = ts._rollout_ev(h, prefix, GATE_CHECK_CALL, 0, 5, 2024)
    assert got == want
    assert depths["n"] >= 2
    assert fake.acts and calls["ppo"] == 0
    assert all(c["seed"] is None and not c["det"] and c["locked"] for c in fake.acts)


def test_mc_under_a_backend_that_covers_nothing_is_the_ppo_path_bit_for_bit(
    T, tmp_path
):
    ppo_nlh, gto_net = _nlh_nets()
    fake = _FakeHost(T, gto_net, covered=lambda seats, street: False)
    ts, _ = _nlh_session(T, tmp_path, ppo_nlh, fake)
    h = _hero_node_on_street(ts, want_postflop=False)
    prefix = list(h.action_log)
    for seed in (9, 10):
        want = _reference_rollout_ev(
            T, ts, h, prefix, GATE_CHECK_CALL, 0, 5, seed, model=ppo_nlh
        )
        assert ts._rollout_ev(h, prefix, GATE_CHECK_CALL, 0, 5, seed) == want
    assert [c for c in fake.acts if c["seed"] is None] == []  # never in the MC


def test_uncovered_nodes_fall_back_to_ppo_and_say_so(T, tmp_path):
    """Opponents, recommendations and payload tags follow per-node coverage:
    a postflop-only teacher never drives (or grades) a preflop node."""
    ppo_nlh, gto_net = _nlh_nets()
    calls = {"ppo": 0, "gto": 0}
    _count_act(ppo_nlh, calls, "ppo")
    _count_act(gto_net, calls, "gto")
    fake = _FakeHost(T, gto_net, covered=lambda seats, street: street >= 1)
    ts, _ = _nlh_session(T, tmp_path, ppo_nlh, fake, mc=0)

    tags_seen: set[str] = set()
    for seed in range(40):
        ts.rng = np.random.default_rng(seed)
        ts.new_hand()
        h = ts.hand
        steps = 0
        while not h.terminal and steps < 30:
            street = int(h.last_info.raw_obs["street"])
            want = "fake_gto" if street >= 1 else T.BACKEND_PPO_FALLBACK
            s = ts.project_state()
            assert s["trainer"]["node_backend"] == want
            assert s["trainer"]["backend"]["backend"] == "fake_gto"  # the badge
            frames = ts.act("check_call" if s["legal"]["check_call"] else "fold", None)
            assert frames[0]["trainer"]["feedback"]["backend"] == want
            assert h.decisions[-1].backend == want
            tags_seen.add(want)
            steps += 1
        assert h.terminal
        # live opponents: the fake host only ever acted postflop
        assert all(c["street"] >= 1 for c in fake.acts)
        assert all(c["seed"] is not None and c["locked"] for c in fake.acts)
        if not h.decisions:
            continue
        # review: every node carries the host that produced its numbers
        for ni, a in enumerate(h.action_log):
            nc = ts.review_at_node(ni)["trainer"]["review"]["node_current"]
            assert nc["backend"] == (
                "fake_gto" if a["street"] >= 1 else T.BACKEND_PPO_FALLBACK
            )
        rv = ts.review_at_node(0)["trainer"]["review"]
        assert [p["backend"] for p in rv["decisions"]] == \
            [d.backend for d in h.decisions]
        assert rv["current"]["backend"] == h.decisions[rv["decision"]].backend
        if tags_seen == {"fake_gto", T.BACKEND_PPO_FALLBACK} and fake.acts \
                and calls["ppo"] > 0:
            break
    assert tags_seen == {"fake_gto", T.BACKEND_PPO_FALLBACK}
    assert fake.acts, "the covering host never drove a postflop opponent"
    assert calls["ppo"] > 0, "the PPO fallback never drove a preflop opponent"
    # terminal projection has no live node
    assert ts.project_state()["trainer"]["node_backend"] is None


def test_ppo_backend_and_hosts_without_supports_are_never_second_guessed(
    T, trainer_factory, tmp_path
):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=0)
    ts.new_hand()
    host, tag = ts._backend_for(ts.hand.last_info)
    assert host is ts.backend and tag == "ppo"
    assert ts.project_state()["trainer"]["node_backend"] in ("ppo", None)

    ppo_nlh, gto_net = _nlh_nets()
    fake = _FakeHost(T, gto_net, covered=lambda seats, street: False)
    fake.supports = None  # a host with NO supports() serves every node
    ts2, _ = _nlh_session(T, tmp_path, ppo_nlh, fake)
    ts2.new_hand()
    if not ts2.hand.terminal:
        assert ts2._backend_for(ts2.hand.last_info) == (fake, "fake_gto")


def test_set_backend_keeps_critic_pairing(T, tmp_path):
    from plo5bp.gto.policy_host import PolicyNetHost

    ppo_nlh, gto_net = _nlh_nets()
    ts = T.TrainerSession(ppo_nlh, CPU, stats_path=tmp_path / "s.json")
    paired = ts._critic_obs_adapt
    ts.set_backend(PolicyNetHost(model=gto_net, device=CPU))
    assert ts.model is gto_net
    assert ts._critic_obs_adapt is paired


# --- H4: signed EV-loss accumulation -------------------------------------------------


def _rec(T, signed: float | None, score: float = 50.0, cat: str = "inaccuracy"):
    return T.DecisionRecord(
        decision_idx=0, street=1, action_log_idx=0,
        gate_probs=[0.2, 0.5, 0.3], alpha=1.0, beta=1.0,
        min_chips=100, max_chips=1000, rec_gate=GATE_CHECK_CALL, rec_chips=0,
        value_bb=0.0, pot_chips=1000, to_call_chips=0,
        user_gate=GATE_FOLD, user_chips=0, gate_ratio=0.4, size_q=1.0,
        score=score, category=cat,
        ev_loss_bb=None if signed is None else max(0.0, signed),
        ev_loss_signed_bb=signed,
    )


def test_stats_accumulate_signed_and_clamp_only_the_aggregate(T):
    b = T.StatsBlock()
    for signed in (-2.0, 3.0, -0.5, None):
        b.add_decision(_rec(T, signed))
    assert b.moves == 4 and b.ev_loss_n == 3
    assert b.ev_loss_signed_sum_bb == pytest.approx(0.5)
    # per-decision clamping (the old rule) would have reported 3.0
    assert b.project()["ev_loss_total_bb"] == 0.5

    neg = T.StatsBlock()
    neg.add_decision(_rec(T, -4.0))
    assert neg.ev_loss_signed_sum_bb == -4.0
    assert neg.project()["ev_loss_total_bb"] == 0.0  # clamped at DISPLAY only
    neg.add_decision(_rec(T, 6.0))  # ... and the noise still cancels later
    assert neg.project()["ev_loss_total_bb"] == 2.0


def test_legacy_stats_file_loads_as_already_clamped_history(T, tmp_path):
    legacy = {
        "hands": 10, "moves": 30, "score_sum": 2400.0,
        "cat_counts": {"best": 20, "wrong": 10},
        "ev_loss_sum_bb": 12.5,  # pre-H4: per-decision-clamped sum
    }
    b = T.StatsBlock.from_dict(legacy)
    assert b.ev_loss_sum_bb == 12.5 and b.ev_loss_signed_sum_bb == 0.0
    assert b.project()["ev_loss_total_bb"] == 12.5
    assert b.project()["ev_loss_per_hand_bb"] == 1.25
    # New estimates land in the signed sum; history is never touched, and
    # negative noise can't eat into it.
    b.add_decision(_rec(T, -3.0))
    assert b.ev_loss_sum_bb == 12.5
    assert b.project()["ev_loss_total_bb"] == 12.5
    b.add_decision(_rec(T, 5.0))
    assert b.project()["ev_loss_total_bb"] == 14.5
    # Round trip keeps both sums.
    again = T.StatsBlock.from_dict(json.loads(json.dumps(b.to_dict())))
    assert again == b

    # And through a real session file.
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"version": 2, "lifetime": legacy}))
    torch.manual_seed(0)
    ts = T.TrainerSession(
        ActorCriticV2(hidden_dim=32, num_layers=1).eval(), CPU, stats_path=path
    )
    assert ts.lifetime_stats.project()["ev_loss_total_bb"] == 12.5
    assert ts.lifetime_stats.hands == 10


def test_estimate_stores_signed_and_clamped(T, trainer_factory):
    ts = trainer_factory(seats_mode="fixed", seats_fixed=3, mc_rollouts=4)
    for _ in range(5):
        ts.new_hand()
        if not ts.hand.terminal:
            break
    if ts.hand.terminal:
        pytest.skip("never reached a live hero node")
    evs = iter([1.25, 0.5])  # user arm, then best arm: best < user
    ts._rollout_ev = lambda *a, **k: next(evs)  # type: ignore[method-assign]
    d = _rec(T, None)
    d.action_log_idx = len(ts.hand.action_log)
    ts._estimate_ev_loss(d)
    assert d.ev_loss_signed_bb == -0.75
    assert d.ev_loss_bb == 0.0
    assert d.ev_best_bb - d.ev_user_bb == pytest.approx(-0.75)


def test_abandoned_hand_contributes_nothing(trainer_factory, play_to_terminal):
    """POLICY (H4): stats are committed per COMPLETED hand. New hand / Repeat
    while a hand is live drops that hand's moves and EV loss entirely."""
    ts = trainer_factory(
        model_cls=ActorCriticV2, seats_mode="fixed", seats_fixed=3,
        mc_rollouts=2, stack_bb=100.0,
    )
    abandoned = 0
    for _ in range(8):
        ts.new_hand()
        if ts.hand.terminal:
            continue
        ts.act("check_call", None) if ts.project_state()["legal"]["check_call"] \
            else ts.act(*_deviate(ts))
        if ts.hand.terminal:
            continue
        assert ts.hand.decisions and not ts.hand.stats_committed
        abandoned += 1
        break
    assert abandoned, "never left a hand live after a hero move"
    before = (ts.session_stats.to_dict(), ts.lifetime_stats.to_dict())
    completed = ts.session_stats.hands

    ts.new_hand(repeat=True)       # abandon via Repeat ...
    ts.new_hand()                  # ... and via New hand
    assert (ts.session_stats.to_dict(), ts.lifetime_stats.to_dict()) == before

    if ts.hand.terminal:
        ts.new_hand()
    play_to_terminal(ts)
    assert ts.session_stats.hands == completed + 1
    assert ts.session_stats.moves == before[0]["moves"] + len(ts.hand.decisions)
    # a second finalize of the same hand can't double count
    ts._commit_hand_stats(ts.hand)
    assert ts.session_stats.hands == completed + 1


# --- F5 / F7: mc_rollouts cap + lock scope -------------------------------------------


def test_public_build_caps_mc_rollouts(T, trainer_factory, tmp_path, monkeypatch):
    local = trainer_factory(stats_name="local.json", mc_rollouts=256)
    assert local.settings.mc_rollouts == 256  # local build: untouched
    assert T.mc_rollouts_cap() == 256

    monkeypatch.setenv("PLO5BP_PUBLIC", "1")
    assert T.mc_rollouts_cap() == T.MC_ROLLOUTS_PUBLIC_CAP == 32
    pub = trainer_factory(stats_name="pub.json", mc_rollouts=256)
    assert pub.settings.mc_rollouts == 32
    assert pub.settings_by_variant[pub.variant].mc_rollouts == 32

    # A file persisted before the cap existed is clamped on load ...
    pub._persist()
    blob = json.loads((tmp_path / "pub.json").read_text())
    blob["settings_by_format"][pub.variant]["mc_rollouts"] = 200
    (tmp_path / "pub.json").write_text(json.dumps(blob))
    reloaded = trainer_factory(stats_name="pub.json")
    assert reloaded.settings.mc_rollouts == 32

    # ... and the EFFECTIVE count is capped even if a bigger value is forced in.
    reloaded.settings = reloaded.settings.model_copy(update={"mc_rollouts": 256})
    for _ in range(5):
        reloaded.new_hand()
        if not reloaded.hand.terminal:
            break
    if reloaded.hand.terminal:
        pytest.skip("never reached a live hero node")
    seen: list[int] = []
    reloaded._rollout_ev = (  # type: ignore[method-assign]
        lambda h, prefix, gate, chips, n, seed: seen.append(n) or 0.0
    )
    d = _rec(T, None)
    d.action_log_idx = len(reloaded.hand.action_log)
    reloaded._estimate_ev_loss(d)
    assert seen == [32, 32]


def test_public_settings_endpoint_clamps(T, tmp_path, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PLO5BP_PUBLIC", "1")
    monkeypatch.setenv("PLO5BP_TRAINER_STATS", str(tmp_path / "ep.json"))
    torch.manual_seed(0)
    model = ActorCriticV2(hidden_dim=32, num_layers=1).eval()
    router = T.create_trainer_router(model, CPU)
    user_ts = T.TrainerSession(model, CPU, stats_path=tmp_path / "u1.json")
    T.set_session_resolver(lambda: user_ts)
    try:
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        body = {**user_ts.settings.model_dump(), "mc_rollouts": 256}
        r = client.post("/trainer/settings", json=body)
        assert r.status_code == 200, r.text
        tr = r.json()["state"]["trainer"]
        assert tr["settings"]["mc_rollouts"] == 32
        assert tr["mc_rollouts_max"] == 32
        assert user_ts.settings.mc_rollouts == 32
    finally:
        T.set_session_resolver(None)


def _reference_rollout_ev(
    T, ts, h, prefix, gate, chips, n, node_seed, model=None
) -> float:
    """The PRE-FIX `_rollout_ev`, verbatim in behaviour: one uninterrupted
    `manual_seed(node_seed)` stream, n full replays, lock held throughout.
    `model` overrides the sampled network (default: the session's)."""
    from plo5bp.network import obs_adapter

    net = ts.model if model is None else model
    adapt = ts._obs_adapt if model is None else obs_adapter(model)
    with T._TORCH_RNG_LOCK:
        torch.manual_seed(node_seed)
        total = 0.0
        live = []
        for _ in range(n):
            env = BombPotEnv(h.config, ev_runout_samples=32)
            obs, info = env.reset(h.seed, h.button)
            for a in prefix:
                obs, _, _, info = env.step_hybrid(a["gate"], a["chips"])
            obs, rewards, done, info = env.step_hybrid(gate, chips)
            if done:
                total += float(rewards[h.hero_seat])
            else:
                live.append([env, obs, info])
        while live:
            obs_b = torch.from_numpy(adapt(np.stack([x[1] for x in live])))
            gm_b = torch.from_numpy(np.stack([x[2].gate_mask for x in live]))
            sizing_b = torch.from_numpy(
                np.stack([sizing_from_info(x[2]) for x in live])
            )
            with torch.no_grad():
                out = net.act(obs_b, gm_b, sizing_b, deterministic=False)
            nxt = []
            for i, x in enumerate(live):
                obs2, rewards, done, info2 = x[0].step_hybrid(
                    int(out.gate[i].item()), int(out.chips[i].item())
                )
                if done:
                    total += float(rewards[h.hero_seat])
                else:
                    nxt.append([x[0], obs2, info2])
            live = nxt
        return total / n / h.config.bb


def _live_hand(trainer_factory, **kw):
    ts = trainer_factory(
        model_cls=ActorCriticV2, seats_mode="fixed", seats_fixed=4,
        mc_rollouts=6, stack_bb=100.0, **kw,
    )
    for _ in range(8):
        ts.new_hand()
        if not ts.hand.terminal:
            return ts
    pytest.skip("never reached a live hero node")


def test_rollout_ev_bit_identical_to_the_hold_the_lock_version(T, trainer_factory):
    """Determinism pin (F7): shrinking the lock must not change one bit."""
    ts = _live_hand(trainer_factory)
    h = ts.hand
    prefix = list(h.action_log)
    info = h.last_info
    cands = [(GATE_CHECK_CALL, 0)]
    if bool(info.gate_mask[GATE_FOLD]):
        cands.append((GATE_FOLD, 0))  # terminal-at-candidate short circuit
    if bool(info.gate_mask[GATE_RAISE]) and int(info.min_raise_chips) > 0:
        cands.append((GATE_RAISE, int(info.min_raise_chips)))
    for gate, chips in cands:
        for seed in (777, 778, 2**62 + 5):
            want = _reference_rollout_ev(T, ts, h, prefix, gate, chips, 6, seed)
            got = ts._rollout_ev(h, prefix, gate, chips, 6, seed)
            assert got == want, (gate, chips, seed)
            assert ts._rollout_ev(h, prefix, gate, chips, 6, seed) == got


def test_rollout_ev_survives_foreign_rng_traffic(T, trainer_factory):
    """Another session hammering the GLOBAL torch RNG between our depths
    (what opponent sampling for other users does) must not perturb the
    common-random-numbers stream."""
    ts = _live_hand(trainer_factory)
    h = ts.hand
    prefix = list(h.action_log)
    want = _reference_rollout_ev(T, ts, h, prefix, GATE_CHECK_CALL, 0, 6, 4242)

    stop = threading.Event()
    grabbed = {"n": 0}

    def noise():
        i = 0
        while not stop.is_set():
            with T._TORCH_RNG_LOCK:
                torch.manual_seed(i)
                torch.rand(3)
                grabbed["n"] += 1
            i += 1
            time.sleep(0)

    th = threading.Thread(target=noise, daemon=True)
    th.start()
    try:
        for _ in range(5):
            assert ts._rollout_ev(h, prefix, GATE_CHECK_CALL, 0, 6, 4242) == want
    finally:
        stop.set()
        th.join(timeout=5)
    assert grabbed["n"] > 0


def test_rollout_ev_resumes_its_own_stream_after_a_clobber(T, trainer_factory, monkeypatch):
    """Deterministic version of the above: reseed the global RNG right after
    EVERY per-depth snapshot (i.e. in every gap where the lock is released).
    The next depth must resume the snapshot, not the clobbered stream."""
    ts = _live_hand(trainer_factory)
    h = ts.hand
    prefix = list(h.action_log)
    want = _reference_rollout_ev(T, ts, h, prefix, GATE_CHECK_CALL, 0, 6, 31337)

    real_snapshot = T._snapshot_rng_state
    depths = {"n": 0}

    def snapshot_then_clobber(device):
        state = real_snapshot(device)
        depths["n"] += 1
        torch.manual_seed(987654321 + depths["n"])  # "another user's" reseed
        torch.rand(7)
        return state

    monkeypatch.setattr(T, "_snapshot_rng_state", snapshot_then_clobber)
    got = ts._rollout_ev(h, prefix, GATE_CHECK_CALL, 0, 6, 31337)
    assert depths["n"] >= 2, "needs a multi-depth rollout to mean anything"
    assert got == want


def test_env_work_never_runs_under_the_rng_lock(T, trainer_factory, monkeypatch):
    ts = _live_hand(trainer_factory)
    h = ts.hand
    seen = {"reset": 0, "step": 0}

    class _Probe(BombPotEnv):
        def reset(self, *a, **k):
            assert not T._TORCH_RNG_LOCK.locked(), "env reset under the RNG lock"
            seen["reset"] += 1
            return super().reset(*a, **k)

        def step_hybrid(self, *a, **k):
            assert not T._TORCH_RNG_LOCK.locked(), "env step under the RNG lock"
            seen["step"] += 1
            return super().step_hybrid(*a, **k)

    monkeypatch.setattr(T, "BombPotEnv", _Probe)
    ts._rollout_ev(h, list(h.action_log), GATE_CHECK_CALL, 0, 4, 99)
    assert seen["reset"] == 4 and seen["step"] > 4


# --- F9 / F10 / latent: router + persistence ----------------------------------------


def _router_app(T, tmp_path, monkeypatch, model=None):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setenv("PLO5BP_TRAINER_STATS", str(tmp_path / "router.json"))
    if model is None:
        torch.manual_seed(0)
        model = ActorCriticV2(hidden_dim=32, num_layers=1).eval()
    router = T.create_trainer_router(model, CPU)
    ts = router.trainer_session
    ts.set_settings(T.TrainerSettings(**{
        **ts.settings.model_dump(),
        "seats_mode": "fixed", "seats_fixed": 3, "mc_rollouts": 0,
    }))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app), ts


def test_act_without_a_hand_is_409_and_deals_nothing(T, tmp_path, monkeypatch):
    client, ts = _router_app(T, tmp_path, monkeypatch)
    assert ts.hand is None
    r = client.post("/trainer/act", json={"gate": "check_call", "chips": None})
    assert r.status_code == 409, r.text
    assert ts.hand is None and ts.hand_no == 0  # no implicit (unmetered) deal
    # GET /state still deals implicitly; acting then works.
    s = client.get("/trainer/state").json()["state"]
    assert ts.hand is not None and ts.hand_no == 1
    if s["trainer"]["hand_active"]:
        gate = "check_call" if s["legal"]["check_call"] else "fold"
        assert client.post(
            "/trainer/act", json={"gate": gate, "chips": None}
        ).status_code == 200
    # A format switch drops the hand: the stale action is refused again.
    ts.hand = None
    assert client.post(
        "/trainer/act", json={"gate": "check_call", "chips": None}
    ).status_code == 409
    assert ts.hand is None


def test_experimental_format_settings_reload(T, tmp_path):
    torch.manual_seed(0)
    m = ActorCriticV2(hidden_dim=32, num_layers=1).eval()
    path = tmp_path / "s.json"
    ts = T.TrainerSession(m, CPU, stats_path=path)
    ts.set_format(T.FORMAT_EXPERIMENTAL, m, None)
    ts.set_settings(T.TrainerSettings(**{
        **ts.settings.model_dump(), "stack_bb": 77.0, "mc_rollouts": 3,
    }))
    ts._persist()
    ts2 = T.TrainerSession(m, CPU, stats_path=path)
    assert ts2.settings_by_variant[T.FORMAT_EXPERIMENTAL].stack_bb == 77.0
    ts2.set_format(T.FORMAT_EXPERIMENTAL, m, None)
    assert ts2.settings.stack_bb == 77.0 and ts2.settings.mc_rollouts == 3


def test_public_router_fails_closed_without_a_user_session(T, tmp_path, monkeypatch):
    client, default_ts = _router_app(T, tmp_path, monkeypatch)
    monkeypatch.setenv("PLO5BP_PUBLIC", "1")
    try:
        T.set_session_resolver(lambda: None)  # request with no signed-in user
        for method, path, body in (
            ("get", "/trainer/state", None),
            ("post", "/trainer/new_hand", {}),
            ("post", "/trainer/act", {"gate": "check_call", "chips": None}),
        ):
            r = getattr(client, method)(path, **({"json": body} if body is not None else {}))
            assert r.status_code == 401, (path, r.status_code, r.text)
        T.set_session_resolver(None)  # public build, resolver never installed
        assert client.get("/trainer/state").status_code == 500
        # the shared default session was never touched
        assert default_ts.hand is None and default_ts.hand_no == 0
    finally:
        T.set_session_resolver(None)
    # Local build (flag off): the default session serves, as always.
    monkeypatch.delenv("PLO5BP_PUBLIC")
    assert client.get("/trainer/state").status_code == 200
    assert default_ts.hand is not None


def test_persist_retries_transient_permission_error(T, trainer_factory, monkeypatch, caplog):
    ts = trainer_factory(mc_rollouts=9)
    real_replace = T.os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError(13, "sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(T.os, "replace", flaky)
    monkeypatch.setattr(T, "_PERSIST_RETRY_SLEEP_S", 0.0)
    ts._persist()
    assert calls["n"] == 3
    saved = json.loads(ts.stats_path.read_text())
    assert saved["settings_by_format"][ts.variant]["mc_rollouts"] == 9

    # Still failing after every retry: logged, never raised.
    calls["n"] = -10**6
    with caplog.at_level("WARNING", logger="plo5bp.ui.trainer"):
        monkeypatch.setattr(
            T.os, "replace",
            lambda s, d: (_ for _ in ()).throw(PermissionError(13, "locked")),
        )
        ts._persist()
    assert any("failed to persist" in r.message for r in caplog.records)


def test_actor_commit_chips_is_exact_with_blinds_and_a_short_bb(T, tmp_path):
    """Raise chips are raise-BY deltas; the client shows raise-TO totals =
    chips + `actor_commit_chips`. The server sends the acting seat's exact
    pre-action street commit (the client's own derivation from the action
    list is off whenever a blind posted short). Pinned against the ENGINE:
    after a hero raise, street_commit[hero] == actor_commit_chips + chips —
    on NLH tables where seat 0 can only post a SHORT big blind."""
    ppo_nlh, _ = _nlh_nets()
    ts = T.TrainerSession(ppo_nlh, CPU, stats_path=tmp_path / "s.json")
    ts.set_format(VARIANT_NLH, ppo_nlh, None)
    # seat 0: 1.2bb -> 0.7bb after the 0.5bb ante -> posts a SHORT (all-in) BB.
    per_seat = [(1.2, 1.2)] + [(100.0, 100.0)] * 5
    ts.set_settings(T.TrainerSettings(**{
        **ts.settings.model_dump(),
        "seats_mode": "fixed", "seats_fixed": 3, "mc_rollouts": 0,
        "stacks_mode": "per_seat", "stacks_per_seat_bb": per_seat,
    }))
    raises_checked = blind_commit_seen = short_bb_hands = 0
    for seed in range(40):
        ts.rng = np.random.default_rng(seed)
        ts.new_hand()
        h = ts.hand
        if h.terminal or h.hero_seat == 0:
            continue
        raw0 = h.last_info.raw_obs
        short_bb = int(raw0["bb_seat"]) == 0
        if short_bb:
            # the fixture really is a short post: BB has < 1bb on the street
            assert 0 < int(raw0["street_commit"][0]) < h.config.bb
        info = h.last_info
        raw = info.raw_obs
        hero = h.hero_seat
        street_name = ts.project_state()["street"]
        commit_before = int(raw["street_commit"][hero])
        to_call_before = ts.project_state()["to_call_chips"]
        can_raise = bool(info.gate_mask[GATE_RAISE]) and int(info.min_raise_chips) > 0
        if can_raise:
            chips = int(info.min_raise_chips)
            frames = ts.act("raise", chips)
        else:
            chips = 0
            frames = ts.act("check_call" if bool(info.gate_mask[GATE_CHECK_CALL])
                            else "fold", None)
        d = h.decisions[0]
        fb = frames[0]["trainer"]["feedback"]
        assert d.actor_commit_chips == commit_before
        assert fb["actor_commit_chips"] == commit_before
        assert fb["to_call_chips"] == to_call_before
        assert fb["street"] == street_name == "preflop"
        blind_commit_seen += commit_before > 0
        short_bb_hands += short_bb
        f0 = frames[0]
        if can_raise and f0["terminal"] is None and f0["street"] == street_name:
            # ENGINE truth: the raise-TO total the client will display.
            after = f0["seats"][hero]["committed_this_street_chips"]
            assert after == commit_before + chips
            raises_checked += 1
        # finish the hand, then the review must carry the same exact numbers
        steps = 0
        while not h.terminal and steps < 60:
            s = ts.project_state()
            ts.act("check_call" if s["legal"]["check_call"] else "fold", None)
            steps += 1
        assert h.terminal
        state = ts.review_at_node(d.action_log_idx)
        rv = state["trainer"]["review"]
        assert rv["node_current"]["actor_commit_chips"] == commit_before
        assert rv["decisions"][0]["actor_commit_chips"] == commit_before
        assert rv["decisions"][0]["to_call_chips"] == to_call_before
        assert rv["current"]["actor_commit_chips"] == \
            h.decisions[rv["decision"]].actor_commit_chips
        # every node (villains too): payload == the replayed engine state
        for ni in range(len(h.action_log)):
            st = ts.review_at_node(ni)
            nc = st["trainer"]["review"]["node_current"]
            assert nc["actor_commit_chips"] == \
                st["seats"][nc["seat"]]["committed_this_street_chips"]
    assert raises_checked > 0, "never checked a hero raise against the engine"
    assert blind_commit_seen > 0, "hero never acted from a blind (commit > 0)"
    assert short_bb_hands > 0, "the short-BB table never occurred"


def test_hero_auto_checks_are_flagged_everywhere(trainer_factory):
    """A covering bet called for all but dust: the engine keeps asking hero
    to check the turn/river. Those are auto-checks — tagged `auto` in the
    action log, the frames and the review, with NO graded fields."""
    from plo5bp.gto.backend import PpoSolverHost

    per_seat = [(100.0, 100.0), (5.0004, 5.0004)] + [(20.0, 20.0)] * 4
    for seed in range(30):
        ts = trainer_factory(
            rng_seed=seed, stats_name=f"auto{seed}.json",
            seats_mode="fixed", seats_fixed=2, mc_rollouts=0,
            stacks_mode="per_seat", stacks_per_seat_bb=per_seat,
        )

        class _Caller(PpoSolverHost):
            def act(self, obs, info, *, deterministic=False, rng_seed=None):
                return GATE_CHECK_CALL, 0

        ts.set_backend(_Caller(model=ts.backend.model, device=ts.backend.device))
        ts.new_hand()
        h = ts.hand
        if h.hero_seat != 0 or h.terminal:
            continue
        frames: list[dict] = []
        steps = 0
        while not h.terminal and steps < 12:
            raw = h.last_info.raw_obs
            s = ts.project_state()
            target = int(raw["stacks"][1]) - 4  # leave the caller dust
            lo, hi = s["raise_bounds"]["min_chips"], s["raise_bounds"]["max_chips"]
            if s["legal"]["raise"] and lo <= target <= hi:
                frames += ts.act("raise", target)
            else:
                frames += ts.act("check_call", None)
            steps += 1
        assert h.terminal
        hero_autos = [
            i for i, a in enumerate(h.action_log)
            if a["seat"] == h.hero_seat and a.get("auto")
        ]
        if not hero_autos:
            continue

        # action log: only moot entries carry the flag; hero's real decisions don't
        decided = {d.action_log_idx for d in h.decisions}
        assert not decided & set(hero_autos)
        for i in decided:
            assert "auto" not in h.action_log[i]

        # frames: the animation entry says whose action it was and that it was forced
        anim = [f["trainer"]["anim_action"] for f in frames
                if f["trainer"].get("anim_action")]
        hero_anim = [a for a in anim if a["seat"] == h.hero_seat]
        assert hero_anim and all(a["auto"] and a["is_hero"] for a in hero_anim)
        assert all(a["is_hero"] is False for a in anim if a["seat"] != h.hero_seat)
        opp_rows = [r for f in frames for r in f["trainer"]["opp_actions"]]
        assert any(r["is_hero"] and r["auto"] for r in opp_rows)

        # review: node index + node view carry `auto`, graded keys are explicit nulls
        node = hero_autos[0]
        rv = ts.review_at_node(node)["trainer"]["review"]
        assert rv["nodes"][node]["auto"] is True
        assert rv["nodes"][node]["is_hero"] is True
        assert rv["nodes"][node]["decision_idx"] is None
        nc = rv["node_current"]
        assert nc["is_hero"] is True and nc["auto"] is True
        for key in ("category", "marks", "score", "decision_idx", "ev_loss_bb"):
            assert key in nc and nc[key] is None, key
        assert nc["user_label"] == nc["actual_label"] == "Check"
        # a real hero decision is unaffected
        real = ts.review_at_node(h.decisions[0].action_log_idx)
        rnc = real["trainer"]["review"]["node_current"]
        assert rnc["auto"] is False and rnc["category"] == h.decisions[0].category
        return
    raise AssertionError("no hero auto-check occurred in 30 seeds")
