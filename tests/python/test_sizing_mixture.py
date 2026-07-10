"""v5 sizing head: K-component mixture of discretized logistics.

Covers the v5 additions on top of the (unchanged) v2/v4 machinery:
  - `ActorCriticV5._anchor_dist`: the marginal equals sum_k w_k * P_k
    exactly (discrete mixture => closed form, no MC), legality masking
    zeroes illegal anchors, the epsilon weight floor holds, and the
    head can express the interior+end menu v4 provably cannot.
  - wiring/sniffing (mix_head, head_version 4, K round-trip), the
    bias-driven init spread, act<->evaluate log-prob parity, exact
    marginal entropy, and the B3 detach (entropy bonus cannot pull the
    mix head toward the atoms).
  - PPO update finite with the v5 head; the KL-to-EMA magnet runs on
    BOTH v4 and v5 without the (B,2)-vs-(B,11) masked_fill crash
    (V5_DESIGN.md B1 regression) and EMA state round-trips.
  - the v4->v5 converter is function-preserving on gate/value/refine
    and only smears the anchor marginal by the minor components.
  - CentralCritic Q-aux head: zero-init Q == V, state-dict round-trip,
    pre-v5 critics load unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from plo5bp.actions import GATE_ACTIONS
from plo5bp.config import GameConfig, TrainingConfig
from plo5bp.encoding import OBS_DIM
from plo5bp.network import (
    ANCHOR_COUNT,
    ActorCriticV2,
    ActorCriticV4,
    ActorCriticV5,
    CentralCritic,
    _discretized_logistic_probs,
    build_actor_from_state_dict,
    build_critic_from_state_dict,
    model_class_for_state_dict,
)
from plo5bp.ppo import PPOTrainer
from plo5bp.rollout import Batch, collect_rollout_batched
from plo5bp.selfplay import OpponentPool
from plo5bp.sizing import anchor_grid_torch

_SIZING = torch.tensor([[200, 10000, 5000, 0]], dtype=torch.int64)  # all 11 legal


def _grid(batch: int = 1):
    return anchor_grid_torch(
        _SIZING.expand(batch, 4).contiguous(), ActorCriticV5().anchor_spec
    )


def _raw_for_mu(target: float, count: int = ANCHOR_COUNT) -> float:
    """Invert mu = c + (c+2)*tanh(raw)."""
    c = (count - 1) / 2.0
    return math.atanh((target - c) / (c + 2.0))


# ---- marginal math --------------------------------------------------------

def test_marginal_equals_weighted_component_sum() -> None:
    torch.manual_seed(0)
    m = ActorCriticV5(hidden_dim=32)
    mix_params = torch.randn(7, 9)
    grid = _grid(7)
    probs = m._anchor_dist(mix_params, grid).probs
    mu, s, w = m.mixture_params(mix_params)
    manual = torch.zeros_like(probs)
    for k in range(3):
        p_k = _discretized_logistic_probs(mu[:, k], s[:, k], grid.legal)
        manual += w[:, k, None] * p_k
    assert torch.allclose(probs, manual, atol=1e-6)
    assert torch.allclose(probs.sum(-1), torch.ones(7), atol=1e-5)


def test_weight_floor_holds_under_extreme_logits() -> None:
    m = ActorCriticV5(hidden_dim=32)
    # Third component's logit pushed to -30: without the floor its weight
    # would vanish (dead component, no gradient path back).
    mix_params = torch.zeros(1, 9)
    mix_params[0, 8] = -30.0
    _, _, w = m.mixture_params(mix_params)
    assert float(w.min()) >= m._mix_floor - 1e-6
    assert abs(float(w.sum()) - 1.0) < 1e-5


def test_mixture_expresses_interior_plus_end_menu() -> None:
    # The v4-impossible shape: meaningful mass on interior anchor 3 AND
    # the pot anchor with a dry valley between (V5_DESIGN.md section 2:
    # v4's best joint mass on {33%, pot} without flooding between is
    # ~0.008). Two tight components at indices 3 and 10, equal weight.
    m = ActorCriticV5(hidden_dim=32)
    mix_params = torch.tensor([[
        _raw_for_mu(3.0), _raw_for_mu(10.0), 0.0,   # mu_raw x3
        -20.0, -20.0, -20.0,                        # s_raw -> s = floor
        0.0, 0.0, -30.0,                            # logits: 50/50 + floored
    ]])
    p = m._anchor_dist(mix_params, _grid()).probs[0]
    assert p[3] > 0.30, p
    assert p[10] > 0.35, p
    assert p[6] < 0.05 and p[7] < 0.05 and p[8] < 0.05, p


def test_mixture_marginal_bf16_autocast_is_finite_and_normalized() -> None:
    # The pod trains under torch.autocast(bf16); the CPU harness is f32.
    # Pin the V5-SPECIFIC math (the mixture -> marginal Categorical +
    # its entropy) under bf16: finite, normalized, NO legal anchor
    # zeroed by underflow, entropy finite across legality regimes.
    # (The Beta refine path is inherited v2/v4 code and CPU-bf16 lacks a
    # Dirichlet kernel, so act()'s sampling isn't exercised here — that
    # path is CUDA-proven by months of v4 training.)
    torch.manual_seed(0)
    m = ActorCriticV5(hidden_dim=64).eval()
    B = 96
    obs = torch.randn(B, OBS_DIM)
    gm = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    for sizing in (
        _SIZING.expand(B, 4).contiguous(),                          # all legal
        torch.tensor([[0, 3000, 5000, 3000]], dtype=torch.int64).expand(B, 4).contiguous(),  # short-shove
        torch.tensor([[200, 1500, 5000, 0]], dtype=torch.int64).expand(B, 4).contiguous(),   # capped
    ):
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            _, mp, _, _ = m(obs, gm)
            dist = m._anchor_dist(mp, anchor_grid_torch(sizing, m.anchor_spec))
            probs = dist.probs
            ent = dist.entropy()
        assert torch.isfinite(probs).all()
        assert torch.allclose(probs.float().sum(-1), torch.ones(B), atol=1e-2)
        assert torch.isfinite(ent).all()
        # legal anchors keep positive mass; illegal stay exactly zero
        grid = anchor_grid_torch(sizing, m.anchor_spec)
        assert (probs.float()[grid.legal] > 0).all()
        assert (probs.float()[~grid.legal] == 0).all()


def test_mixture_legality_masking() -> None:
    m = ActorCriticV5(hidden_dim=32)
    sizing = torch.tensor([[200, 10000, 5000, 0]], dtype=torch.int64)
    grid = anchor_grid_torch(sizing, m.anchor_spec)
    legal = grid.legal.clone()
    legal[0, 4] = False
    legal[0, 9] = False

    class _G:
        pass

    g = _G()
    g.legal = legal
    g.refine_ok = grid.refine_ok
    g.chips = grid.chips
    p = m._anchor_dist(torch.randn(1, 9), g).probs[0]
    assert p[4] == 0.0 and p[9] == 0.0
    assert abs(float(p.sum()) - 1.0) < 1e-5


def test_evaluate_anchor_entropy_is_exact_marginal_entropy() -> None:
    torch.manual_seed(1)
    m = ActorCriticV5(hidden_dim=32).eval()
    B = 8
    obs = torch.randn(B, OBS_DIM)
    gm = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = _SIZING.expand(B, 4).contiguous()
    out = m.act(obs, gm, sizing)
    _, _, _, _, anchor_h, _, _, _, *_raw = m.evaluate(
        obs, gm, sizing, out.gate, out.anchor, out.refine_u
    )
    with torch.no_grad():
        _, mix_params, _, _ = m(obs, gm)
        p = m._anchor_dist(mix_params, anchor_grid_torch(sizing, m.anchor_spec)).probs
        manual = -(p * p.clamp_min(1e-12).log()).sum(-1)
    assert torch.allclose(anchor_h, manual, atol=1e-5)


# ---- wiring / init / sniffing --------------------------------------------

def test_v5_head_wiring_and_sniff() -> None:
    m = ActorCriticV5(hidden_dim=32)
    assert m.head_version == 4
    sd = m.state_dict()
    assert "mix_head.weight" in sd and "size_head.weight" not in sd
    assert sd["mix_head.weight"].shape == (9, 32)
    assert "refine_head.weight" in sd
    assert model_class_for_state_dict(sd) is ActorCriticV5
    # Regression guard: v4/v2 still sniff to their own classes.
    assert model_class_for_state_dict(
        ActorCriticV4(hidden_dim=32).state_dict()
    ) is ActorCriticV4
    assert model_class_for_state_dict(
        ActorCriticV2(hidden_dim=32).state_dict()
    ) is ActorCriticV2


def test_v5_builder_round_trips_k() -> None:
    for k in (2, 3):
        src = ActorCriticV5(hidden_dim=32, mixture_k=k)
        rebuilt = build_actor_from_state_dict(src.state_dict(), 32, 2)
        assert isinstance(rebuilt, ActorCriticV5)
        assert rebuilt._mixture_k == k
        obs = torch.randn(3, OBS_DIM)
        gm = torch.ones(3, GATE_ACTIONS, dtype=torch.bool)
        with torch.no_grad():
            a = src(obs, gm)[1]
            b = rebuilt(obs, gm)[1]
        assert torch.equal(a, b)


def test_init_spread_gives_menu_not_coincident_humps() -> None:
    m = ActorCriticV5(hidden_dim=64)
    obs = torch.randn(5, OBS_DIM)
    gm = torch.ones(5, GATE_ACTIONS, dtype=torch.bool)
    with torch.no_grad():
        _, mix_params, _, _ = m(obs, gm)
        mu, s, w = m.mixture_params(mix_params)
    # Zeroed weights -> biases dominate: identical across rows.
    assert torch.allclose(mu[0], mu[-1], atol=1e-6)
    # Locations spread across the ladder; weights uniform.
    assert mu[0, 0] < 1.0 and abs(mu[0, 1] - 5.0) < 1e-4 and mu[0, 2] > 9.0
    assert torch.allclose(w[0], torch.full((3,), 1.0 / 3.0), atol=1e-5)


def test_v5_act_evaluate_finite_and_parity() -> None:
    torch.manual_seed(0)
    m = ActorCriticV5(hidden_dim=32).eval()
    B = 16
    obs = torch.randn(B, OBS_DIM)
    gate_mask = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = _SIZING.expand(B, 4).contiguous()

    out = m.act(obs, gate_mask, sizing, deterministic=False)
    assert out.anchor.min() >= 0 and out.anchor.max() < ANCHOR_COUNT
    assert torch.isfinite(out.log_prob).all()

    lp, ent, val, gate_h, anchor_h, beta_h, glp, alp, *_raw = m.evaluate(
        obs, gate_mask, sizing, out.gate, out.anchor, out.refine_u
    )
    for t in (lp, ent, val, gate_h, anchor_h, beta_h):
        assert torch.isfinite(t).all()
    assert torch.allclose(out.log_prob, lp, atol=1e-5), (out.log_prob - lp).abs().max()


# ---- B3: entropy bonus cannot pull the size head toward the atoms --------

def test_beta_h_weight_detached_on_v5_not_v4() -> None:
    torch.manual_seed(0)
    B = 8
    obs = torch.randn(B, OBS_DIM)
    gm = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = _SIZING.expand(B, 4).contiguous()

    def size_grad_from_beta_term(model, head_param):
        out = model.act(obs, gm, sizing)
        res = model.evaluate(obs, gm, sizing, out.gate, out.anchor, out.refine_u)
        beta_h_eff = res[5]
        model.zero_grad(set_to_none=True)
        beta_h_eff.sum().backward()
        g = head_param.grad
        return 0.0 if g is None else float(g.abs().max())

    v4 = ActorCriticV4(hidden_dim=32)
    v5 = ActorCriticV5(hidden_dim=32)
    # Break v5's zero-weight init so a graph connection would show up.
    with torch.no_grad():
        v5.mix_head.weight.normal_(0, 0.05)
    # v4 (legacy): the anchor-prob weighting flows into size_head — the
    # verified end-anchor-subsidy path (kept for byte-identical resumes).
    assert size_grad_from_beta_term(v4, v4.size_head.weight) > 0.0
    # v5: detached — the entropy bonus's Beta term cannot move the mixture.
    assert size_grad_from_beta_term(v5, v5.mix_head.weight) == 0.0


# ---- PPO integration ------------------------------------------------------

def _small_setup(model_cls, **cfg_kwargs):
    game_cfg = GameConfig(num_seats=4)
    train_cfg = TrainingConfig(
        num_envs=4, rollout_length=128, hidden_dim=32, ppo_epochs=2,
        batch_size=64, critic_hidden_dim=64, critic_num_blocks=1,
        **cfg_kwargs,
    )
    model = model_cls(hidden_dim=train_cfg.hidden_dim)
    critic = CentralCritic(
        hidden_dim=train_cfg.critic_hidden_dim,
        num_blocks=train_cfg.critic_num_blocks,
    )
    return game_cfg, train_cfg, model, critic


def test_v5_ppo_update_finite() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg, train_cfg, model, critic = _small_setup(ActorCriticV5)
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    pool = OpponentPool(capacity=1)
    rng = np.random.default_rng(0)
    model.eval()
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    for name in ("policy_loss", "value_loss", "entropy", "approx_kl",
                 "gate_entropy", "anchor_entropy", "beta_entropy"):
        assert np.isfinite(getattr(stats, name)), f"non-finite {name}: {stats}"


@pytest.mark.parametrize("model_cls", [ActorCriticV4, ActorCriticV5])
def test_kl_anchor_magnet_runs_on_ordinal_heads(model_cls) -> None:
    # Regression for V5_DESIGN.md B1: any --kl-anchor-coef > 0 run used to
    # crash on v4 heads at the first minibatch ((B, 2) size params
    # masked_fill'ed against (B, 11) grid.legal).
    torch.manual_seed(0)
    np.random.seed(0)
    game_cfg, train_cfg, model, critic = _small_setup(
        model_cls, kl_anchor_coef=0.05
    )
    trainer = PPOTrainer(model, train_cfg, critic=critic)
    assert trainer._ref is not None
    pool = OpponentPool(capacity=1)
    rng = np.random.default_rng(0)
    model.eval()
    batch = collect_rollout_batched(model, pool, game_cfg, train_cfg, rng, critic=critic)
    model.train()
    stats = trainer.update(batch, rng)
    assert np.isfinite(stats.kl_anchor)
    # EMA reference persistence round-trip.
    ref_sd = trainer.ref_state_dict()
    assert ref_sd is not None
    trainer2 = PPOTrainer(model_cls(hidden_dim=train_cfg.hidden_dim), train_cfg)
    trainer2.load_ref_state_dict(ref_sd)
    for a, b in zip(trainer2._ref.state_dict().values(), ref_sd.values()):
        assert torch.equal(a, b)


# ---- magnet memory fix: reuse evaluate's forward ---------------------------

def test_kl_reference_reuse_matches_fresh_forward() -> None:
    # The magnet memory fix reuses evaluate()'s current-model forward for
    # KL(current||ref) instead of a second forward. In f32 (no autocast)
    # the reused path must equal the fresh-forward path bit-for-bit.
    from plo5bp.ppo import _kl_to_reference
    import copy

    torch.manual_seed(0)
    m = ActorCriticV5(hidden_dim=32).eval()
    ref = copy.deepcopy(m).eval()
    with torch.no_grad():  # perturb ref so KL > 0
        ref.mix_head.weight.add_(torch.randn_like(ref.mix_head.weight) * 0.2)
        ref.gate_head.weight.add_(torch.randn_like(ref.gate_head.weight) * 0.2)
    B = 40
    obs = torch.randn(B, OBS_DIM)
    gm = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = _SIZING.expand(B, 4).contiguous()
    out = m.act(obs, gm, sizing)
    mb = Batch(
        obs=obs, gate_masks=gm, gate_actions=out.gate,
        raise_chips=out.chips, sizing=sizing, anchor_actions=out.anchor,
        refine_u=out.refine_u,
        opp_holes=torch.full((B, 5, 5), 255, dtype=torch.uint8),
        log_probs=out.log_prob.detach(), values=torch.zeros(B),
        returns=torch.zeros(B), advantages=torch.zeros(B),
        old_gate_logp=torch.zeros(B), old_anchor_logp=torch.zeros(B),
    )
    # fresh path (cur=None → second forward)
    kl_fresh = _kl_to_reference(m, ref, mb, cur=None)
    # reuse path: pass evaluate's raw head outputs
    res = m.evaluate(obs, gm, sizing, out.gate, out.anchor, out.refine_u)
    cur = (res[8], res[9], res[10])  # gate_logits, anchor_out, refine
    kl_reuse = _kl_to_reference(m, ref, mb, cur=cur)
    assert torch.allclose(kl_fresh, kl_reuse, atol=1e-5), (kl_fresh, kl_reuse)
    assert float(kl_reuse.detach()) > 1e-4  # actually nonzero (ref perturbed)


# ---- converter -------------------------------------------------------------

def _load_converter():
    import importlib.util
    from pathlib import Path

    path = (
        Path(__file__).resolve().parents[2] / "scripts" / "convert_v4_to_v5.py"
    )
    spec = importlib.util.spec_from_file_location("convert_v4_to_v5", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_converter_is_function_preserving() -> None:
    convert = _load_converter().convert

    torch.manual_seed(0)
    v4 = ActorCriticV4(hidden_dim=32)
    critic = CentralCritic(hidden_dim=64, num_blocks=1)
    ckpt = {
        "model": v4.state_dict(),
        "critic": critic.state_dict(),
        "config": {"hidden_dim": 32, "num_layers": 2},
        "head_version": 3,
        "variant": "plo5_double_bomb",
        "update_counter": 917,
        "anneal_tier_ent": {"clubgg": 0.1},
        "pool_member_updates": [905, 910, 915],
    }
    out, log = convert(ckpt, mixture_k=3, w0_logit=3.0)
    assert out["head_version"] == 4
    for stale in ("update_counter", "anneal_tier_ent", "pool_member_updates"):
        assert stale not in out
    v5 = build_actor_from_state_dict(out["model"], 32, 2)
    assert isinstance(v5, ActorCriticV5)

    B = 32
    obs = torch.randn(B, OBS_DIM)
    gm = torch.ones(B, GATE_ACTIONS, dtype=torch.bool)
    sizing = _SIZING.expand(B, 4).contiguous()
    grid = anchor_grid_torch(sizing, v5.anchor_spec)
    with torch.no_grad():
        g4, a4, r4, val4 = v4(obs, gm)
        g5, a5, r5, val5 = v5(obs, gm)
        p4 = v4._anchor_dist(a4, grid).probs
        p5 = v5._anchor_dist(a5, grid).probs
        _, _, w = v5.mixture_params(a5)
    # Gate / refine / value: exactly preserved.
    assert torch.allclose(g4, g5, atol=1e-6)
    assert torch.allclose(r4, r5, atol=1e-6)
    assert torch.allclose(val4, val5, atol=1e-6)
    # Component 0 dominant; marginal within the minor-component smear.
    assert float(w[:, 0].min()) > 0.9
    assert float((p4 - p5).abs().max()) < 0.12
    # Critic gained the zero-init adv head and still loads.
    c2 = build_critic_from_state_dict(out["critic"])
    assert c2.q_actions == 2 + ANCHOR_COUNT
    assert float(out["critic"]["adv_head.weight"].abs().max()) == 0.0
    # Converting a v5 checkpoint is refused.
    with pytest.raises(SystemExit):
        convert(out)


# ---- UI serving smoke -------------------------------------------------------

def test_v5_recommendation_serves_mixture_block(monkeypatch) -> None:
    # The study endpoint must serve a v5 model end-to-end: head_version 4,
    # the 11-bin anchors histogram (mixture marginal), and the additive
    # `mixture` block with per-component (mu, s, w).
    from starlette.testclient import TestClient

    import plo5bp.ui.server as srv

    m = ActorCriticV5(hidden_dim=32).eval()
    fmt = srv.FORMATS["plo5_double_bomb"]
    monkeypatch.setitem(fmt, "model", m)
    monkeypatch.setitem(fmt, "adapter", lambda obs: obs)
    c = TestClient(srv.app)
    s = c.post("/format", json={"format": "plo5_double_bomb"}).json()["state"]
    r = c.post("/cards", json={
        "hero_hole": [0, 1, 2, 3, 4],
        "flop_a": [5, 6, 7], "flop_b": [8, 9, 10],
        "turn": [None, None], "river": [None, None],
    })
    assert r.status_code == 200
    s = r.json()["state"]
    for _ in range(12):
        if s["actor"] == s["hero_seat"] or s["terminal"]:
            break
        s = c.post("/action", json={"gate": "check_call"}).json()["state"]
    assert s["actor"] == s["hero_seat"]
    rec = s["recommendation"]
    assert rec is not None and rec["head_version"] == 4
    assert rec["anchor_count"] == ANCHOR_COUNT
    mix = rec["mixture"]
    assert mix is not None
    assert len(mix["mu"]) == 3 and len(mix["s"]) == 3 and len(mix["w"]) == 3
    assert abs(sum(mix["w"]) - 1.0) < 1e-3
    # Leave the module-global session on PLO5 defaults for other tests.
    c.post("/format", json={"format": "plo5_double_bomb"})


# ---- EMA serving ------------------------------------------------------------

def test_serve_ema_gated_and_graceful(tmp_path, monkeypatch) -> None:
    # PLO5BP_SERVE_EMA=1 serves the model_ema actor (smoother/less
    # exploitable, for the study tool); off/absent → last iterate;
    # missing model_ema → graceful fallback, never a crash.
    import plo5bp.ui.server as srv

    v5 = ActorCriticV5(hidden_dim=32)
    ema = {k: v.clone() for k, v in v5.state_dict().items()}
    with torch.no_grad():
        ema["mix_head.weight"] = ema["mix_head.weight"] + 5.0  # unambiguous
    crit = CentralCritic(obs_dim=OBS_DIM, hidden_dim=64, num_blocks=1)
    ck = tmp_path / "v5.pt"
    torch.save({
        "model": v5.state_dict(), "critic": crit.state_dict(), "model_ema": ema,
        "config": {"hidden_dim": 32, "num_layers": 2}, "head_version": 4,
        "variant": "plo5_double_bomb", "anchor_count": 11,
    }, ck)
    monkeypatch.setenv("PLO5BP_CHECKPOINT", str(ck))

    monkeypatch.delenv("PLO5BP_SERVE_EMA", raising=False)
    m_last, ok1 = srv._load_model("plo5_double_bomb")
    monkeypatch.setenv("PLO5BP_SERVE_EMA", "1")
    m_ema, ok2 = srv._load_model("plo5_double_bomb")
    assert ok1 and ok2
    delta = float((m_ema.mix_head.weight - m_last.mix_head.weight).abs().max())
    assert delta > 1.0, f"EMA weights not served (delta={delta})"

    # SERVE_EMA=1 but no model_ema key → falls back to last iterate, no crash.
    torch.save({
        "model": v5.state_dict(), "critic": crit.state_dict(),
        "config": {"hidden_dim": 32, "num_layers": 2}, "head_version": 4,
        "variant": "plo5_double_bomb", "anchor_count": 11,
    }, ck)
    _, ok3 = srv._load_model("plo5_double_bomb")
    assert ok3


# ---- CentralCritic Q-aux head ---------------------------------------------

def test_critic_q_head_zero_init_and_round_trip() -> None:
    torch.manual_seed(0)
    c = CentralCritic(hidden_dim=64, num_blocks=1, q_actions=13)
    obs = torch.randn(4, OBS_DIM)
    opp = torch.zeros(4, 5 * 52)
    v, q = c.q_values(obs, opp)
    assert q.shape == (4, 13)
    assert torch.allclose(q, v.detach()[:, None].expand(4, 13), atol=1e-6)
    assert torch.allclose(v, c(obs, opp))
    rebuilt = build_critic_from_state_dict(c.state_dict())
    assert rebuilt.q_actions == 13
    # Pre-v5 critics (no adv_head) still load with q_actions == 0.
    legacy = CentralCritic(hidden_dim=64, num_blocks=1)
    assert build_critic_from_state_dict(legacy.state_dict()).q_actions == 0
