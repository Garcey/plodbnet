"""The live control file (`plo5bp.train.control._apply_anneal_control`):
retune a running trainer without a restart.

Ported from tests/python/test_entropy_anneal.py when the block-rotation
entropy anneal it also covered was retired (2026-09-28, ML-030); the control
file's `step` key (the anneal's decrement) is now an unknown key. More edge
cases (bad values, unknown keys, malformed JSON, the entropy broadcast) are in
test_review_training_train.py.
"""
from __future__ import annotations

from plo5bp.ppo import PPOTrainer
from plo5bp.train import control as train


class _StubTrainer:
    target_kl = 2.0
    kl_hard = 10.0
    sizing_entropy_scale = 1.0
    _clip_room_mid = 0.05
    _clip_room_ext = 0.10
    _q_fold_sup = 1.0
    # The real live-control surface (ML-047), on this attribute stub.
    _LIVE_ATTRS = PPOTrainer._LIVE_ATTRS
    LIVE_KEYS = PPOTrainer.LIVE_KEYS
    live_value = PPOTrainer.live_value
    apply_live_control = PPOTrainer.apply_live_control


def test_q_fold_sup_is_live():
    # 2026-07-11 audit #2: the fold-supervision weight must be retunable
    # live (it was drowned at 1.0 — bb² scale mismatch vs the taken MSE).
    apply = train._apply_anneal_control
    tr = _StubTrainer()
    apply('{"q_fold_sup_coef": 15.0}', None, {"clubgg": 0.2}, 3e-4, 0.2, 0.2, trainer=tr)
    assert tr._q_fold_sup == 15.0
    tr2 = _StubTrainer()
    apply('{"q_fold_sup_coef": "loud"}', None, {"clubgg": 0.2}, 3e-4, 0.2, 0.2, trainer=tr2)
    assert tr2._q_fold_sup == 1.0


def test_clip_rooms_are_live():
    # 2026-07-11: the prob-dependent clip's mid band IS the per-update KL
    # ceiling at mixed gates, so it must be live-tunable like lr/ent.
    apply = train._apply_anneal_control
    tr = _StubTrainer()
    tier_ent = {"clubgg": 0.2}
    raw = '{"clip_room_mid": 0.07, "clip_room_ext": 0.12}'
    last, lr, ent, entd = apply(raw, None, tier_ent, 3e-4, 0.2, 0.2, trainer=tr)
    assert tr._clip_room_mid == 0.07 and tr._clip_room_ext == 0.12
    assert last == raw
    # Bad value -> the whole edit is ignored, attributes untouched.
    tr2 = _StubTrainer()
    apply('{"clip_room_mid": "wide"}', None, tier_ent, 3e-4, 0.2, 0.2, trainer=tr2)
    assert tr2._clip_room_mid == 0.05
    # Without a trainer handle the keys are inert.
    apply(raw, None, tier_ent, 3e-4, 0.2, 0.2, trainer=None)


def test_tiers_lr_and_flat_entropy():
    apply = train._apply_anneal_control
    tier_ent = {"clubgg": 0.09, "deep": 0.15}
    lr0, ent0, entd0 = 3e-4, 0.40, 0.45

    # No file content -> everything unchanged. Returns
    # (applied_content, live_lr, live_ent, live_ent_deep).
    last, lr, ent, entd = apply(None, None, tier_ent, lr0, ent0, entd0)
    assert last is None and lr == lr0 and ent == ent0 and entd == entd0

    # Tier override applies in place; unknown tiers ignored; content tracked.
    raw = '{"tier_ent": {"deep": 0.08, "bogus": 1.0}}'
    last, lr, ent, entd = apply(raw, last, tier_ent, lr, ent, entd)
    assert tier_ent == {"clubgg": 0.09, "deep": 0.08} and last == raw
    tier_ent["deep"] = 0.5
    last, lr, ent, entd = apply(raw, last, tier_ent, lr, ent, entd)  # same content: no-op
    assert tier_ent["deep"] == 0.5

    last, lr, ent, entd = apply('{"lr": 0.0001, "tier_ent": {"clubgg": 0.05}}',
                                last, tier_ent, lr, ent, entd)
    assert lr == 0.0001 and tier_ent["clubgg"] == 0.05

    # The flat coefs are independent of each other.
    last, lr, ent, entd = apply('{"entropy_coef": 0.38}', last, tier_ent, lr, ent, entd)
    assert ent == 0.38 and entd == 0.45
    last, lr, ent, entd = apply('{"entropy_coef_deep": 0.30}', last, tier_ent, lr, ent, entd)
    assert entd == 0.30 and ent == 0.38

    # target_kl / kl_hard / sizing_entropy_scale mutate the trainer in place.
    trn = _StubTrainer()
    last, lr, ent, entd = apply(
        '{"target_kl": 1.5, "kl_hard": 12.0, "sizing_entropy_scale": 2.5}',
        last, tier_ent, lr, ent, entd, trainer=trn,
    )
    assert (trn.target_kl, trn.kl_hard, trn.sizing_entropy_scale) == (1.5, 12.0, 2.5)

    # Malformed JSON: ignored and not marked applied (a half-written save is
    # retried next loop); every field comes back unchanged.
    out = apply('{"lr": 0.0', last, tier_ent, lr, ent, entd)
    assert out == (last, lr, ent, entd)


def test_retired_anneal_step_key_is_skipped_rest_applies(capsys):
    train._CONTROL_WARNED.clear()
    tier_ent = {"deep": 0.15}
    raw = '{"step": 0.003, "tier_ent": {"deep": 0.1}}'
    last, _lr, _e, _ed = train._apply_anneal_control(raw, None, tier_ent, 3e-4, 0.2, 0.2)
    assert last == raw and tier_ent == {"deep": 0.1}
    assert "unknown key(s) ['step']" in capsys.readouterr().out


def test_wrong_shape_json_is_ignored():
    # C1: valid JSON that is NOT a dict (or whose tier_ent is not a dict) must be
    # ignored like malformed JSON, not crash the trainer.
    base = {"clubgg": 0.09, "deep": 0.15}
    for bad in (
        '[{"lr": 0.0001}]', '"0.003"', '0.003', 'true', 'null',
        '{"tier_ent": ["deep", 0.08]}', '{"tier_ent": "deep"}',
    ):
        tier_ent = dict(base)
        out = train._apply_anneal_control(bad, None, tier_ent, 3e-4, 0.40, 0.45)
        assert out == (None, 3e-4, 0.40, 0.45), bad
        assert tier_ent == base, bad
