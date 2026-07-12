"""C2 (2026-07-10): --v6 preset resolution via None-sentinel defaults.

The old implementation compared each covered flag against parser.get_default,
which cannot distinguish "flag not passed" from "explicitly passed at the
default value" — so `--v6 --advantage-estimator gae` silently trained vrpo and
`--v6 --q-aux-coef 0` silently trained the Q head at 0.5: a mislabeled
ablation, the project's documented worst failure mode. The rework gives the
covered flags default=None sentinels (booleans via BooleanOptionalAction, so
--no-<flag> can force a feature off under --v6) and resolves them in the pure
module-level `_apply_v6_preset`. These are the first tests of the
preset-resolution logic.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

_TRAIN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train.py"


def _load_train():
    spec = importlib.util.spec_from_file_location("train_under_test", _TRAIN_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


train = _load_train()

# The 11 covered flags and their (legacy_default, v6_value) pairs — asserted
# against the module table so this test fails loudly if the preset grows a
# key without gaining coverage here.
EXPECTED_PRESET = {
    "sizing_head": ("anchor", "mixture"),
    "advantage_estimator": ("gae", "vrpo"),
    "q_aux_coef": (0.0, 0.5),
    "q_pooled": (False, True),
    # 15.0 since 2026-07-11 (audit #2): coef 1.0 was drowned at ~4% of the
    # q gradient against the raw-bb² taken-action MSE.
    "q_fold_sup_coef": (0.0, 15.0),
    "torso_norm": (False, True),
    "l2_init_coef": (0.0, 1e-4),
    "agc_clip": (0.0, 0.1),
    "grad_checkpoint": (False, True),
    "value_bins": (0, 51),
    "clip_prob_dependent": (False, True),
}


def _ns(v6: bool, **overrides) -> argparse.Namespace:
    """A namespace as parse_args would produce it: every covered flag None
    (not passed) unless overridden."""
    attrs = {attr: None for attr in EXPECTED_PRESET}
    attrs["v6"] = v6
    attrs.update(overrides)
    return argparse.Namespace(**attrs)


def test_preset_table_matches_expected() -> None:
    assert dict(train._V6_PRESET) == EXPECTED_PRESET


def test_argparse_sentinels_present_in_source() -> None:
    # Drift tripwire (review finding): the resolver is only sound while every
    # covered flag keeps its default=None sentinel in the ACTUAL argparse
    # definition (booleans additionally their BooleanOptionalAction). A future
    # edit restoring a concrete argparse default would re-open the
    # explicit-at-default override bug while the Namespace-based tests above
    # stayed green. Checked at source level because the parser is built inside
    # main() and cannot be constructed standalone.
    src = _TRAIN_PATH.read_text(encoding="utf-8")
    booleans = {"q_pooled", "torso_norm", "grad_checkpoint", "clip_prob_dependent"}
    for attr in EXPECTED_PRESET:
        flag = '"--' + attr.replace("_", "-") + '"'
        idx = src.index(flag)  # definition site (quoted-flag form is unique)
        block = src[idx : idx + 400]
        assert "default=None" in block, f"{flag} lost its None sentinel"
        if attr in booleans:
            assert "BooleanOptionalAction" in block, (
                f"{flag} lost BooleanOptionalAction"
            )


def test_v6_alone_applies_every_key() -> None:
    args = _ns(v6=True)
    applied, kept = train._apply_v6_preset(args)
    assert kept == {}
    assert set(applied) == set(EXPECTED_PRESET)
    for attr, (_, v6_value) in EXPECTED_PRESET.items():
        assert getattr(args, attr) == v6_value, attr


def test_v6_explicit_at_default_value_wins() -> None:
    # THE bug case: flags explicitly passed at their legacy-default value must
    # survive --v6 (the old get_default comparison silently overrode them).
    args = _ns(
        v6=True,
        advantage_estimator="gae",  # ablation: v6 substrate, GAE advantages
        q_aux_coef=0.0,             # ablation: Q head frozen
        sizing_head="anchor",       # ablation: v2 head under the v6 substrate
    )
    applied, kept = train._apply_v6_preset(args)
    assert args.advantage_estimator == "gae"
    assert args.q_aux_coef == 0.0
    assert args.sizing_head == "anchor"
    assert kept == {
        "advantage_estimator": "gae",
        "q_aux_coef": 0.0,
        "sizing_head": "anchor",
    }
    # Everything else still gets the preset.
    for attr in set(EXPECTED_PRESET) - set(kept):
        assert attr in applied, attr
        assert getattr(args, attr) == EXPECTED_PRESET[attr][1], attr


def test_v6_boolean_no_flag_forces_off() -> None:
    # --no-grad-checkpoint / --no-torso-norm parse to False (not None) via
    # BooleanOptionalAction; False must survive the preset.
    args = _ns(v6=True, grad_checkpoint=False, torso_norm=False)
    applied, kept = train._apply_v6_preset(args)
    assert args.grad_checkpoint is False
    assert args.torso_norm is False
    assert kept == {"grad_checkpoint": False, "torso_norm": False}
    assert "grad_checkpoint" not in applied and "torso_norm" not in applied


def test_no_v6_fills_legacy_defaults() -> None:
    args = _ns(v6=False)
    applied, kept = train._apply_v6_preset(args)
    assert applied == {} and kept == {}
    for attr, (legacy_default, _) in EXPECTED_PRESET.items():
        assert getattr(args, attr) == legacy_default, attr


def test_no_v6_explicit_values_kept() -> None:
    args = _ns(v6=False, value_bins=51, torso_norm=True, q_aux_coef=0.25)
    train._apply_v6_preset(args)
    assert args.value_bins == 51
    assert args.torso_norm is True
    assert args.q_aux_coef == 0.25
    # Non-overridden flags still resolve to legacy defaults.
    assert args.sizing_head == "anchor"
    assert args.advantage_estimator == "gae"
