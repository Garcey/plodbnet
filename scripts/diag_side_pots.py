"""Side-pot handling diagnostic.

Constructs four hand-crafted payout scenarios with explicit commits and
known hole cards / boards, calls the engine's pure
``double_board_payout`` via the new ``_engine.compute_double_board_payout``
binding, and prints:

  (a) the constructed pot layers (size, contributors, eligibility),
  (b) the per-layer per-board winner determination,
  (c) the final per-seat payout from the engine,
  (d) the per-seat net delta vs commit,
  (e) zero-sum confirmation.

If any scenario's deltas don't match expectation, the script prints
PASS / FAIL per scenario and exits non-zero on any failure.

Run after a fresh ``maturin develop --release`` so the Rust binding is
picked up. Does not touch training state.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from plo5bp import _engine

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def card(s: str) -> int:
    """Parse a 2-char card string ('Ah', 'Tc', ...) to a 0..51 index."""
    if len(s) != 2:
        raise ValueError(f"bad card {s!r}")
    r = RANK_CHARS.index(s[0].upper())
    suit = SUIT_CHARS.index(s[1].lower())
    return r * 4 + suit


def card_str(idx: int) -> str:
    return f"{RANK_CHARS[idx // 4]}{SUIT_CHARS[idx % 4]}"


def hand(s: str) -> list[int]:
    parts = s.split()
    if len(parts) != 5:
        raise ValueError(f"need 5 cards, got {len(parts)}: {s!r}")
    return [card(p) for p in parts]


def board(s: str) -> list[int]:
    parts = s.split()
    if len(parts) != 5:
        raise ValueError(f"need 5 cards, got {len(parts)}: {s!r}")
    return [card(p) for p in parts]


def fmt_cards(indices: list[int]) -> str:
    return " ".join(card_str(i) for i in indices)


@dataclass
class Scenario:
    name: str
    description: str
    commits: list[int]
    folded: list[bool]
    holes: list[list[int]]
    board_a: list[int]
    board_b: list[int]
    button: int
    expected_deltas: list[int]


def assert_no_collisions(scn: Scenario) -> None:
    """Sanity check: every card index appears at most once across all
    hole cards and both boards. Raises AssertionError on conflict."""
    seen: dict[int, str] = {}
    for s, h in enumerate(scn.holes):
        for c in h:
            if c in seen:
                raise AssertionError(
                    f"Scenario {scn.name!r}: card {card_str(c)} in seat {s} "
                    f"already used by {seen[c]}"
                )
            seen[c] = f"seat {s}"
    for c in scn.board_a:
        if c in seen:
            raise AssertionError(
                f"Scenario {scn.name!r}: board A {card_str(c)} already in "
                f"{seen[c]}"
            )
        seen[c] = "board A"
    for c in scn.board_b:
        if c in seen:
            raise AssertionError(
                f"Scenario {scn.name!r}: board B {card_str(c)} already in "
                f"{seen[c]}"
            )
        seen[c] = "board B"


def build_layers(commits: list[int], folded: list[bool]) -> list[dict]:
    """Mirror the layering algorithm in double_board.rs for display."""
    n = len(commits)
    levels = sorted(set(c for c in commits if c > 0))
    layers = []
    prev = 0
    for level in levels:
        contributors = sum(1 for c in commits if c >= level)
        chips = (level - prev) * contributors
        eligible = [i for i in range(n) if commits[i] >= level and not folded[i]]
        layers.append({
            "prev_level": prev,
            "level": level,
            "contributors": contributors,
            "chips": chips,
            "eligible": eligible,
        })
        prev = level
    return layers


def per_layer_per_board_payout(
    holes: list[list[int]],
    folded: list[bool],
    board_a: list[int],
    board_b: list[int],
    button: int,
    layer: dict,
    n: int,
) -> tuple[list[int], list[int]]:
    """Isolate one layer's per-board distribution by calling the engine
    with synthetic commits that produce only this layer.

    For board-A isolation, pass (board_a, board_a). The engine will split
    each layer half-and-half between the two board arguments; passing the
    same board twice means the *full* layer goes to board A's winner(s).
    Same logic for board B."""
    per_seat = layer["level"] - layer["prev_level"]
    synthetic = [
        per_seat if i in layer["eligible"] else 0
        for i in range(n)
    ]
    pa = list(_engine.compute_double_board_payout(
        sum(holes, []), folded, synthetic,
        board_a, board_a, button,
    ))
    pb = list(_engine.compute_double_board_payout(
        sum(holes, []), folded, synthetic,
        board_b, board_b, button,
    ))
    return pa, pb


def winners_from_payout(payout: list[int]) -> list[int]:
    return [i for i, v in enumerate(payout) if v > 0]


def run_scenario(scn: Scenario) -> bool:
    print("=" * 78)
    print(f"SCENARIO: {scn.name}")
    print(f"  {scn.description}")
    print()

    assert_no_collisions(scn)

    n = len(scn.commits)
    print(f"  Seats: {n}, button: {scn.button}")
    for i in range(n):
        status = "folded" if scn.folded[i] else "alive "
        print(
            f"    seat {i}: {status}  commit=${scn.commits[i]:>5}  "
            f"hole={fmt_cards(scn.holes[i])}"
        )
    print(f"  Board A: {fmt_cards(scn.board_a)}")
    print(f"  Board B: {fmt_cards(scn.board_b)}")
    print()

    # (a) Pot layers
    layers = build_layers(scn.commits, scn.folded)
    print("(a) Pot layers (built from sorted unique commits):")
    for k, ly in enumerate(layers):
        elig_str = ", ".join(f"seat{i}" for i in ly["eligible"])
        print(
            f"    layer {k+1}: level=${ly['level']:>5}  "
            f"size=${ly['chips']:>5} = "
            f"({ly['level']}-{ly['prev_level']})*{ly['contributors']}  "
            f"eligible=[{elig_str}]"
        )
    total_layer_chips = sum(ly["chips"] for ly in layers)
    print(f"    TOTAL POT: ${total_layer_chips} "
          f"(== sum(commits) = ${sum(scn.commits)})")
    assert total_layer_chips == sum(scn.commits), "layer chips don't sum to commits"
    print()

    # (b) Per-layer per-board winners
    print("(b) Per-layer per-board winners (isolated via synthetic commits):")
    for k, ly in enumerate(layers):
        pa, pb = per_layer_per_board_payout(
            scn.holes, scn.folded, scn.board_a, scn.board_b,
            scn.button, ly, n,
        )
        layer_chips = ly["chips"]
        half_a = layer_chips // 2
        half_b = layer_chips - half_a
        wa = winners_from_payout(pa)
        wb = winners_from_payout(pb)
        wa_str = ", ".join(f"seat{i}" for i in wa) or "(none)"
        wb_str = ", ".join(f"seat{i}" for i in wb) or "(none)"
        print(
            f"    layer {k+1} (${layer_chips}): "
            f"board A half ${half_a} -> [{wa_str}], "
            f"board B half ${half_b} -> [{wb_str}]"
        )
    print()

    # (c) Final per-seat payout via engine
    payout = list(_engine.compute_double_board_payout(
        sum(scn.holes, []), scn.folded, scn.commits,
        scn.board_a, scn.board_b, scn.button,
    ))
    print("(c) Final per-seat payouts (engine call):")
    for i in range(n):
        print(f"    seat {i}: won ${payout[i]}")
    print(f"    sum(payouts) = ${sum(payout)}  (== sum(commits) = ${sum(scn.commits)})")
    print()

    # (d) Net deltas
    deltas = [payout[i] - scn.commits[i] for i in range(n)]
    print("(d) Per-seat net delta (payout - commit):")
    all_match = True
    for i in range(n):
        sign = "+" if deltas[i] >= 0 else ""
        exp_sign = "+" if scn.expected_deltas[i] >= 0 else ""
        ok = deltas[i] == scn.expected_deltas[i]
        all_match &= ok
        match = "OK" if ok else "MISMATCH"
        print(
            f"    seat {i}: {sign}${deltas[i]:<5}  "
            f"(expected {exp_sign}${scn.expected_deltas[i]})  [{match}]"
        )
    print()

    # (e) Zero-sum
    delta_sum = sum(deltas)
    zero_sum_ok = delta_sum == 0
    print(f"(e) Zero-sum check: sum(deltas) = ${delta_sum}  "
          f"[{'OK' if zero_sum_ok else 'FAIL'}]")
    print()

    ok = all_match and zero_sum_ok
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    print()
    return ok


# ---------------------------------------------------------------------------
# Constants for shared boards / hands.
#
# BOARD_A and BOARD_B are both QQ-paired, J/T/9 unconnected, with no flush
# potential alone. Hands holding AAxx make AA-QQ-J on each board; KKxx
# make KK-QQ-J. The two boards use different suits/cards to avoid any
# card-collision with the test hands.
# ---------------------------------------------------------------------------
# Both boards: QQ-paired with low rags spread far apart so no junk hand
# can make a straight by connecting through the board. Disjoint from each
# other and from the test hands.
BOARD_A_AA_WINS = "Qc Qd 7h 4h 2s"
BOARD_B_AA_WINS = "Qh Qs 8d 5c 3d"

HAND_AAAA_2c = "Ac Ad Ah As 2c"
HAND_KKKK_2h = "Kc Kd Kh Ks 2h"
HAND_JUNK_1 = "3c 4d 5h 7s 9c"        # no Q/J/T/A/K, no all-suited bunch
HAND_JUNK_2 = "2d 6h 8s Tc Jd"        # disjoint from JUNK_1 and AA/KK hands


SCENARIOS = [
    # ----------------------------------------------------------------
    # Scenario 1: 3-player, commits [400, 400, 100], short stack scoops
    # main pot; deeper seats split the side pot per their relative
    # strength on each board.
    # ----------------------------------------------------------------
    Scenario(
        name="1: short stack scoops main pot, deeper seat scoops side pot",
        description=(
            "Commits [400, 400, 100]. Seat 2 holds AAAA -> AA-QQ-J on both "
            "boards (best). Seat 0 holds KKKK -> KK-QQ-J (second). Seat 1 "
            "junk. Main pot $300 -> seat 2; side pot $600 -> seat 0 (seat 1 "
            "loses to seat 0 on both boards)."
        ),
        commits=[400, 400, 100],
        folded=[False, False, False],
        holes=[
            hand(HAND_KKKK_2h),
            hand(HAND_JUNK_1),
            hand(HAND_AAAA_2c),
        ],
        board_a=board(BOARD_A_AA_WINS),
        board_b=board(BOARD_B_AA_WINS),
        button=0,
        expected_deltas=[200, -400, 200],
    ),

    # ----------------------------------------------------------------
    # Scenario 2: short stack wins nothing; seat 0 wins board A both
    # pots, seat 1 wins board B both pots.
    # ----------------------------------------------------------------
    # Seat 0 holds Ac Kc + non-flushy filler -> A-K-high club flush on
    # the 3-club board A. Seat 1 holds Ad Kd + filler -> nut diamond
    # flush on board B. Seat 2 has at best pair of 8s on either board
    # (loses to both flushes).
    Scenario(
        name="2: 0 wins board A, 1 wins board B (short stack loses)",
        description=(
            "Commits [400, 400, 100]. Seat 0 makes nut club flush on "
            "board A; seat 1 makes nut diamond flush on board B; seat 2 "
            "never wins. Main pot $300 -> 150 (board A) to seat 0 + 150 "
            "(board B) to seat 1. Side pot $600 -> 300 each to seats 0/1."
        ),
        commits=[400, 400, 100],
        folded=[False, False, False],
        holes=[
            hand("Ac Kc 5h 7h 9s"),  # 2 clubs -> club flush on board A.
            hand("Ad Kd 5s 7s 9c"),  # 2 diamonds -> diamond flush on board B.
            hand("2c 3d 8h 3s 9h"),  # at best pair of 8s on either board.
        ],
        board_a=board("Tc 8c 6c 4s 2h"),  # 3 clubs unpaired
        board_b=board("Td 8d 6d 4h 2s"),  # 3 diamonds unpaired
        button=0,
        expected_deltas=[50, 50, -100],
    ),

    # ----------------------------------------------------------------
    # Scenario 3: short stack wins board A, seat 0 wins board B + side
    # pot board A half (since seat 2 isn't eligible for the side pot).
    # ----------------------------------------------------------------
    Scenario(
        name="3: short wins board A; seat 0 wins board B + side pot board A",
        description=(
            "Commits [400, 400, 100]. Seat 2 (AA-QQ-J) wins board A "
            "outright. Seat 0 (K-high spade flush) wins board B. In the "
            "side pot, seat 2 isn't eligible, so seat 0 (KK-QQ-J) takes "
            "board A's half too."
        ),
        commits=[400, 400, 100],
        folded=[False, False, False],
        holes=[
            # Seat 0: KK + As Ks for nut spade flush on board B. On board A
            # (Qc Qd 7h 4h 2s), best is KK-QQ-7.
            hand("Ks Kh As 4d 9d"),
            # Seat 1: junk; QQ-44-T on board A, 88-66-J on board B.
            hand("3c 4c 6h 8h Tc"),
            # Seat 2: AA + offsuit junk. AA-QQ-7 on board A; AA pair on
            # board B (loses to seat 0's flush).
            hand("Ac Ad 7c 5d 9c"),
        ],
        board_a=board(BOARD_A_AA_WINS),
        # Board B: 3 spades (Js 8s 5s) + 6d 3h — no overlap with hands.
        board_b=board("Js 8s 5s 6d 3h"),
        button=0,
        expected_deltas=[350, -400, 50],
    ),

    # ----------------------------------------------------------------
    # Scenario 4: 4 seats, commits [400, 400, 200, 100] -> three layers.
    # Seat 0 has the best hand on both boards and scoops every layer.
    # ----------------------------------------------------------------
    # Layer 1 = $400 (4x$100), eligible {0,1,2,3}, -> seat 0.
    # Layer 2 = $300 (3x$100), eligible {0,1,2},   -> seat 0.
    # Layer 3 = $400 (2x$200), eligible {0,1},     -> seat 0.
    # Total $1100 -> seat 0.
    Scenario(
        name="4: 4-player scoop across three side-pot layers",
        description=(
            "Commits [400, 400, 200, 100]. Three layers (400/300/400). "
            "Seat 0 (AA-QQ-J) is best on both boards -> scoops every "
            "eligible layer."
        ),
        commits=[400, 400, 200, 100],
        folded=[False, False, False, False],
        holes=[
            hand(HAND_AAAA_2c),
            hand(HAND_KKKK_2h),
            hand(HAND_JUNK_1),
            hand(HAND_JUNK_2),
        ],
        board_a=board(BOARD_A_AA_WINS),
        board_b=board(BOARD_B_AA_WINS),
        button=0,
        expected_deltas=[700, -400, -200, -100],
    ),
]


def main() -> int:
    print()
    print("=" * 78)
    print("Side-pot diagnostic — engine call:")
    print("  _engine.compute_double_board_payout(holes, folded, commits, A, B, btn)")
    print("=" * 78)
    print()

    failed = []
    for scn in SCENARIOS:
        try:
            ok = run_scenario(scn)
        except AssertionError as e:
            print(f"  SETUP ERROR: {e}")
            print(f"RESULT: FAIL ({scn.name})")
            print()
            ok = False
        if not ok:
            failed.append(scn.name)

    print("=" * 78)
    if not failed:
        print("ALL SCENARIOS PASSED. Side-pot handling is verified.")
        return 0
    print(f"FAILURES ({len(failed)}/{len(SCENARIOS)}):")
    for name in failed:
        print(f"  - {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
