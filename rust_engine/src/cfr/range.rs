//! HU range over 1326 hole combos (C(52,2)).

use crate::cards::Card;
use crate::hand_eval::evaluate_nlh;

use super::CfrError;

/// Number of unordered 2-card combos from a 52-card deck.
pub const NUM_COMBOS: usize = 1326; // C(52,2)

/// Map combo id 0..1325 → (c0, c1) with c0 < c1.
#[inline]
pub fn combo_cards(id: usize) -> (u8, u8) {
    debug_assert!(id < NUM_COMBOS);
    // Inverse of triangular: find c1 such that C(c1,2) <= id < C(c1+1,2)
    // c1 from 1..51, base = c1*(c1-1)/2
    let mut c1 = 1u8;
    while {
        let next = (c1 as usize + 1) * (c1 as usize) / 2;
        id >= next && c1 < 51
    } {
        c1 += 1;
    }
    let base = (c1 as usize) * (c1 as usize - 1) / 2;
    let c0 = (id - base) as u8;
    (c0, c1)
}

/// `combo id → (c0, c1)` lookup table (built once). `combo_cards` walks the
/// triangular numbers on every call, which is too slow for per-combo loops.
pub fn combo_table() -> &'static [(u8, u8)] {
    static TABLE: std::sync::OnceLock<Vec<(u8, u8)>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| (0..NUM_COMBOS).map(combo_cards).collect())
}

/// Map (c0, c1) with c0 != c1 → combo id.
#[inline]
pub fn cards_to_combo(a: u8, b: u8) -> usize {
    let (c0, c1) = if a < b { (a, b) } else { (b, a) };
    let c1u = c1 as usize;
    c1u * (c1u - 1) / 2 + c0 as usize
}

/// Reach weights over combos (non-negative). Not necessarily normalized.
#[derive(Debug, Clone)]
pub struct Range {
    pub weights: Vec<f64>,
}

impl Range {
    pub fn uniform_unblocked(board: &[u8]) -> Self {
        let mut blocked = [false; 52];
        for &c in board {
            if (c as usize) < 52 {
                blocked[c as usize] = true;
            }
        }
        let mut weights = vec![0.0; NUM_COMBOS];
        for id in 0..NUM_COMBOS {
            let (c0, c1) = combo_cards(id);
            if !blocked[c0 as usize] && !blocked[c1 as usize] {
                weights[id] = 1.0;
            }
        }
        Self { weights }
    }

    pub fn total(&self) -> f64 {
        self.weights.iter().sum()
    }

    pub fn normalize(&mut self) {
        let t = self.total();
        if t > 0.0 {
            for w in &mut self.weights {
                *w /= t;
            }
        }
    }

    /// Zero combos that share a card with `hand` or board.
    pub fn block_cards(&mut self, cards: &[u8]) {
        let mut blocked = [false; 52];
        for &c in cards {
            if (c as usize) < 52 {
                blocked[c as usize] = true;
            }
        }
        for id in 0..NUM_COMBOS {
            let (c0, c1) = combo_cards(id);
            if blocked[c0 as usize] || blocked[c1 as usize] {
                self.weights[id] = 0.0;
            }
        }
    }

    /// Bayes update: multiply by strategy probability for action (per combo).
    pub fn multiply_action_probs(&mut self, probs: &[f64]) {
        assert_eq!(probs.len(), NUM_COMBOS);
        for id in 0..NUM_COMBOS {
            self.weights[id] *= probs[id];
        }
    }

    /// True when `spec` means "no range given" (uniform over unblocked combos).
    pub fn spec_is_uniform(spec: &str) -> bool {
        let s = spec.trim();
        s.is_empty()
            || s.eq_ignore_ascii_case("random")
            || s.eq_ignore_ascii_case("any")
            || s == "100%"
            || s == "*"
    }

    /// Number of combos with positive weight.
    pub fn live_combos(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }

    /// Parse a range string into combo weights.
    ///
    /// Tokens are separated by commas, semicolons or any whitespace/newlines;
    /// each token is `item` or `item:weight` (`weight` finite, >= 0):
    ///
    /// - empty / `random` / `any` / `100%` / `*` → uniform over unblocked combos
    /// - combo id `0..1325`: `#44`, or any all-digit token that is not exactly
    ///   two digits (`7`, `104`, zero-padded `0044`). A TWO-digit token is a
    ///   hand (`44` = pocket fours, `98` = 98s+98o) unless the whole spec is the
    ///   numeric machine format (see the comment in the body)
    /// - explicit combo `AhKh` (rank+suit twice; suits `c d h s`)
    /// - class `AA`, `AKs`, `AKo`, `AK` (= suited + offsuit), case-insensitive
    /// - `QQ+` (pairs up), `ATs+` / `A9o+` / `KT+` (kicker up to one below the
    ///   high card; hence `76s+` ≡ `76s`)
    /// - `KK-TT` (pair run), `A5s-A2s` (same high card, kicker run),
    ///   `T9s-65s` (constant-gap ladder)
    ///
    /// A combo named by several tokens takes the LAST token's weight — weights
    /// never accumulate, so `"AA,AA,AA,KK"` ≡ `"AA,KK"`. Combos blocked by
    /// `board` are dropped.
    ///
    /// (review 2026-09-20 D12) Any token that does not parse, and a range
    /// with no live combo left, is an ERROR. The old parser skipped what it
    /// did not understand (`AA:0.5`, `AhKh`, `KK-TT`, `QQ+`, whitespace lists)
    /// and, when nothing was left, silently solved the UNIFORM range while the
    /// report said `ranges=parsed`.
    pub fn parse(spec: &str, board: &[u8]) -> Result<Self, CfrError> {
        if Self::spec_is_uniform(spec) {
            return Ok(Self::uniform_unblocked(board));
        }
        let mut blocked = [false; 52];
        for &c in board {
            if (c as usize) < 52 {
                blocked[c as usize] = true;
            }
        }
        let bad = |token: &str, why: &str| {
            CfrError::InvalidRoot(format!("range: cannot parse token {token:?} ({why})"))
        };
        // Pass 1: split `item[:weight]`.
        let mut items: Vec<(&str, &str, f64, bool)> = Vec::new();
        for token in spec.split(|c: char| c == ',' || c == ';' || c.is_whitespace()) {
            let token = token.trim();
            if token.is_empty() {
                continue;
            }
            let (item, weight, has_weight) = match token.split_once(':') {
                Some((item, w_s)) => {
                    let w: f64 = w_s
                        .trim()
                        .parse()
                        .map_err(|_| bad(token, "weight is not a number"))?;
                    if !w.is_finite() || w < 0.0 {
                        return Err(bad(token, "weight must be finite and >= 0"));
                    }
                    (item.trim(), w, true)
                }
                None => (token, 1.0, false),
            };
            items.push((token, item, weight, has_weight));
        }
        // TWO-digit tokens are ambiguous: `44` is pocket fours AND combo id 44,
        // `98` is the class 98 AND combo id 98. Every other all-digit token
        // (`9`, `104`, the desktop app's zero-padded `0044`) and `#N` can only
        // be a combo id. A two-digit token is read as an id only in the
        // pre-existing machine format: the whole spec is `id:weight` / `#id`
        // tokens and at least one of them is unambiguous
        // (`2:1,9:1,27:1,44:1`). Anywhere else it is a hand label.
        let digits = |s: &str| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit());
        let sure_id = |s: &str| s.starts_with('#') || (digits(s) && s.len() != 2);
        let numeric_mode = items
            .iter()
            .all(|&(_, it, _, has_w)| it.starts_with('#') || (digits(it) && has_w))
            && items.iter().any(|&(_, it, _, _)| sure_id(it));
        // None = never mentioned; Some(w) = last weight assigned.
        let mut assigned: Vec<Option<f64>> = vec![None; NUM_COMBOS];
        let n_tokens = items.len();
        for &(token, item, weight, _) in &items {
            let combos = expand_range_item(item, numeric_mode).map_err(|why| bad(token, &why))?;
            for id in combos {
                assigned[id] = Some(weight);
            }
        }
        let mut weights = vec![0.0; NUM_COMBOS];
        for (id, w) in assigned.iter().enumerate() {
            if let Some(w) = *w {
                let (c0, c1) = combo_cards(id);
                if !blocked[c0 as usize] && !blocked[c1 as usize] {
                    weights[id] = w;
                }
            }
        }
        let range = Self { weights };
        if range.live_combos() == 0 {
            return Err(CfrError::InvalidRoot(format!(
                "range: no live combo left after board blockers / zero weights \
                 ({n_tokens} token(s) parsed); pass an empty string for a uniform range"
            )));
        }
        Ok(range)
    }
}

fn rank_of(c: char) -> Option<u8> {
    match c.to_ascii_uppercase() {
        '2' => Some(0),
        '3' => Some(1),
        '4' => Some(2),
        '5' => Some(3),
        '6' => Some(4),
        '7' => Some(5),
        '8' => Some(6),
        '9' => Some(7),
        'T' => Some(8),
        'J' => Some(9),
        'Q' => Some(10),
        'K' => Some(11),
        'A' => Some(12),
        _ => None,
    }
}

fn suit_of(c: char) -> Option<u8> {
    match c.to_ascii_lowercase() {
        'c' => Some(0),
        'd' => Some(1),
        'h' => Some(2),
        's' => Some(3),
        _ => None,
    }
}

/// Suitedness filter of a class label.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Suitedness {
    Suited,
    Offsuit,
    Both,
}

/// A parsed class label: `hi >= lo`; pairs have `hi == lo` (suitedness Both).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct ClassLabel {
    hi: u8,
    lo: u8,
    suit: Suitedness,
}

/// `AA` / `AKs` / `AKo` / `AK` → label; `None` when `s` is not a class label.
fn parse_class_label(s: &str) -> Option<ClassLabel> {
    let ch: Vec<char> = s.chars().collect();
    if ch.len() < 2 || ch.len() > 3 {
        return None;
    }
    let (r0, r1) = (rank_of(ch[0])?, rank_of(ch[1])?);
    let suit = match ch.get(2).map(|c| c.to_ascii_lowercase()) {
        None => Suitedness::Both,
        Some('s') => Suitedness::Suited,
        Some('o') => Suitedness::Offsuit,
        Some(_) => return None,
    };
    if r0 == r1 && suit != Suitedness::Both {
        return None; // "AAs" is not a hand
    }
    Some(ClassLabel {
        hi: r0.max(r1),
        lo: r0.min(r1),
        suit,
    })
}

fn class_combos(label: ClassLabel, out: &mut Vec<usize>) {
    for s0 in 0..4u8 {
        for s1 in 0..4u8 {
            let (c0, c1) = (label.hi * 4 + s0, label.lo * 4 + s1);
            if c0 == c1 {
                continue;
            }
            if label.hi == label.lo && s0 >= s1 {
                continue; // each pair combo once
            }
            let keep = match label.suit {
                Suitedness::Both => true,
                Suitedness::Suited => s0 == s1,
                Suitedness::Offsuit => s0 != s1,
            };
            if keep {
                out.push(cards_to_combo(c0, c1));
            }
        }
    }
}

/// Expand one range item (no weight suffix) to combo ids, or say why not.
/// `numeric_mode`: bare digits are combo ids (see `Range::parse`).
fn expand_range_item(item: &str, numeric_mode: bool) -> Result<Vec<usize>, String> {
    if item.is_empty() {
        return Err("empty item".into());
    }
    let mut out = Vec::new();
    // Numeric combo id: `#44` and any all-digit token that is not two digits
    // long are always ids; a two-digit token only in numeric mode.
    let all_digits = item.chars().all(|c| c.is_ascii_digit());
    let id_text = match item.strip_prefix('#') {
        Some(rest) => Some(rest),
        None if all_digits && (numeric_mode || item.len() != 2) => Some(item),
        None => None,
    };
    if let Some(txt) = id_text {
        if txt.is_empty() || !txt.chars().all(|c| c.is_ascii_digit()) {
            return Err("expected a combo id after '#'".into());
        }
        let id: usize = txt.parse().map_err(|_| "combo id out of range".to_string())?;
        if id >= NUM_COMBOS {
            return Err(format!("combo id must be 0..{}", NUM_COMBOS - 1));
        }
        return Ok(vec![id]);
    }
    if all_digits && parse_class_label(item).is_none() {
        return Err("two-digit token is not a hand (ranks 2-9 only); write a combo id as #N".into());
    }
    let ch: Vec<char> = item.chars().collect();
    // Explicit combo: rank suit rank suit.
    if ch.len() == 4 {
        if let (Some(r0), Some(s0), Some(r1), Some(s1)) =
            (rank_of(ch[0]), suit_of(ch[1]), rank_of(ch[2]), suit_of(ch[3]))
        {
            let (c0, c1) = (r0 * 4 + s0, r1 * 4 + s1);
            if c0 == c1 {
                return Err("both cards are the same".into());
            }
            return Ok(vec![cards_to_combo(c0, c1)]);
        }
    }
    // Run: A-B
    if let Some((a, b)) = item.split_once('-') {
        let (la, lb) = match (parse_class_label(a.trim()), parse_class_label(b.trim())) {
            (Some(la), Some(lb)) => (la, lb),
            _ => return Err("expected CLASS-CLASS, e.g. KK-TT or A5s-A2s".into()),
        };
        if la.suit != lb.suit {
            return Err("both ends of a run need the same s/o suffix".into());
        }
        let (pair_a, pair_b) = (la.hi == la.lo, lb.hi == lb.lo);
        if pair_a != pair_b {
            return Err("cannot mix a pair with a non-pair in a run".into());
        }
        if pair_a {
            for r in la.hi.min(lb.hi)..=la.hi.max(lb.hi) {
                class_combos(ClassLabel { hi: r, lo: r, suit: Suitedness::Both }, &mut out);
            }
        } else if la.hi == lb.hi {
            for lo in la.lo.min(lb.lo)..=la.lo.max(lb.lo) {
                class_combos(ClassLabel { hi: la.hi, lo, suit: la.suit }, &mut out);
            }
        } else if la.hi - la.lo == lb.hi - lb.lo {
            let gap = la.hi - la.lo;
            for hi in la.hi.min(lb.hi)..=la.hi.max(lb.hi) {
                class_combos(ClassLabel { hi, lo: hi - gap, suit: la.suit }, &mut out);
            }
        } else {
            return Err("run ends must share the high card or the gap".into());
        }
        return Ok(out);
    }
    // Plus: QQ+, ATs+, KT+
    if let Some(base) = item.strip_suffix('+') {
        let l = parse_class_label(base).ok_or_else(|| "expected CLASS+, e.g. QQ+ or ATs+".to_string())?;
        if l.hi == l.lo {
            for r in l.hi..=12 {
                class_combos(ClassLabel { hi: r, lo: r, suit: Suitedness::Both }, &mut out);
            }
        } else {
            for lo in l.lo..l.hi {
                class_combos(ClassLabel { hi: l.hi, lo, suit: l.suit }, &mut out);
            }
        }
        return Ok(out);
    }
    // Plain class.
    match parse_class_label(item) {
        Some(l) => {
            class_combos(l, &mut out);
            Ok(out)
        }
        None => Err("not a combo id, AhKh combo, class (AA/AKs/AKo/AK), CLASS+ or CLASS-CLASS".into()),
    }
}

/// Precompute showdown strength ranks for all combos on a fixed 5-card board.
pub fn combo_ranks_on_board(board: &[u8; 5]) -> Vec<u32> {
    let board_cards: [Card; 5] = [
        Card(board[0]),
        Card(board[1]),
        Card(board[2]),
        Card(board[3]),
        Card(board[4]),
    ];
    let mut blocked = [false; 52];
    for &c in board {
        blocked[c as usize] = true;
    }
    let mut ranks = vec![0u32; NUM_COMBOS];
    for id in 0..NUM_COMBOS {
        let (c0, c1) = combo_cards(id);
        if blocked[c0 as usize] || blocked[c1 as usize] {
            ranks[id] = 0;
            continue;
        }
        let hole = [Card(c0), Card(c1)];
        ranks[id] = evaluate_nlh(&hole, &board_cards);
    }
    ranks
}

/// Equity of hero combo vs villain range on fixed board (already ranked).
/// Returns P(win) + 0.5*P(tie) in [0,1], or 0 if blocked.
pub fn equity_vs_range(
    hero_id: usize,
    hero_rank: u32,
    villain: &Range,
    ranks: &[u32],
) -> f64 {
    if hero_rank == 0 {
        return 0.0;
    }
    let (h0, h1) = combo_cards(hero_id);
    let mut win = 0.0;
    let mut tie = 0.0;
    let mut total = 0.0;
    for vid in 0..NUM_COMBOS {
        let w = villain.weights[vid];
        if w <= 0.0 {
            continue;
        }
        let vr = ranks[vid];
        if vr == 0 {
            continue;
        }
        let (v0, v1) = combo_cards(vid);
        if v0 == h0 || v0 == h1 || v1 == h0 || v1 == h1 {
            continue;
        }
        total += w;
        if hero_rank > vr {
            win += w;
        } else if hero_rank == vr {
            tie += w;
        }
    }
    if total <= 0.0 {
        return 0.5; // no mass — split assumption
    }
    (win + 0.5 * tie) / total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_roundtrip() {
        for id in 0..NUM_COMBOS {
            let (c0, c1) = combo_cards(id);
            assert!(c0 < c1);
            assert_eq!(cards_to_combo(c0, c1), id);
        }
    }

    #[test]
    fn uniform_blocks_board() {
        let r = Range::uniform_unblocked(&[0, 1, 2, 3, 4]);
        let (c0, c1) = combo_cards(0); // cards 0,1 — both on board if board starts 0,1
        // combo 0 is (0,1) which is blocked
        assert_eq!(c0, 0);
        assert_eq!(c1, 1);
        assert_eq!(r.weights[0], 0.0);
        assert!(r.total() > 1000.0);
    }

    #[test]
    fn num_combos_is_1326() {
        assert_eq!(NUM_COMBOS, 52 * 51 / 2);
    }

    // ---- (review 2026-09-20 D12) parser ----

    fn live(spec: &str) -> usize {
        Range::parse(spec, &[]).unwrap_or_else(|e| panic!("{spec:?}: {e}")).live_combos()
    }

    fn card(rank: u8, suit: u8) -> u8 {
        rank * 4 + suit
    }

    #[test]
    fn parse_class_labels_and_weights() {
        assert_eq!(live("AA"), 6);
        assert_eq!(live("AKs"), 4);
        assert_eq!(live("AKo"), 12);
        assert_eq!(live("AK"), 16);
        assert_eq!(live("aks"), 4, "labels are case-insensitive");
        assert_eq!(live("22"), 6, "22 is pocket deuces, not combo id 22");
        let r = Range::parse("AA:0.5,KK", &[]).unwrap();
        let aa = cards_to_combo(card(12, 0), card(12, 1));
        let kk = cards_to_combo(card(11, 2), card(11, 3));
        assert_eq!(r.weights[aa], 0.5);
        assert_eq!(r.weights[kk], 1.0);
        assert_eq!(r.live_combos(), 12);
    }

    #[test]
    fn parse_explicit_combo() {
        let r = Range::parse("AhKh", &[]).unwrap();
        assert_eq!(r.live_combos(), 1);
        assert_eq!(r.weights[cards_to_combo(card(12, 2), card(11, 2))], 1.0);
        assert_eq!(live("AsKs,KhAh"), 2);
        assert!(Range::parse("AhAh", &[]).is_err());
    }

    #[test]
    fn parse_plus_and_runs() {
        assert_eq!(live("QQ+"), 18);
        assert_eq!(live("KK-TT"), 24);
        assert_eq!(live("TT-KK"), 24);
        assert_eq!(live("A5s-A2s"), 16);
        assert_eq!(live("ATs+"), 16); // ATs AJs AQs AKs
        assert_eq!(live("A9o+"), 60); // A9o..AKo
        assert_eq!(live("76s+"), 4, "kicker runs up to one below the high card");
        assert_eq!(live("T9s-65s"), 20); // constant-gap ladder
        assert!(Range::parse("KK-AKs", &[]).is_err());
        assert!(Range::parse("A5s-A2o", &[]).is_err());
        assert!(Range::parse("A5s-K9s", &[]).is_err());
    }

    #[test]
    fn parse_separators_and_dedup() {
        assert_eq!(live("AA KK\nQQ"), 18);
        assert_eq!(live("AA;KK ,  QQ\r\n"), 18);
        let a = Range::parse("AA,AA,AA,KK", &[]).unwrap();
        let b = Range::parse("AA,KK", &[]).unwrap();
        assert_eq!(a.weights, b.weights, "duplicates must not accumulate weight");
        // Last token wins.
        let r = Range::parse("AA:0.5,AhAd:1", &[]).unwrap();
        assert_eq!(r.weights[cards_to_combo(card(12, 2), card(12, 1))], 1.0);
        assert_eq!(r.weights[cards_to_combo(card(12, 0), card(12, 3))], 0.5);
    }

    #[test]
    fn parse_numeric_combo_ids() {
        // Pre-existing machine format keeps meaning combo ids.
        let r = Range::parse("2:1,9:1,27:1,44:1", &[]).unwrap();
        assert_eq!(r.live_combos(), 4);
        assert_eq!(r.weights[44], 1.0);
        // Unambiguous id form works next to labels.
        let r = Range::parse("#44:0.25,AA", &[]).unwrap();
        assert_eq!(r.weights[44], 0.25);
        assert_eq!(r.live_combos(), 7);
        // A human "44" is pocket fours.
        assert_eq!(live("44"), 6);
        assert_eq!(live("44:0.5,55:0.5"), 12);
        assert_eq!(live("98:0.5"), 16, "two digits next to nothing unambiguous = class 98");
        // The desktop app's canonical form: zero-padded 4-digit ids mixed with
        // class tokens. Anything but a TWO-digit token is always an id.
        let r = Range::parse("0044:0.5,1224:0.25,KK", &[]).unwrap();
        assert_eq!((r.weights[44], r.weights[1224]), (0.5, 0.25));
        assert_eq!(r.live_combos(), 2 + 6);
        assert_eq!(live("AA,7:1"), 7);
        assert!(Range::parse("AA,1326", &[]).is_err());
        assert!(Range::parse("#1326", &[]).is_err());
        assert!(Range::parse("AA,10", &[]).is_err(), "'10' is neither a hand nor a sure id");
    }

    #[test]
    fn parse_errors_instead_of_uniform_fallback() {
        for bad in ["AKx", "AA:abc", "AA:-1", "AAs", "XY", "AA,,K", "QQ++", "AA:nan"] {
            assert!(Range::parse(bad, &[]).is_err(), "{bad:?} must not parse");
        }
        // Entirely board-blocked range → error (used to become uniform).
        let board = [card(12, 0), card(12, 1), card(12, 2), card(12, 3), 0];
        assert!(Range::parse("AA", &board).is_err());
        // Zero-weight-only range → error.
        assert!(Range::parse("AA:0", &[]).is_err());
        // No range given → uniform.
        for s in ["", "  ", "random", "100%", "ANY"] {
            assert_eq!(Range::parse(s, &board).unwrap().live_combos(), 47 * 46 / 2);
        }
    }

    #[test]
    fn parse_drops_board_blocked_combos() {
        let board = [card(12, 0), 1, 2, 3, 5];
        let r = Range::parse("AA,KK", &board).unwrap();
        assert_eq!(r.live_combos(), 3 + 6);
    }
}
