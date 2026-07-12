# V7 observation candidates (obs v3) — reviewed draft

*Produced 2026-07-12 by the obs-v3 workflow: 6 lens-scoped proposers (each read encoding.py + MASTER.md Part 2 before proposing; 73 raw ideas) → merge/dedup (63 ideas; drop log at bottom) → 3-panel architecture review (every idea ruled agree / pick-one / better-arch, with normalization, edge-case, and 3-encoder implementability checks). The four user-seeded ideas are marked. Status: DRAFT for the keep/modify/drop go-through — nothing here is committed design until moved into V7_DESIGN.md 2.4.*

| category | ideas | dims (all) | dims (sans remove-recs) | remove-recs |
|---|---|---|---|---|
| Position & action order | 7 | 42 | 42 | 0 |
| Action history | 14 | 418 | 320 | 4 |
| Stack / pot / price geometry | 11 | 49 | 41 | 1 |
| Board texture & hand-board combinatorics | 13 | 78 | 78 | 0 |
| Double-board structure | 5 | 32 | 32 | 0 |
| Monte-Carlo / equity extensions | 13 | 63 | 49 | 3 |
| **total** | **63** | **682** | **562** | **8** |

Verdict split: 43 agree / 7 pick / 13 better-arch. Current obs = 1020 dims; shipping everything below (minus remove-recs) ≈ 1582 dims.


## Position & action order

### POS-1 · Button distance (U1) — 4 dims  ★ USER SEED

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero's structural distance from the button both ways: seats after the button clockwise (first-to-act=0) and seats before it (button=0), plus table-size-normalized fractions. Counts over STRUCTURAL seats (folds ignored).

**Why:** Positional tightness/aggression should scale with relative depth in the order, not the raw slot — the existing one-hot conflates table sizes (slot 2 = 'middle' 6-handed but 'button' 3-handed) and forces the net to join it with the seat-exists thermometer to recover depth-as-fraction; the actor rel-position one-hot (188-196) is a dead constant.

**Final architecture** (verdict: agree): Obs-v3 tail append, 4 float32: [0] seats_after_button = (hero-button-1) mod n, raw (button seat reads n-1); [1] seats_before_button = (button-hero) mod n, raw (button reads 0); [2] after_frac = seats_after/max(n-1,1) in [0,1]; [3] before_frac = 1-after_frac. n = num_seats, counts STRUCTURAL (folds/all-ins ignored, matching the 942-950 one-hot semantics). HU: button=(1,0,1,0), non-button=(0,1,0,1). Pure modular arithmetic on hero/button/num_seats, available in serial dict, batched arrays, and the Rust encoder; no engine change.

**Reviewer reasoning:** Verified: the 188-196 actor one-hot is a dead constant (encoding.py:910-911 computes (hero-hero) mod n), so nothing normalized carries order depth today. [0],[1] are linear decodes of the existing button one-hot index + seat-exists thermometer, but the /(n-1) fractions are genuinely nonlinear and table-size-invariant — the payload. Nothing saturates (counts 0-7, fractions [0,1]); HU edge cases check out exactly as stated.

**Overlap with existing dims:** Dims 942-950 (hero-to-button one-hot) carry the same info without n-normalization; seat_exists (910-918) carries n as a thermometer.  
**Cost:** Pure numpy from already-packed button/actor/num_seats in all three encoders; trivial modular arithmetic.

### POS-2 · Live-player position (U2) — 5 dims  ★ USER SEED

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero's rank in this street's ACTUAL acting order among live acting players (non-folded, non-all-in, hero included): players before hero, players after, normalized fraction, and first/last-to-act flags. Postflop order is street-invariant, so this is also hero's future-street position modulo pending folds.

**Why:** Effective position after folds drives bet/check-back, probe, and closing-option decisions: button-distance says 'CO' but with button and SB folded hero IS last to act. Existing dims force a modular filtered walk composing active mask x all-in mask x button one-hot — deep multiplicative composition across three blocks.

**Final architecture** (verdict: agree): Obs-v3 tail, 5 float32. Acting set = seats with !folded & !all_in (hero always qualifies at his own decision); L = |acting|. Static postflop order: order_idx(s) = (s-button-1) mod n; hero_rank = count of acting seats with order_idx < order_idx(hero). [0] live_before = hero_rank raw; [1] live_after = L-1-hero_rank raw; [2] after_frac = live_after/max(L-1,1) in [0,1]; [3] first_to_act = (live_before==0); [4] last_to_act = (live_after==0). L=1 -> (0,0,0,1,1). Serial/Rust: masked walk; batched: cumsum over the button-rotated acting mask. No engine change.

**Reviewer reasoning:** Effective position after folds/all-ins is a three-block multiplicative join (active 160-168 x all-in 168-176 x button one-hot 942-950) today — exactly the deep composition MLPs represent poorly, and postflop order being street-invariant makes this hero's future-street position too. Normalization bounded; L=1 and HU edges are well-defined as specified; computable from already-packed fields in all three encoders.

**Overlap with existing dims:** Exact composition of active (160-168) + all-in (168-176) + button one-hot (942-950); first/last flags are this feature's own binary extremes.  
**Cost:** Numpy mask walk; batched = gather into button-rotated order, cumsum, read hero rank; no engine change.

### POS-3 · Players behind if call vs raise (U3) — 5 dims  ★ USER SEED

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Count of opponents guaranteed still to act this street if hero calls/checks (engine pending set: alive, non-all-in, not-yet-acted OR street_commit<bet_to_call) versus if hero raises (ALL alive non-all-in opponents must respond).

**Why:** Closing-call vs reopening-raise is THE multiway action-mechanics distinction; the required 'has this seat already acted' bit is ABSENT from the obs — a live seat at street_commit 0 that already checked is indistinguishable from one yet to act except by scanning the truncatable history window.

**Final architecture** (verdict: agree): Obs-v3 tail, 5 float32. Requires exposing engine acted_this_street (state.rs:245, already maintained for round-close) via observation_dict + batched packer arrays + Rust encoder input (trivial state copy). pending_if_call(s), s != hero: !folded & !all_in & (!acted_this_street[s] | street_commit[s] < bet_to_call). [0] behind_if_call = |pending_if_call| raw; [1] behind_if_raise = |{s != hero: !folded & !all_in}| raw, defined as responders to a min-raise-or-larger hero raise (the sub-min-raise all-in reopen nuance via street_level_acted is deliberately ignored — rare and small); [2]=[0]/max(acting_opp,1); [3]=[1]/max(acting_opp,1) with acting_opp=|{s != hero: !folded & !all_in}|; [4] call_closes_action = ([0]==0). When Raise is illegal for hero, [1],[3] := [0],[2] so the raise-vs-call delta reads 0. All-acting-opponents-gone edge: (0,0,0,0,1).

**Reviewer reasoning:** I verified acted_this_street exists in engine state but is absent from observation_dict — the proposal's central claim (a checked seat at street_commit 0 is indistinguishable from a yet-to-act seat except via the truncatable history) is true, and close-vs-reopen is the core multiway mechanics distinction. The mirror of the engine pending rule is correct; I pinned behind_if_raise to the full-raise assumption to keep it objective and cheap. Plumbing is a state copy, honestly costed.

**Overlap with existing dims:** street_commit (926-934) vs bet_to_call covers the facing-a-raise half; the acted-this-street bit exists nowhere except the truncatable history (196-772).  
**Cost:** Requires exposing engine acted_this_street via observation_dict + batched arrays + Rust encoder (trivial state copy); counts then numpy mask math.

### POS-4 · Per-seat acts-after-hero mask — 8 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero-rotated 8-dim binary block: slot k=1 iff seat (hero+k) is a live acting opponent positioned AFTER hero in the static street order (button+1 clockwise) — i.e. that specific opponent has position on hero this street and every future street.

**Why:** Joins WHO with WHERE: which specific opponents act behind hero requires per-seat modular arithmetic on the button one-hot gated by two masks. Direct form changes check-to-the-covering-IP-stack vs bet-into-OOP-short-stacks decisions in multiway pots.

**Final architecture** (verdict: agree): Obs-v3 tail, 8 binary float32, hero-rotated (slot k = seat (hero+k) mod n), padded to 8. Slot k=1 iff k>0 and active[seat] & !all_in[seat] & order_idx(seat) > order_idx(hero), order_idx(s) = (s-button-1) mod n. Slot 0 (hero) always 0; folded, all-in, and padded slots 0; HU has at most one set bit. The acts-before complement is derivable (acting mask minus this block minus hero) and not stored. Pure per-seat modular compare, vectorized in batched numpy and a trivial Rust loop; no engine change.

**Reviewer reasoning:** Joins WHO with WHERE: the per-seat 'has position on me' bit is the single most decision-relevant per-seat positional signal (check-to-covering-IP-stack vs bet-into-OOP-shorties) and today requires a per-seat modular compare across three blocks. Kept alongside POS-5 per the list's own rule: this is the linear-readable threshold of the scalar field, and the project explicitly welcomes redundant formats of nonlinear derivations. All-in exclusion is right — an all-in seat is no positional threat.

**Overlap with existing dims:** Derivable from button one-hot x active x all-in via deep modular compare; U2 gives only hero's aggregate counts, not which seats.  
**Cost:** Numpy per-seat modular compare; integer-cheap, vectorized in batched numpy and Rust.

### POS-5 · Per-seat acting-order field — 8 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero-rotated 8-dim scalar field of the full acting order: slot k=(rank+1)/L for live acting seats (rank 0=first to act, L=acting count), 0.0 sentinel for folded/all-in/padded. Carries opponent-vs-opponent adjacency, not just each seat vs hero.

**Why:** Squeeze/relay geometry the binary mask cannot express: whether the aggressor acts immediately before hero or with two cold-callers sandwiched between changes raise/float ranges, and the field's argmax identifies who holds the closing option.

**Final architecture** (verdict: agree): Obs-v3 tail, 8 float32, hero-rotated, padded to 8. Acting seats (!folded & !all_in) ranked 0..L-1 by order_idx(s) = (s-button-1) mod n; slot k = (rank+1)/L in (0,1]; non-acting/padded slots 0.0. Sentinel is unambiguous: smallest live value 1/L >= 1/8 > 0. Slot 0 carries (hero_rank+1)/L (deliberate redundant format of POS-2 [2]). Recomputed each decision from current masks (order itself is street-invariant; only membership changes). Batched: cumsum over button-rotated acting mask, scatter to hero rotation; Rust straightforward. L=1: slot 0 = 1, rest 0.

**Reviewer reasoning:** Genuine superset of POS-4: opponent-vs-opponent adjacency (aggressor immediately before hero vs two cold-callers sandwiched) and the closing-option argmax are unrecoverable from the binary mask, and squeeze/relay geometry is a real PLO multiway driver. The 0.0 sentinel vs (0,1] live values is clean; normalization can't saturate. Cheap in all three encoders with no engine change.

**Overlap with existing dims:** KEPT SEPARATE from POS-4: field is a scalar full-order superset (O-vs-O adjacency + closing-option argmax) the binary mask cannot express; POS-4 = thresholded pairwise compare of this field vs hero's slot; U2 = its slot-0 aggregate.  
**Cost:** Numpy; batched = cumsum over button-rotated acting mask then scatter to hero rotation; Rust straightforward.

### POS-6 · Next-to-respond one-hot — 8 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero-rotated one-hot (8) of the first live acting opponent CLOCKWISE FROM HERO — the seat that must respond first to a hero bet/raise. Distinct from static-order features (walk continues clockwise from hero mid-street, not from the button).

**Why:** The first responder sets the immediate hurdle for a bet/bluff (pot-committed short first-to-respond vs deep flat-caller changes bluff selection and sizing); recovering it means simulating the engine actor-advance walk across three masks with wraparound. A raise reopens every seat, so this needs no acted-bit.

**Final architecture** (verdict: agree): Obs-v3 tail, 8 binary float32, hero-rotated. One-hot of the smallest k in 1..n-1 with active[(hero+k) mod n] & !all_in[(hero+k) mod n] — the seat that must respond first to a hero bet/raise (engine actor-advance walks clockwise and a full raise reopens every acting seat, so no acted-bit is needed). Slot 0 structurally impossible; all-zero when no acting opponent remains (betting dead). First-set-bit/argmax over the AND of two existing hero-rotated masks; trivial in all three encoders, no engine change.

**Reviewer reasoning:** Shallower than the rest of the family (one AND plus a prefix scan over existing masks) but the wraparound-from-hero ordering is exactly what POS-4/POS-5 cannot express linearly — their order is from the button, and re-rotating a rank field around hero's own rank is a genuinely awkward modular op. The first responder's stack/commitment (via joining with per-seat blocks) drives bluff sizing; argmin-to-one-hot is a winner-take-all op MLPs do poorly. Cheapest positional add; keep.

**Overlap with existing dims:** Contained in acting masks + order info only via the wraparound walk; POS-4/POS-5 give order-from-button, not clockwise-from-hero.  
**Cost:** Numpy first-set-bit over the hero-rotated acting mask (argmax of boolean); trivial in all three encoders; no engine change.

### POS-7 · Aggressor geometry — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero's relation to the current street's aggressor: in-position flag vs the bettor, acting-order gap between them, count of players who already called the current bet, and whether the aggressor is all-in.

**Why:** Float/call vs raise vs fold facing a c-bet hinges on being IP vs the bettor and on cold-callers in between (squeeze targets, worse pot-share multiway); raising an ALL-IN bettor buys zero fold equity from him. Existing encoding carries WHO but the relation requires three-block multiplicative joins.

**Final architecture** (verdict: agree): Obs-v3 tail, 4 float32; all-zero when last_aggressor == -1 or (defensive guard) == hero. [0] hero_ip_vs_aggressor = 1 iff order_idx(hero) > order_idx(agg), order_idx(s) = (s-button-1) mod n; [1] count of seats s not in {hero, agg} with !folded & !all_in lying strictly between agg and hero walking CLOCKWISE agg+1..hero-1 mod n (the cold-caller sandwich; 'between' pinned to the response-walk order, not static order), raw 0..6; [2] callers_of_current_bet = |{s not in {hero, agg}: !folded & street_commit[s] == bet_to_call}| raw; [3] aggressor_is_allin = all_in[agg]. Pure numpy/Rust from already-exposed last_aggressor + masks + street_commit; no engine change.

**Reviewer reasoning:** The relation to the aggressor (IP flag, sandwich count, dead-fold-equity all-in flag) is a three-block multiplicative join over the 934-942 identity one-hot today; [3] alone changes raise-for-fold-equity decisions categorically. I pinned the ambiguous 'between' to clockwise-from-aggressor (those seats already acted on the bet, i.e., the actual cold-callers) and added the agg==hero guard. [2] intentionally stays here and is dropped from HIST-11 (where it was the exact sum of that block's own flags).

**Overlap with existing dims:** Last-aggressor one-hot (934-942) gives identity only; the caller-count dim also appears in HIST-11; the IP bit is the deep three-block join.  
**Cost:** Numpy from already-exposed last_aggressor + masks + street_commit; cheap scalar math.


## Action history

### HIST-1 · Per-player histories (U4) — 80 dims  ★ USER SEED

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero-rotated per-seat block summarizing each seat's whole-hand betting record over the FULL engine history: action counts, voluntary money in, commitment fraction, last raise size, and per-street final-aggressor flags — the per-seat marginal of the action history.

**Why:** Bluff-catch vs value-raise against a seat that raised twice differs radically from one that only called, and you should not bluff a pot-committed seat — but today per-seat counts require matching seat one-hots across 32 history slots, commitment fraction exists ONLY for hero (1009), and cross-street initiative is erased (engine resets last_aggressor each street).

**Final architecture** (verdict: agree): Obs-v3 tail, 10 dims x 8 hero-rotated seats = 80 float32. ENGINE plumbing (truncation-proof; the batched window ships only the newest 32 records so encoder scans can't be parity-safe): per-seat counters maintained in apply(), with aggression defined as an action that STRICTLY INCREASED bet_to_call — immune to the encoder's all-in-call->Raise gate mislabel (_gate_from_action maps every AllIn to Raise). Per seat: [0] aggressions this hand raw, [1] calls this hand raw, [2] checks this hand raw, [3] aggressions this street raw (reset at street advance), [4] log1p(voluntary invested chips/bb) (sum of ActionRecord deltas; antes excluded — they're never recorded), [5] commitment = total_commit_s/(total_commit_s + eff_stack_s) in [0,1] using effective (dead-chip-subtracted) stacks, same formula as hero's existing dim 1009, [6] last aggression size chips/pot-before clip[0,2] (0 if none; matches the history-slot frac convention), [7..10) final-aggressor flags for flop/turn/river (engine captures street_aggressor[street] = last_aggressor at street close, immediately before the reset at engine.rs:451). Folded seats keep stats sticky; padded seats and empty history all-zero; HU uses slots 0-1. Exposed via observation_dict + new (N,8,10)-equivalent batched arrays + Rust encoder.

**Reviewer reasoning:** Highest-value history idea: per-seat aggression/commitment profiles are today only recoverable by matching seat one-hots across 32 truncatable history slots, commitment exists for hero only (dim 1009), and cross-street initiative is erased (last_aggressor reset verified at engine.rs:451). My one substantive tightening is pinning the aggression predicate to bet-level increase in the engine (gate-scan would overcount via mislabeled all-in calls). Counts are small-bounded so raw is fine; money dims use log1p per project lesson. Assembly note: [5] duplicates STK-4 and [7..10) subsumes HIST-2 — dedupe there, not here.

**Overlap with existing dims:** total/street commit /bb (918-934) carry invested implicitly (no fraction); last-aggressor one-hot within-street only; subsumes HIST-2 (per-street aggressor) and most of HIST-3/HIST-10 in a fatter format.  
**Cost:** Needs engine per-seat counters in apply() or a full-history scan at obs build, exposed via observation_dict + new batched arrays; then trivial in all encoders.

### HIST-2 · Prior-street aggressor trail — 16 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero-rotated one-hot (8) of the FLOP's final aggressor plus one-hot (8) of the TURN's, frozen when each street closes; all-zero for checked-through/unreached streets and the current street.

**Why:** Barrel/probe/delayed-cbet/check-raise-turn lines condition on who led on PRIOR streets ('flop bettor now checks' is the canonical probe trigger), but engine last_aggressor resets every street and the only trace is scanning the truncatable 32-record window.

**Final architecture** (verdict: agree): Obs-v3 tail, 16 binary float32: two 8-dim hero-rotated one-hots — flop final aggressor, turn final aggressor — frozen at each street close from an engine street_aggressor[street] capture taken where last_aggressor is currently reset (engine.rs:451); aggressor defined by bet-level increase. All-zero for checked-through/unreached streets and for the current street (the live 934-942 one-hot covers it); seats that have since folded still fire. Two ints exposed via observation_dict + batched arrays; encoders scatter to one-hots.

**Reviewer reasoning:** The premise is verified (last_aggressor genuinely resets every street, and the only trace is the truncatable window), and the engine capture is required for serial/batched parity — the arch is right as a standalone. But it is exactly contained in HIST-1's per-street final-aggressor flags, which add the river flag plus counts on the same plumbing.

**⚠ Remove recommended:** Strictly subsumed by HIST-1 (its dims [7..10) carry flop/turn/river final-aggressor flags per seat on identical engine plumbing). Adopt HIST-2 only if HIST-1 is cut for dim budget.

**Overlap with existing dims:** CONTAINED in U4's per-street final-aggressor flags (which add river); kept as a cheaper standalone alternative. History records carry it implicitly under truncation.  
**Cost:** Small engine add: capture last_aggressor into street_aggressor[street] at the reset point, expose 2 ints; required for serial/batched parity (batched ships only the truncated window).

### HIST-3 · Street-intensity counters — 5 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** How contested the street/hand are plus hero's own role: aggressive-action count this street and hand, whether hero already acted this street, whether hero was an aggressor this street (which + facing a bet = being check-raised/3-bet).

**Why:** 3-bet-pot vs single-raised vs unopened streets shift value/bluff thresholds sharply, and 'my own bet just got raised' is a different node family; today both live only in history scanning that is deep, truncation-lossy, and gate-mislabeled (all-in CALLS marked as Raise overcount aggression).

**Final architecture** (verdict: agree): Obs-v3 tail, 5 float32. ENGINE counters exposed via observation_dict + batched arrays + Rust encoder: u16 aggression counts (street + hand) where aggression = bet_to_call strictly increased (mislabel-immune), plus per-seat aggressed_this_street bools; hero_acted reads the existing acted_this_street vec (shares POS-3's exposure). [0] log1p(aggressions_this_street); [1] log1p(aggressions_this_hand); [2] hero_acted_this_street binary; [3] hero_aggressed_this_street binary; [4] facing_checkraise = [3] AND (to_call > 0). Street counters reset at advance; empty history zeros; HU well-defined.

**Reviewer reasoning:** 3-bet-pot vs single-raised vs unopened is a range-width regime variable, and 'my bet just got raised' ([4]) is a distinct node family — both live only in the truncation-lossy, gate-mislabeled history today. log1p on counts follows the project's normalization lesson. [2] is not carried by U3 (POS-3 counts opponents' pending-ness, not hero's own acted bit) and [3]/[4] are not in HIST-1 as binaries of the current street plus the faced-bet conjunction. Cheap shared plumbing with POS-3/HIST-1.

**Overlap with existing dims:** Raise counts (this street/hand) also appear raw in HIST-9; hero_acted overlaps U3's acted bit; [2]-[4] subsumed by U4 flags but not the counts; facing_checkraise ~ HIST-11 hero-was-raised.  
**Cost:** Engine counters (u16 raise counts, per-seat aggressed bools) exposed through observation_dict + batched arrays + Rust; required for parity under truncation.

### HIST-4 · Per-street betting digest — 20 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Global per-street summary for each of 4 streets: raise count, call count, check count, pot at street start (log1p /bb), and pot growth across the street (log1p pot_end_or_now/pot_start).

**Why:** River decisions need 'how did the pot get built': a 3-bet flop reaching the river means polarized ranges, a checked-through flop means capped ranges — currently a 32-slot aggregation keyed on street one-hots. Raise-count 0 prior street is the checked-through stab trigger.

**Final architecture** (verdict: agree): Obs-v3 tail, 5 dims x 4 streets = 20 float32, street-major (preflop row structurally zero in bomb pots — kept for layout regularity, same precedent as the dead preflop slot in 156-160). Per street: aggression count raw (bet-level-increase definition; all-in calls count as calls), call count raw, check count raw, log1p(pot_at_street_start/bb), pot growth = log1p(max(pot_end_or_now - pot_start, 0)/max(pot_start, 1)) — NOT log1p(ratio), so a checked-through street reads exactly 0 (the proposed log1p(pot_end/pot_start) reads 0.69 for an unchanged pot, a normalization bug). ENGINE: four u64 pot checkpoints captured at street advance + per-street action counters; current street's growth uses pot_now. Truncation-proof by construction; encoder-only aggregation rejected for the batched-window parity reason.

**Reviewer reasoning:** River range-reading genuinely needs how the pot was built (3-bet flop = polarized, checked-through = capped), currently a 32-slot aggregation keyed on street one-hots that breaks under truncation. I fixed one real normalization defect: the proposed growth form doesn't zero at no-growth. log1p(pot/bb) at 300bb pots is ~5.7, no saturation. Overlaps HIST-3's current-street aggression count in one cell — acceptable redundant format.

**Overlap with existing dims:** History records carry every action with street one-hots (deep); log1p current pot exists (1011); pot_before chain reconstructable only inside the 32-slot window.  
**Cost:** Engine stores 4 pot-at-street-start checkpoints (u64) + per-street counters at street advance; then cheap in all encoders. Encoder-only aggregation possible but truncation-limited.

### HIST-5 · Last-aggression context — 9 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Global block describing the most recent raise in the hand regardless of street: its size (pot-frac-when-made and log1p chips/bb), the street, callers-so-far, whether it was a check-raise, and how many actions ago.

**Why:** Probe/stab/delayed-cbet decisions when checked to on a new street hinge on who bet last street and HOW BIG — but last_aggressor resets every street so the one-hot is all-zero exactly when this matters, and bet-faced dims only cover a currently-faced bet.

**Final architecture** (verdict: agree): Obs-v3 tail, 9 float32, from an ENGINE-side last-aggression record rather than the proposed pure-encoder scan (the proposal itself offers this fix): on every bet-level increase, apply() stores (chips, pot_before, street, history_index, raiser_had_checked_this_street — engine tracks a per-seat checked_this_street bool reset at advance) and resets a callers-since counter; calls increment it. Dims: [0] chips/pot_before clip[0,2]; [1] log1p(chips/bb); [2..6) street one-hot (4); [6] callers-since raw; [7] check-raise flag; [8] log1p(history_len_now - history_index). All-zero when no aggression this hand. Exposed via observation_dict + batched arrays; Rust encoder copies.

**Reviewer reasoning:** The premise is verified — last_aggressor resets at street advance, so the 934-942 one-hot is all-zero exactly at the probe/delayed-cbet nodes this feature serves. The pure-encoder scan must be rejected, not merely caveated: a backward scan for Raise-gate records would return mislabeled all-in CALLS (confirmed in _gate_from_action) as 'the last aggression', and >32-action truncation silently reads as no-raise; the engine record is exact, truncation-proof, and parity-safe. Content and dims unchanged from the proposal.

**Overlap with existing dims:** Covers only the single most recent raise (HIST-8 gives last 4); last-aggressor one-hot within-street only; overlaps U4 last-raise size.  
**Cost:** Pure encoder backward scan of history (numpy trivial; batched = vectorized last-match over (N,32) arrays). >32-action hands whose last raise scrolled out read as no-raise (acceptable or fix via engine field).

### HIST-6 · Per-seat current-street stance — 32 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per seat, a 4-way one-hot of the seat's most recent action THIS street: {none-yet, checked, called/matched, raised}; all-zero for folded-out seats.

**Why:** When hero considers betting on an unraised street, every live seat has street_commit 0 and the obs cannot distinguish 'already checked' (can only check-raise, range partially capped) from 'still to act behind' (full range, positional threat) — engine acted_this_street is never exposed. Changes thin-value/protection bet and bluff-target decisions.

**Final architecture** (verdict: agree): Obs-v3 tail, 4-way one-hot x 8 hero-rotated seats = 32 float32: each seat's most recent action THIS street, {none-yet, checked, called/matched, raised}, with ENGINE-exact gates (per-seat last-gate-this-street byte maintained in apply(): check = zero-owed no-chip action, call = matched including short all-in calls, raise = bet-level increase — fixes the all-in-call mislabel). Tightened edge: none-yet fires ONLY for live acting seats (active & !all_in) with no action this street; folded seats, all-in-from-a-prior-street seats (who will never act), and padded slots read all-zero.

**Reviewer reasoning:** The blind spot is real (verified: acted_this_street is never exposed, and every live seat on an unraised street shows street_commit 0), and the explicit none-yet slot is the right design. My all-in tightening matters: the proposal would mark a prior-street all-in seat as 'none-yet', which misreads a dead seat as a pending threat. But this block is exactly the current-street column of HIST-13.

**⚠ Remove recommended:** Exact current-street projection of the stronger HIST-13 (same engine plumbing, same gate semantics). Ship only if HIST-13 is cut for size; carrying both duplicates 32 dims.

**Overlap with existing dims:** street_commit implies called/raised only when a bet is outstanding; subsumed by the current-street column of HIST-13; gives U2/U3 counts their IDENTITY companion.  
**Cost:** Cheapest via exposing acted_this_street + a per-seat last-gate-this-street byte array; encoder scan of this-street records also works.

### HIST-7 · Line-pattern flags per seat — 24 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per seat, three whole-hand line-shape booleans: check-raised (checked then raised within a street), donk-led (made a street's first raise while not the prior street's final aggressor, acting before it), and re-raised (3-bet+: raised over a raise).

**Why:** In PLO these three lines are the strongest range-strength statements on the table — check-raise and 3-bet lines are nut-dense, dual-board donk-leads announce board-specific strength. Extracting 'same seat checked then raised within one street' from the 32x18 history is the deepest positional match the encoding demands.

**Final architecture** (verdict: agree): Obs-v3 tail, 3 sticky whole-hand binary flags x 8 hero-rotated seats = 24 float32, ENGINE-maintained in apply() (3 bools/seat; encoder-side batched vectorization rightly rejected — the intra-street state machine doesn't vectorize). check_raised: seat made a bet-level-increasing action on a street where it had already checked (per-seat checked_this_street bool, reset at advance — shared with HIST-5). donk_led: seat made the FIRST aggression of street s where the prior street's captured street_aggressor (shared with HIST-1/HIST-2 plumbing) exists, != seat, and order_idx(seat) < order_idx(aggressor) in static order; structurally never fires on the flop (bomb pots have no prior-street aggressor). re_raised: seat made an aggression when >= 2 aggressions had already occurred that street (raised over a raise, 3-bet+; a plain check-raise over the opening bet does NOT fire it). Sticky until hand end, survive the seat folding; padded slots 0; HU well-defined via the order compare.

**Reviewer reasoning:** These three lines are the strongest objective range statements in PLO and are NOT subsumed by HIST-13: last-gate-per-street overwrites the check of a check-raise and carries no intra-street ordering, and donk-led needs prior-street aggressor + order info absent from the matrix. I pinned the three predicate edge cases the proposal left loose (flop has no donk; re-raise threshold >= 2 prior aggressions; aggression = bet-level increase). Engine bits are trivial and truncation-proof.

**Overlap with existing dims:** None directly — raw history only; HIST-5's single check-raise flag covers only the most recent raise.  
**Cost:** Encoder-side per-street state machine (serial numpy easy; batched vectorization awkward — engine-side sticky flags in apply() are 3 bits/seat, trivial Rust; encoders then read arrays).

### HIST-8 · Raise-size ladder — 9 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The raises-only subsequence of the hand compressed: the last 4 raises newest-first, each as (pot-frac-when-made, log1p chips/bb), plus a flag for strictly escalating pot-fractions.

**Why:** Geometric/escalating sizing (small flop, bigger turn, jam river) signals building commitment — hero should resolve fold-or-jam EARLIER (SPR planning); small-then-small distinguishes give-up from value lines. The per-raise fracs exist but interleaved with calls/checks across 32 slots.

**Final architecture** (verdict: agree): Obs-v3 tail, 9 float32: last 4 TRUE aggressions newest-first x (chips/pot_before clip[0,2], log1p(chips/bb)) = 8 dims, plus [8] escalating flag = 1 iff >= 2 aggressions present and pot-fracs strictly increase in CHRONOLOGICAL order. ENGINE-side 4-deep ring buffer of (chips, pot_before) pushed on every bet-level increase, exposed via observation_dict + batched arrays (rejects the proposed pure-encoder Raise-gate filter: all-in-call mislabels would insert non-raises into the ladder, and window truncation drops early raises). Zero-padding for < 4 raises is unambiguous — real aggressions always have chips > 0 so log1p > 0. Shares the aggression-event plumbing with HIST-5 (whose newest-slot size dims it deliberately duplicates).

**Reviewer reasoning:** Escalating-geometry detection (small flop, bigger turn, jam river) is an SPR-planning trigger that per-record fracs scattered across 32 interleaved slots cannot surface, and raise NUMBER x size pattern is nut-density information in PL. Same correction as HIST-5: the encoder-side filter is unsound given the verified gate mislabel, so the primary arch moves engine-side at trivial cost. Clip[0,2] matches the existing history-frac convention and cannot saturate under PL sizing.

**Overlap with existing dims:** HIST-5 covers only the single most recent raise; per-record fracs (history dim 17) carry the raw material scattered.  
**Cost:** Pure encoder: filter Raise-gate records, newest 4 (batched masked argsort/take). Window-truncation caveat as HIST-5.

### HIST-9 · Hand-shape counters + truncation repair — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Global scalars: total actions this hand, actions this street, raises this hand, raises this street, an explicit history-truncated flag (full length>32), and the ante pot (live seats x ante, /bb).

**Why:** Raise NUMBER is a nut-density ladder in PL (the 3rd/4th raise is the nuts); 6-handed raising wars exceed 32 actions and the net cannot know its window is incomplete; the ante pot separates forced from voluntary money mid-hand.

**Final architecture** (verdict: better): Obs-v3 tail, 4 float32 (down from 6): [0] log1p(total_actions_this_hand) from the engine's UNTRUNCATED history length — the batched packer must ship the true length as an extra int array since its window is capped at 32 (serial already sees the full list); [1] log1p(actions_this_street) from an engine per-street action counter (exact under truncation; shares HIST-4's plumbing); [2] history_truncated = (total_actions > 32) binary; [3] log1p(ante_pot/bb) with ante_pot = dealt_seats x ante frozen at hand start (config + hand-start mask, derivable in all three encoders; Rust packer adds ante + dealt count). DROPPED: the raw raises-this-hand/raises-this-street dims — exact duplicates of HIST-3 [0],[1] up to normalization, and HIST-3's engine-counter log1p versions are strictly better (mislabel-immune).

**Reviewer reasoning:** The truncation-repair pair ([0],[2]) is unique and load-bearing — the net currently cannot know its 32-slot window is incomplete, and 6-handed raising wars do exceed it. But shipping the same raise counts twice in one revision (raw here, log1p in HIST-3) is dead weight, so I cut them; the proposal itself flags the overlap. [3] near-duplicates STK-10's internal pot_at_flop — dedupe at assembly if STK-10 ships (its emitted dims are ante/bb + bloat, not the ante pot itself, so [3] stays for now).

**Overlap with existing dims:** Raise counts overlap HIST-3 (log1p there); ante pot overlaps STK-10 and the pot scalar at the first flop decision only.  
**Cost:** Trivial everywhere; batched packer must ship the UNTRUNCATED history length (extra int array) + ante pot (config-derivable).

### HIST-10 · Decayed aggression per seat — 16 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per seat, two exponentially street-decayed scalars: aggression score = sum over that seat's raises of 0.5^(streets ago) (current street weight 1), and the same-decay call score. Fixed decay constant.

**Why:** U4's flat counts lose ordering: a river raise should dominate a flop raise when hero bluff-catches at the river, but a full per-seat-per-street count table costs 4x the dims. The decayed pair is the compact recency-weighted compromise.

**Final architecture** (verdict: agree): Obs-v3 tail, 2 x 8 hero-rotated = 16 float32: decayed aggression score = sum over the seat's bet-level-increasing actions of 0.5^(current_street - action_street) (current street weight 1), and the same-decay call score; fixed decay 0.5; raw values bounded ~[0,7] by the geometric sum so no log needed. ENGINE per-seat per-street aggression/call counts rolled up at street advance (engine-side wins over the proposed encoder scan for truncation-proofness and gate exactness). Preflop contributes nothing in bomb pots; folded seats sticky; padded slots 0.

**Reviewer reasoning:** The arch is sound as written (bounded, hero-rotated, cheap), but its marginal value collapses once HIST-1 and HIST-13 both ship: HIST-1 gives per-seat counts + per-street final-aggressor flags and HIST-13 gives per-seat per-street last-gate placement, from which the torso can form its own recency weighting — the fixed 0.5 decay is a modeling choice baked into the obs rather than information. The list itself concedes it 'spans most of' U4.

**⚠ Remove recommended:** Redundant with the stronger HIST-1 + HIST-13 combination (recency structure is recoverable from their per-street placement); a fixed decay constant adds a prior, not information. Adopt only if HIST-13 is cut and only counts (HIST-1) ship.

**Overlap with existing dims:** Deliberately redundant with U4 counts + per-street aggressor flags (spans most of it); redundant-format rule invoked.  
**Cost:** Encoder scan weighting records by (current_street - record_street) — vectorizes cleanly (power-of-decay lookup); Rust trivial; or engine counters at street advance.

### HIST-11 · Current-bet response map — 9 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per seat, a live flag 'has matched the current outstanding bet this street', plus a global count of callers of the live bet, plus a hero-was-raised flag (hero has chips in this street AND faces a bet — being check-raised/3-bet right now).

**Why:** PL squeeze math: every caller of the live bet inflates hero's max raise and the field's price, and bluff-raise EV drops per caller — but 'who already matched' requires cross-dim equality matching of street_commit against bet_to_call. Hero-was-raised is THE trigger for continue-vs-fold-vs-4-bet re-evaluation.

**Final architecture** (verdict: better): Obs-v3 tail, 9 float32 (down from 10): [0..8) hero-rotated matched-current-bet flags: slot k = 1 iff bet_to_call > 0 AND !folded[(hero+k) mod n] AND street_commit[(hero+k) mod n] == bet_to_call (hero slot 0 included for rotation uniformity — it is ~always 0 at hero's own decision in bomb pots; the aggressor's slot fires by construction, which together with the 934-942 one-hot identifies the level-setter; all-in seats that matched fire; short all-in callers below the level read 0 — their partial commit is visible in 926-934); [8] hero_was_raised = (street_commit[hero] > 0) AND (to_call > 0). All-zero when no bet outstanding. Pure numpy/Rust from already-exposed fields (street_commit, bet_to_call, folded) — the cheapest history item, no engine change. DROPPED: the global caller-count dim — it is exactly the sum of the flags minus the aggressor's flag (linear within the block) and POS-7 [2] carries the count anyway.

**Reviewer reasoning:** The per-seat matched identity requires cross-dim equality matching of street_commit against bet_to_call — genuinely hard for an MLP, and the squeeze-math/bluff-raise-EV payload is real. But the proposal's own count dim is an internal linear redundancy (sum of its own flags), so I cut it and left the count to POS-7 where it belongs to the aggressor-geometry bundle. [8] complements HIST-3 [4]: together they distinguish called-then-raised from bet-then-raised.

**Overlap with existing dims:** street_commit /bb + bet_to_call carry it implicitly; caller-count also in POS-7; hero-was-raised ~ HIST-3 facing_checkraise; U3 carries left-to-act counts, not already-called identity.  
**Cost:** Pure numpy from already-exposed observation_dict fields (street_commit, bet_to_call, folded) — cheapest history entry.

### HIST-12 · Dead-money & fold-timing — 34 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per folded seat, WHICH street it folded on (4-way one-hot, all-zero while live), plus two global scalars: dead-money fraction of pot (chips from now-folded seats / pot) and log1p dead money /bb.

**Why:** Bluff frequency should rise in dead-money-rich pots (live opponents own less of what they defend); and fold TIMING (turn fold after flop call vs instant flop fold) is the only observable correlate of unseen-deck skew that the uniform-deck MC features assume away.

**Final architecture** (verdict: agree): Obs-v3 tail, 34 float32: [0..32) fold-street 4-way one-hot x 8 hero-rotated seats (which street each folded seat folded on; all-zero while live; preflop slot structurally zero in bomb pots), sourced from an ENGINE per-seat fold_street i8 (-1 = live) written in apply() — a window scan misses folds that scrolled past 32 actions, so the engine array is required for parity; [32] dead_money_frac = sum over folded seats of total_commit[s] / max(pot, 1) in [0,1] (total_commit includes antes, so folded-seat ante share is captured); [33] log1p(dead_chips/bb). Scalars are pure numpy from already-exposed folded/total_commit/pot in all three encoders.

**Reviewer reasoning:** The dead-money bluff-frequency argument and the fold-timing-as-range-signal argument are both sound, and the arch is correct with the engine fold_street array. But every component is carried elsewhere in this list: the fold-street matrix is recoverable from HIST-13 (a folded seat's fold street is the unique street whose last-gate cell is Fold), and both scalars appear in STK-8's merged arch (dead_money_frac explicitly).

**⚠ Remove recommended:** Fully covered by the combination of HIST-13 (fold-street placement via the Fold gate cells) and STK-8 (dead-money fraction dims); ship HIST-12 only if BOTH of those are cut.

**Overlap with existing dims:** Active mask + total_commit imply dead money deeply; dead-money frac also computed in STK-8; fold-street one-hots subsumed by HIST-13; MC features are uniform-deck by design.  
**Cost:** Scalars pure numpy from folded/total_commit/pot; fold-street needs a history scan for Fold records or a tiny engine per-seat fold-street array (survives truncation).

### HIST-13 · Seat-street last-gate matrix — 96 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The full 'who did what when' table: for every seat and street, a 4-way one-hot {checked, called, raised, folded} of that seat's LAST action on that street (all-zero = no action). The maximalist linear-readable line summary of every player.

**Why:** Range-reading each opponent's full line ('check-called flop, check-called turn' = capped; 'raised flop, checked turn' = give-up-or-trap) drives river polarization; every curated digest is a projection of this table, each currently a deep multi-slot seat/street match.

**Final architecture** (verdict: better): Obs-v3 tail, 96 float32 (down from 128): 8 hero-rotated seats x 3 streets (flop/turn/river — the preflop column is structurally all-zero in PLO5 bomb pots and is DROPPED, saving 32 dead dims; this encoder is PLO-only so no NLH-layout argument applies) x 4-way one-hot {checked, called, raised, folded} of the seat's LAST action on that street; all-zero cell = no action (unreached street / none yet / padded seat / never-acted). ENGINE-side per-(seat, street) last-gate u8 array updated in apply() with engine-exact gate semantics: check = zero-owed no-chip action, call = matched including short all-in calls (fixes the encoder's all-in-call->Raise mislabel), raise = bet-level increase, fold recorded on its street. Truncation-proof; shipped via observation_dict + batched (N, 8, 3) arrays; all three encoders scatter to one-hots. Street-major, seat-minor within street. Known non-coverage, by design: intra-street sequences (a check-raiser shows only 'raised' — HIST-7 carries the pattern) and the street's FINAL aggressor when two seats raised (HIST-1 [7..10) carries that).

**Reviewer reasoning:** The right maximalist anchor for the history family: every curated line digest is a projection of this table, and 'check-called flop, check-called turn = capped' is the canonical river polarization read that today needs deep multi-slot seat/street matching. The engine last-gate array is clean, truncation-proof, and fixes the gate mislabel at the source. The 32-dim always-zero preflop column fails the cost-discipline bar (nothing that large is dead in the current 1020 layout), hence the trim; the 156-160 precedent covers 4 dead dims, not 32.

**Overlap with existing dims:** Subsumes HIST-6 (current-street column), HIST-12's fold-street one-hots, and U4's per-street aggressor flags approximately; raw history carries all of it deep.  
**Cost:** Engine-side per-(seat,street) last-gate byte array in apply() is clean and truncation-proof; encoder scan also possible (batched scatter of newest record per (env,seat,street)). Largest block proposed.

### HIST-14 · History record enrichment: all-in + stack-after — 64 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Widen each of the 32 history records by 2 dims: a flag that the action left the actor all-in, and the actor's remaining stack AFTER the action (log1p /bb) — per-record stack context.

**Why:** The record gate folds jams into 'Raise': a pot bet with 100bb behind and an all-in are different range statements, and evaluating a past bet's meaning requires the actor's stack depth AT THAT TIME — reconstructing it means replaying stack arithmetic backward, which the net cannot do.

**Final architecture** (verdict: better): Obs-v3 tail, 64 float32 as a PURE TAIL APPEND rather than the proposed 18->20 in-place slot widening: two 32-dim blocks index-aligned with the existing history window — [0..32) went_all_in flag for history slot j (action left the actor all-in), [32..64) log1p(stack_after/bb) for slot j (actor's remaining stack AFTER the action). Empty slots read 0/0, unambiguous: a jam has stack_after 0 but flag 1, and empty slots also have an all-zero seat one-hot in the main block. ENGINE: ActionRecord gains stack_after: u64 + all_in: bool captured in apply() (historical stacks are otherwise unrecoverable — the proposal is right that backward stack replay is not something the net can do); batched history arrays widen by two columns; serial dict tuples gain two fields; all three encoders copy into the tail with the same newest-32 alignment as the main history block. Rationale for the layout change: widening slots in place breaks the project's append-only policy (MASTER.md: 991-era checkpoints serve via a plain tail slice; warm-starts zero-pad APPENDED first-layer columns) and would force a _V1_INDEX-style permutation projection exactly like the v1 17->18 migration; an MLP torso is permutation-blind, so tail placement carries identical information at zero migration cost.

**Reviewer reasoning:** The information claim is verified: ActionRecord ships only (seat, action, chips, street), the all-in mask (168-176) is now-only, and per-record stack context distinguishes a pot bet with 100bb behind from a jam — different range statements the encoder currently collapses. The proposal's only flaw is architectural: in-place slot widening gratuitously breaks the append-only checkpoint-compat contract this project treats as policy. The tail-aligned twin blocks preserve dims 0..1020 byte-identical while carrying the same two dims per record.

**Overlap with existing dims:** All-in mask (168-176) shows who is all-in NOW only; per-record chips/bb + current stacks cannot recover past stack context under multi-action streets.  
**Cost:** ENGINE change: ActionRecord gains stack_after (u64) + all_in (bool) at apply() (historical stack not otherwise recoverable); batched history arrays widen by 2 columns; encoders then trivial.


## Stack / pot / price geometry

### STK-1 · Money / raise-exposure still behind — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Aggregate money behind hero over the players still pending this street (the U3 if-call set): the stack mass / raise power that can still punish a bluff or pay off a value bet, as max/sum aggregates, pot-relative, with a covered flag.

**Why:** Bet/bluff sizing targets the money BEHIND, not the money already in: betting into three pending 15bb stacks vs one pending 250bb covering stack are different decisions at identical pot/hero scalars, and a call is 'closing-ish' only if the players behind CAN meaningfully raise. Joining per-seat stacks with pending-ness needs the not-yet-exposed acted bit plus a masked reduction.

**Final architecture** (verdict: better): 4 floats, global tail-append block, all-zero when the pending set is empty (hero closes / betting dead / terminal). Pending set = engine pending rule over opponents: alive, not all-in, and (not acted_this_street OR street_commit_s < bet_to_call) — requires exposing acted_this_street via observation_dict + batched arrays + Rust encoder (the same one-field plumbing POS-3/U3 needs; trivial state copy). Per pending seat s (effective, dead-chip-subtracted stacks, same recipe as _STACKS_OFF): owed_s = min(max(bet_to_call − street_commit_s, 0), eff_s); capacity_s = max(0, eff_s − owed_s). Dims: [0] log1p(max_s capacity_s / max(pot,1)); [1] log1p(Σ_s capacity_s / max(pot,1)); [2] log1p(max_s eff_s / bb); [3] pending_covers_hero = 1 iff max_s eff_s ≥ eff_hero. HU: pending set is villain-or-empty, well-defined; log1p unclipped so nothing saturates at 300bb or 6-way.

**Reviewer reasoning:** Merged rather than picked: the two listed archs share the pending set and near-identical aggregates, but each has one thing the other lacks. The stack arch's raise CAPACITY (eff − owed) is the semantically correct threat measure — a pending seat owing 80% of its stack cannot punish a bluff — so it wins for the pot-relative dims; the position arch's covered flag and /bb money companion are worth keeping (redundant money formats are house style, cf. _EFF_PRICE_OFF carrying both a ratio and log1p /bb). Checks: (a) log1p unclipped everywhere; (b) folded/all-in seats never pending, HU degenerates cleanly; (c) needs the acted_this_street exposure the cost note already declares (shared with U3 — one engine change serves both); (d) aggregates, no rotation needed.

**All proposed architectures (pre-review):**

- *position* (4 dims): 4 floats (raw effective stacks): log1p(max eff stack among pending /bb); log1p(sum pending eff /bb); log1p(max pending eff /max(pot,1)) SPR of biggest lurker; pending_covers_hero binary. Zeros when pending set empty.
- *stack_geometry* (2 dims): 2 floats (raise CAPACITY = max(0, eff_seat - owed) over seats yet to act): log1p(max_behind_capacity/max(pot,1)) and log1p(sum_behind_capacity/max(pot,1)); 0 when hero closes. Same left-to-act set as U3.

**Overlap with existing dims:** MERGE of position 'Money still to act' + stack 'behind_raise_exposure' (raw eff-stack vs stack-minus-owed — same pending set, near-identical aggregate). Per-seat stacks (176-184) and log1p SPR (1012-1020) carry per-seat values; max_bet embeds max-reachable over ALL live opponents, not the pending subset. Complements U3 counts with amounts.  
**Cost:** Numpy masked max/sum once acted_this_street is exposed (shared engine change with U3); trivial to vectorize in batched numpy and Rust.

### STK-2 · raise_ladder_envelope — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The legal raise window in the sizing head's own coordinates: where min_raise and max_raise land on the 0..1000pm anchor ladder (as fractions of base=pot+to_call), the same deltas as fractions of hero's effective stack, a stack-capped flag, and the count of distinct legal anchors after dedupe.

**Why:** Anchor selection when the ladder is compressed — facing a pot bet shallow, rungs collapse into the min atom and 'pot' means jam. Today min_bet/max_bet exist only as /bb TOTALS (186-187); recovering ladder shape requires subtracting hero street commit and replaying the dedupe rule — the policy head chooses among anchors the obs never describes.

**Final architecture** (verdict: agree): 6 floats, tail append, all-zero when Raise is illegal (predicate pinned identically in all three encoders: max_d > to_call, where min_d = min_bet_total − street_commit_hero, max_d = max_bet_total − street_commit_hero are the DELTAS recovered from the already-packed totals — no engine change even for batched, contrary to the cost note's caution). base = pot + to_call (the anchor spec's own base). Dims: [0] (min_d − to_call)/base clip[0,1]; [1] (max_d − to_call)/base clip[0,1]; [2] min_d/max(eff_hero,1) clip[0,1]; [3] max_d/max(eff_hero,1) clip[0,1]; [4] stack_capped = 1 iff max_d < to_call + base (PL cap unreached because stack ran out); [5] n_legal_anchors/11 after the exact sizing.py dedupe (reuse anchor_grid_np in both python paths; Rust reimplements the same 11-rung closed form + strictly-greater dedupe, pinned by parity test).

**Reviewer reasoning:** High-value and correctly designed: the policy head selects among 11 anchors whose collapsed/legal shape the obs never describes; min_bet/max_bet at 186-187 are /bb totals only. The clips here are by-construction bounds (PL guarantees frac ≤ 1; a raise delta never exceeds the engine stack, and eff_hero < raw stack only via dead chips the encoding is contractually invariant to), not information-deleting saturation — so the [0,4]-clip lesson doesn't bite. Checks: (a) no saturation at 300bb (all dims are [0,1] ratios of co-scaled quantities); (b) raise-illegal → zeros covers all-in/short cases, HU fine; (c) verified min_bet/max_bet totals + street_commit are in both the serial dict and packed arrays (bindings.rs lines 335/565+), and anchor_grid_np is a pure function of (min_raise, max_raise, pot, to_call); (d) global block, no rotation.

**Overlap with existing dims:** min_bet/max_bet totals /bb (186-187) carry the raw endpoints in bb only; ladder-coordinate, stack-fraction, and anchor-count views are new.  
**Cost:** Pure numpy; anchor count reuses sizing.anchor_grid_np; batched packer may need min_raise/max_raise deltas added (trivial, non-MC); Rust same closed form.

### STK-3 · pairwise_eff_spr — 8 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per-seat money hero can actually play for against THAT specific opponent: log1p(min(eff_hero, eff_seat)/pot), hero-rotated, folded/non-exist seats 0.

**Why:** Multiway sizing and implied-odds hinge on the pairwise effective stack — the raiser is short (nothing behind the call) but a deep player lurks; hero's own SPR overstates what is winnable vs the shorty and the shorty's SPR ignores the deep threat. Existing per-seat SPRs force an element-wise min across two slots.

**Final architecture** (verdict: agree): 8 floats, hero-rotated, padded to 8, tail append: slot k = log1p(min(eff_hero, eff_seat_k) / max(pot,1)), unclipped, effective (dead-chip-subtracted) stacks; slot 0 = hero (duplicates hero's own log-SPR as a consistency anchor); folded, all-in-with-0-behind, and padded slots read 0 (unambiguous: a live seat with chips always reads > 0; an all-in seat's pairwise playable money genuinely is 0).

**Reviewer reasoning:** The arch is right as proposed (log1p unclipped follows the deep-tier saturation lesson; hero-rotation and padding match house convention; computable from already-packed stacks/eff_stack_cap/pot in all three encoders). But the information content is thinner than the pitch: log1p is monotone, so log1p(min(eff_h,eff_s)/pot) = min(SPR_LOG[0], SPR_LOG[k]) elementwise over the existing unclipped tail block at 1012-1020, and min(a,b) = a − relu(a−b) is a single hidden unit per seat — the shallowest composition on the whole candidate list. Checks (a)-(d) all pass; the question is purely whether 8 dims buy anything a handful of first-layer units can't.

**⚠ Remove recommended:** Exactly an elementwise min of two existing dims (_SPR_LOG_OFF slot 0 vs slot k) — one ReLU unit per seat recovers it. Weakest marginal value in the stack family; STK-7 (aggregate ceiling) covers the implied-odds use with 2 dims.

**Overlap with existing dims:** _SPR_LOG_OFF (1012-1019) carries both operands; the pairwise min is new; kin to STK-7 (per-seat view vs aggregate ceiling).  
**Cost:** Per-seat (x6) single vectorized min over already-packed arrays; trivial everywhere.

### STK-4 · seat_commitment_ratio — 8 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per-seat commitment fraction commit_s/(commit_s+eff_stack_s) in [0,1] — how pot-stuck each player is (1.0=all-in), hero-rotated.

**Why:** Bluff-target selection and shove/call reads: betting to fold out a 70%-committed opponent is torching money, and a committed player's continuing range is inelastic. Only HERO's commitment ratio exists (1009); for opponents the net must divide across two per-seat blocks.

**Final architecture** (verdict: agree): 8 floats, hero-rotated, padded to 8, tail append: slot k = total_commit_k / (total_commit_k + eff_stack_k), matching dim 1009's exact recipe (RAW total_commit chips, EFFECTIVE dead-chip-subtracted stack — the existing hero dim does not effective-ize the commit, so neither does this block). Slot 0 = hero (duplicate of 1009, consistency anchor); folded seats and padded slots read 0 (unambiguous — every dealt-in live seat has ante committed, so live ratios are strictly > 0); all-in live seats read 1.0.

**Reviewer reasoning:** Agree with two tightenings: pin the recipe to dim 1009's raw-commit/eff-stack formula (the proposal's 'both effective' would silently diverge from the hero dim and break the cheapest consistency check), and zero folded seats so the block reads 'live players' pot-stuckness' (dead money belongs to HIST-12/STK-8). The composition the net currently faces — a divide across the 918-926 commit block and 176-184 stack block — is genuinely nonlinear, unlike STK-3's min. Bounded [0,1], no saturation at any stack depth. Checks: (a) fraction, safe; (b) all-in→1.0, folded→0, HU fine; (c) pure elementwise math on packed arrays in all three encoders; (d) hero-rotated per spec. Kept despite the U4 commitment column per the list's own not-a-strict-dup ruling — U4 needs engine counters and 80 dims; this is free.

**Overlap with existing dims:** Same quantity as U4's per-seat commitment-fraction column, but U4 is a mega-block and this is focused — NOT a strict dup, kept. Hero-only version at eff-price+2 (1009); ingredients in total_commit + stacks per seat.  
**Cost:** Per-seat (x6) elementwise divide on packed arrays; trivial.

### STK-5 · spr_after_action — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hypothetical next-node geometry: hero's effective SPR and pot after (a) hero calls (pot+to_call, stack-to_call) and (b) hero max-raises and one opponent calls in full.

**Why:** The call-vs-raise and size decision is really about what the NEXT street looks like — landing the turn at SPR ~1 vs ~0.3 vs ~4 changes everything. Current dims describe only the present node; the net must simulate pot-limit arithmetic to plan one node ahead.

**Final architecture** (verdict: agree): 4 floats, tail append. With eff_to_call = min(to_call, eff_hero) and D = max_bet_total − street_commit_hero (the max-raise DELTA, recoverable from packed totals — no engine change): [0] log1p((eff_hero − eff_to_call) / max(pot + eff_to_call, 1)); [1] log1p((pot + eff_to_call)/bb); [2] log1p((eff_hero − D) / max(pot + 2D − to_call, 1)); [3] log1p((pot + 2D − to_call)/bb). Dims 0-1 always computed (to_call = 0 ⇒ after-check = current geometry, harmless); dims 2-3 zero when Raise illegal (same predicate as STK-2). Caller model for [2]-[3] is the idealized single full-match: pot' = pot + D + (D − to_call), i.e. the current aggressor (already at the bet level) matching hero's total. An all-in call/raise yields stack' = 0 ⇒ log1p(0) = 0 = 'no next-street geometry', which is the correct reading.

**Reviewer reasoning:** Agree — this is the strongest kind of feature per the project's own lessons: multi-step pot-limit arithmetic (subtract, add, divide) that an MLP must otherwise simulate to plan one node ahead, delivered in 4 log1p dims. The tightening makes the caller-match arithmetic explicit (pot + 2D − to_call) and caps the call by hero's effective stack so the all-in edge is exact rather than negative. Checks: (a) log1p unclipped, 6-way 300bb pot-after ≈ 900bb → log1p ≈ 6.8, fine; (b) raise-illegal zeros cover short/all-in states, HU fine, river rows stay well-defined (street one-hot lets the net discount them); (c) verified max_bet totals + street_commit are in the packed arrays — pure encoder math in all three paths; (d) global scalars, no rotation.

**Overlap with existing dims:** Composable from pot, to_call, max_raise, eff stack (all present) only via multi-step arithmetic; no direct dim exists.  
**Cost:** Pure scalar arithmetic in numpy; identical Rust; no engine change beyond max_raise in packed arrays.

### STK-6 · geometric_jam_plan — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Two planning scalars: bets_to_jam = ceil(log3(1+2*SPR_eff)) (successive pot-bet-and-call rounds until all-in, HU) and geometric_frac = ((1+2*SPR_eff)^(1/r)-1)/2 with r = streets remaining incl current — the per-street pot fraction that exactly jams by the river.

**Why:** Multi-street commitment planning is a streets x SPR INTERACTION: same SPR of 2.5 means one pot bet on the river but a 40%-pot geometric plan on the flop. Street one-hot and log-SPR exist separately; their multiplicative/exponential coupling is exactly what MLPs represent poorly.

**Final architecture** (verdict: agree): 2 floats, tail append. SPR_e = eff_hero / max(pot,1) (hero's effective stack, same operand as _SPR_LOG_OFF slot 0); r = streets remaining including current: flop 3, turn 2, river 1 (preflop rows impossible in bomb pots; emit r=4 arithmetic if ever hit, or zeros — pick one and pin it in all three encoders). [0] bets_to_jam = ceil(log3(1 + 2·SPR_e)) clip[0,6]; [1] geometric_frac = ((1 + 2·SPR_e)^(1/r) − 1)/2 clip[0,2]. SPR_e = 0 (hero all-in) ⇒ (0,0). Note: at the river geometric_frac == SPR_e and the [0,2] clip saturates for deep stacks — acceptable because values ≥ 2 all mean 'even a pot bet cannot jam this street' and the exact SPR rides unclipped at 1012; document that as intentional next to the offset.

**Reviewer reasoning:** Agree — verified the math: one pot-bet-and-call triples the pot, so n = log3(1+2·SPR); the geometric per-street fraction solves (1+2f)^r = 1+2·SPR. This is precisely the streets × SPR exponential interaction the encoding lessons call out as MLP-hostile, and it costs 2 dims of closed-form scalar arithmetic (np.power / powf) in all three encoders with zero engine work. Checks: (a) realistic maxima — deep 6-way SPR ≈ 14 → bets_to_jam ≈ 3.1, geometric_frac at flop ≈ 1.05; HU 300bb over a 6bb ante pot → SPR 50 → bets_to_jam 4.2 — neither clip binds except the documented river case; (b) all-in hero → zeros, HU fine, multiway is an idealized HU plan (a heuristic, fine as long as it's deterministic); (c)/(d) trivial, global.

**Overlap with existing dims:** Derivable from _SPR_LOG_OFF + street one-hot only through an exponential interaction; no existing dim carries it.  
**Cost:** Closed-form scalar math (one pow/log per row); np.power trivially; same in Rust.

### STK-7 · pot_ceiling_implied_odds — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero's maximum winnable pot if all remaining money goes in: ceiling = pot + sum over live opponents of min(eff_opp, eff_hero); encoded as growth multiple and as the best-case (implied) price of the current call.

**Why:** Drawing-hand continues beyond direct pot odds are governed by the implied-odds ceiling — a nut-draw call at bad direct price is right when 4x pot is still behind across two live stacks, wrong when everyone is near all-in. Direct pot odds (780) and eff price (1007) describe only money already in.

**Final architecture** (verdict: agree): 2 floats, tail append. ceiling = pot + Σ over live (non-folded) opponents, all-in included (they contribute 0), of min(eff_opp, eff_hero) — effective stacks throughout, so the block inherits dead-chip invariance. [0] log1p(ceiling / max(pot,1)); [1] implied_price = eff_to_call / (ceiling + eff_to_call) with eff_to_call = min(to_call, eff_hero); 0 when no bet faced. HU: single-term sum; everyone all-in ⇒ ceiling = pot ⇒ [0] = log1p(1), [1] = 0 — exactly the no-implied-odds reading.

**Reviewer reasoning:** Agree with two tightenings: use eff_to_call in the price (mirrors _EFF_PRICE_OFF's stack-capped convention) and pin all-in opponents as contributing 0 (their committed money is already inside pot). The sum-of-pairwise-min reduction is the correct side-pot-aware winnable ceiling and does the aggregation STK-3 spends 8 dims approximating per-seat; the best-case-price framing directly complements the existing worst-case-price dims at 1007. Checks: (a) log1p unclipped — 6-way 300bb gives ceiling/pot ≈ 70 → log1p ≈ 4.3, fine; (b) folded/all-in/HU all defined above; (c) one vectorized min + masked sum over packed stacks/folded in all three encoders, no engine change; (d) global aggregate, no rotation.

**Overlap with existing dims:** Ingredients in stacks/SPR blocks; the sum-of-pairwise-min reduction and price framing are new; partially kin to STK-3 (per-seat vs aggregate).  
**Cost:** One vectorized min+sum over packed stacks; trivial everywhere.

### STK-8 · Side-pot eligibility (winnable pot) — 3 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The portion of the current pot hero is actually eligible to win given all-in side-pot layering, plus dead-money fraction. Two definitions of eligibility (at current commit vs at full reach) and the after-call value.

**Why:** Multiway all-in tangles: pot odds on the FULL pot overstate hero's claim whenever a covering stack's commits exceed what hero can match; every lock/share feature should be read against the ELIGIBLE pot. Reconstructing side-pot structure from all-in mask + per-seat commits + stacks is deeply implicit; the eff-price block caps hero's PRICE by stack but never corrects the PRIZE.

**Final architecture** (verdict: pick): The stack_geometry arch (current-commit eligibility), tightened, 3 floats, tail append: [0] eligible_frac_now = Σ over ALL seats s (hero included, folded included — dead money is matchable up to level) of min(total_commit_s, total_commit_hero) / max(pot,1); [1] eligible_frac_after_call = same sum at hero level total_commit_hero + eff_to_call, where eff_to_call = min(max(bet_to_call − street_commit_hero,0), eff_hero); [2] dead_money_frac = Σ over folded seats of total_commit_s / max(pot,1). RAW commits everywhere (side-pot math is about actual chips in the pot; the dead-chip contract concerns stacks, not commits). No bet ⇒ [1] == [0]; no all-ins ⇒ [0] = [1] = 1.0 and the dims idle harmlessly; HU fine. Pure vectorized min/sum over packed total_commit/folded in all three encoders.

**Reviewer reasoning:** Pick over the dual_board full-reach variant for two reasons: (1) the immediate call/fold/raise decision is priced against what hero is eligible for NOW and AFTER THIS CALL — the full-reach ceiling answers a different (get-it-all-in) question that STK-7's pot-ceiling already frames from the money-behind side, so the dual variant is half-redundant with STK-7; (2) the current-commit form needs no per-seat excess-over-reach subtraction and degrades more gracefully multiway. Every lock/share feature in the v2 tail (991-999) implicitly assumes full-pot prize; this is the PRIZE correction to pair with the existing PRICE correction at 1007. Checks: (a) all [0,1] fractions; (b) edge cases above; (c) packed fields only, easy Rust; (d) global. Dim [2] duplicates HIST-12's dead-money scalar — if HIST-12 ships, drop [2] here (advice, not removal).

**All proposed architectures (pre-review):**

- *stack_geometry* (3 dims): 3 floats (current-commit eligibility): eligible_frac_now = sum_s min(commit_s, commit_hero)/pot; eligible_frac_after_call (hero commit + to_call); dead_money_frac (folded seats' total_commit incl ante share).
- *dual_board* (3 dims): 3 floats (full-reach eligibility): eligible_pot = pot - sum_s max(0, total_commit_s - (hero_total_commit+hero_stack)); [eligible_pot/max(pot,1), log1p(eligible_pot/bb), (pot-eligible_pot)/max(pot,1)].

**Overlap with existing dims:** MERGE of stack 'side_pot_structure' + dual_board 'eligible_pot_sidepot_block' (current-commit vs max-reach eligibility). dead_money_frac also in HIST-12; all-in mask (168), per-seat commits (918-934), eff-price (1007-1012) carry raw material; the eligibility reductions are new.  
**Cost:** Vectorized min/sum over commits + folded/stack mask; no sort needed for these aggregates; easy in Rust.

### STK-9 · call_risk_fraction — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The facing call priced against hero's OWN money: to_call as a fraction of hero's effective remaining stack ([0,1], 1=all-in call) and as a fraction of hero's total hand bankroll (to_call/(commit_hero+eff_hero)).

**Why:** At identical pot odds, calling off 5% of stack vs 60% are different decisions (re-raise exposure, realization); the eff-price block flags only the binary all-in-if-called endpoint and gives to_call in log-bb — the ratio to own stack is absent and is the natural [0,1] commitment-delta scale.

**Final architecture** (verdict: agree): 2 floats, tail append, both zero when no bet faced. eff_to_call = min(to_call, eff_hero): [0] eff_to_call / max(eff_hero, 1), raw [0,1] (1.0 = all-in call — the continuous version of the binary flag at _EFF_PRICE_OFF+1); [1] eff_to_call / max(total_commit_hero + eff_hero, 1), raw [0,1] — the commitment-DELTA this call represents, denominator matching dim 1009's raw-commit + eff-stack bankroll recipe exactly.

**Reviewer reasoning:** Agree — the eff-price block flags only the binary endpoint (to_call ≥ stack) and gives to_call in log-bb; the stack-relative ratio in between (5% vs 60% of stack at identical pot odds) is absent and is the natural [0,1] scale for re-raise exposure and realization decisions. Tightened to use eff_to_call and max(·,1) guards, and to pin dim [1]'s denominator to the 1009 recipe so the two commitment dims are arithmetically consistent (ratio [1] is literally the increment dim 1009 would gain on calling). Checks: (a) bounded fractions, nothing saturates; (b) hero acting always has stack > 0, guards cover degenerate zeros, HU fine; (c) two divides on values every encoder already computes (eff_per_seat, to_call, total_commit); (d) global.

**Overlap with existing dims:** _EFF_PRICE_OFF dims 1 (all-in flag) and 3 (log1p to_call/bb) are coarse/absolute versions; the stack-relative ratio is new format.  
**Cost:** Two scalar divides; trivial in all encoders.

### STK-10 · ante_pot_bloat — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Stake context: ante/bb (the bomb-pot stake parameter) and pot bloat = log1p(pot/pot_at_flop - 1) where pot_at_flop = dealt_seats x ante — how much betting inflated the pot beyond the forced starting pot.

**Why:** A 40bb pot means opposite things when the antes made it 18bb (one modest bet) vs 6bb (raise war) — aggression calibration and range-strength inference need the bloat factor, which today requires summing chips across up to 32 slots. Ante/bb pins tier identity directly.

**Final architecture** (verdict: agree): 2 floats, tail append: [0] ante/bb, raw (config scalar; bounded ~3); [1] pot_bloat = log1p(max(pot − pot_at_flop, 0) / max(pot_at_flop, 1)), with pot_at_flop = Σ over dealt seats of min(ante, resolved_stack_s) — pure config arithmetic (config.ante × config.resolved_stacks), frozen by construction, correct even for a sub-ante short stack, and identical in serial numpy, batched numpy, and the Rust encoder (the engine owns the config) with NO engine change and NO history dependence (so 32-slot truncation is irrelevant).

**Reviewer reasoning:** Agree with one load-bearing tightening: define pot_at_flop from config (Σ min(ante, stack)) instead of 'seats dealt × ante' — that removes the all-in-for-less edge case and, more importantly, removes any need to sum history chips (which the batched packer truncates to 32). Bloat is genuinely deep to recover today (a 32-term masked sum) and directly separates '40bb pot from 18bb antes' from '40bb pot from a raise war'. Caveat noted for the human pass: if training never varies ante (all tiers at 3bb), dim [0] is a dead constant like the actor rel-position one-hot — harmless at 1 dim and future-proofing for stake variation, but know that's what it is. Checks: (a) log1p unclipped, bloat at 6-way 300bb ≈ log1p(100) ≈ 4.6; (b) config-level, immune to folds/all-ins/HU; (c) config is already an argument to every encoder; (d) global.

**Overlap with existing dims:** History chips/bb slots contain bloat implicitly (32-term sum); ante pot also in HIST-9; ante/bb is currently absent everywhere.  
**Cost:** Config-derived scalars; trivial numpy; Rust needs ante + dealt-count in packed arrays (both known to the engine).

### STK-11 · per_seat_price_to_continue — 8 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The price EACH live opponent currently faces: per-seat effective amount owed min(max(bet_to_call - street_commit_s,0), eff_seat) as a pot-odds fraction, hero-rotated.

**Why:** Deny-equity raise sizing is about the prices OTHERS get: sizing so the short stack is all-in to continue, or so a multiway caller closes at a bad price. bet_to_call (scalar) and per-seat street commits exist, but the per-seat subtraction, stack cap, and normalization are left to the net; hero got a dedicated eff-price block, opponents got none.

**Final architecture** (verdict: agree): 8 floats, hero-rotated, padded to 8, tail append: slot k = owed_k / (max(pot,1) + owed_k) where owed_k = min(max(bet_to_call − street_commit_k, 0), eff_k) — each live opponent's stack-capped effective price in pot-odds form, [0,1]. Slot 0 (hero), folded seats, all-in seats, non-existent seats, and no-outstanding-bet states all read 0. owed_k == eff_k (the ratio's cap binding) is the 'all-in to continue' point, mirrored from hero's own eff-price convention at 1007.

**Reviewer reasoning:** Agree as proposed — hero got a dedicated stack-capped price block in the v2 tail while opponents got none, and deny-equity sizing is computed against the prices OTHERS face. The per-seat subtract/cap/normalize chain across bet_to_call (scalar 185), street commits (926-934), and stacks (176-184) is a real multi-block composition today. One honest limitation acknowledged: this encodes prices at the CURRENT bet level, and the sizing decision changes them — but current prices are the correct input; the counterfactual is the policy's job (and STK-2's ladder dims tell it what sizes are available). Checks: (a) [0,1] fractions; (b) all-in/folded/HU zeroing specified, and a live seat facing no bet reads 0 exactly like the hero pot-odds dim's convention; (c) pure elementwise math on packed arrays in all three encoders; (d) hero-rotated per spec.

**Overlap with existing dims:** bet_to_call (185), per-seat street commits (926-934), eff-price hero dim (1007) — per-seat opponent framing is new.  
**Cost:** Per-seat (x6) vectorized subtract/min/divide on packed arrays; trivial in all three encoders.


## Board texture & hand-board combinatorics

### BRD-1 · board_rank_ladder — 10 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The sorted (descending) ranks of the visible board cards, per board — the digested 'how high is this board' spectrum that pair-with-board counts index into but never expose.

**Why:** Value-bet/bluff-catch thresholds shift with board highness (A-K-x vs 8-6-2): the net must extract sortedness nonlinearly from the 52-multi-hot; pair_count slots say WHICH slots hero matches but not what rank those slots ARE, so 'top pair on a Q-high board' vs 'a 7-high board' is deeply implicit.

**Final architecture** (verdict: agree): 10 dims, tail append: per board, 5 scalars = (rank_of_ith_highest_visible_board_card + 1)/13, in the SAME rank-descending slot order as _PAIR_COUNT_*, trailing slots 0.0 pre-river and on empty boards. The +1/13 shift (instead of the proposed rank/12) keeps a deuce (0.077) distinct from the 0.0 empty-slot sentinel — the street one-hot would disambiguate globally, but per-slot self-describing values are cheaper for the net and cost nothing. Board A slots then board B slots.

**Reviewer reasoning:** Agree with the sentinel tightening. The claim that this is nearly free is verified: _pair_features_batch already computes sorted_desc (encoding.py ~line 1208) for the pair-count gather, so the batched path reuses it; the scalar path sorts ≤5 ints; Rust is a fixed-size sort. The value is real: pair_count slots tell hero how many hole cards match slot i but never what rank slot i IS, so 'top pair on Q-high vs 7-high' currently requires a nonlinear sort extraction from the 52-multi-hot — order statistics are exactly what MLPs do poorly. Sharing the slot ordering with _PAIR_COUNT_* makes the two blocks a keyed pair (count_i, rank_i). Checks: (a) bounded (0,1]; (b) board-only, immune to seats/HU; (c) all three encoders trivial, no engine change; (d) no rotation.

**Overlap with existing dims:** Implicit in board multi-hots (nonlinear sort); pair_count slots share the ordering but carry hero-match counts, not the ranks.  
**Cost:** Free in batched numpy — _pair_features_batch already computes sorted_desc; scalar path sorts 5 ints; trivial Rust.

### BRD-2 · board_suit_census — 10 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Board-only suit texture between existing thresholds: per-suit 'exactly 2 of this suit on board' flags (flush-draw-possible for anyone) plus a 'four-plus flush cards on board' flag, per board.

**Why:** Barrel planning and draw-denial sizing on two-tone boards need 'can villains hold flush draws here' and the 3-flush vs 4-flush distinction; flush_possible_per_suit only fires at >=3 and flush_draw_outs only when HERO holds the draw, so board-only two-tone/rainbow and 3-vs-4-flush reads are extractable only from the raw multi-hot.

**Final architecture** (verdict: agree): 10 dims, tail append: per board, 4 binary flags (board suit count == 2, per suit — per-suit identity kept so the net can join with hero's suit holdings) + 1 binary flag (any suit with ≥ 4 board cards). Empty/preflop boards → zeros. Board-only (no hero conditioning). A block then B block.

**Reviewer reasoning:** Agree as proposed. board_suit_counts exists in both python paths and the Rust encoder's port; ==2 and ≥4 are one comparison each. Honest assessment of depth: ==2 is a hat function (two ReLUs per suit) over a count that is linear in the multi-hot, so this is a shallow-ish feature — but the codebase's explicit convention is to hand exactly such thresholds over anyway (flush_possible ≥3 is already an explicit dim), and the two-tone/rainbow and 3-vs-4-flush distinctions gate barrel/draw-denial decisions on every flop. The ≥4 flag adds the flush-over-flush / 1-card-nut-geometry regime the ≥3 flag can't see. Checks: (a) binary; (b) board-only; (c) trivial in all three encoders; (d) no rotation.

**Overlap with existing dims:** flush_possible_per_suit covers the >=3 threshold; suit counts are linear in the multi-hot but ==2 and >=4 thresholds are not.  
**Cost:** Pure numpy — board_suit_counts already exists in scalar and batched; trivial Rust.

### BRD-3 · straight_pair_wetness — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Count of DISTINCT 2-rank combinations that complete a straight on this board (over all windows, deduped), per board — the canonical 'how many two-card hands just made a straight' wetness scalar.

**Why:** C-bet/check-back frequency is driven by completing-combo multiplicity, not just which windows are live; straight_possible_per_window is binary per window and misses the combinatorial blow-up on 4-and-5-rank windows, so wetness of connected boards is under-specified.

**Final architecture** (verdict: agree): 2 dims, tail append: per board, the raw count of distinct rank PAIRS {r1,r2} (r1≠r2, both ∈ some window W) such that W is completed under the engine's exactly-2-hole rule: (W − board_ranks) ⊆ {r1,r2} and |W − board_ranks| ≤ 2 — i.e. the pair covers everything the board is missing, deduped across the 10 windows. Board-only rank combinatorics (no unseen-copy weighting — consistent with straight_possible's board-only convention and DUAL-5's /78 pair-count sibling). Empty board → 0. Batched via the existing _PAIR_BITS_13 (78 masks) × _WINDOW_BITS_13 popcount machinery; scalar via the 10-window loop; Rust mirrors the bitmask form.

**Reviewer reasoning:** Agree, with the completion predicate made exact: the loose 'pair ∪ board ⊇ W' phrasing would miss the engine's makes-rule subtleties, so I pinned it to |W − B| ≤ 2 with the pair covering W − B (matching _straight_flush_features' L ⊆ H_W, nL ≤ 2 logic, including boards supplying 4 of a window where many pairs complete — which is precisely the combinatorial blow-up the binary straight_possible flags flatten). Raw count (~0-25) matches the encoding's raw-outs style. The infrastructure claim is verified: _PAIR_BITS_13 and the window bitmasks already exist for the cross-board straight block, so the batched implementation is a reuse, not new machinery. Checks: (a) bounded raw count; (b) board-only; (c) all three encoders straightforward; (d) no rotation.

**Overlap with existing dims:** Approximately equals sum of straight_possible_per_window on 3-rank flops; diverges (the new info) once windows hold 4-5 board ranks.  
**Cost:** Cheap numpy: 10-window loop over the existing 13-bit rank masks; straightforward Rust.

### BRD-4 · board_arrival_volatility census — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, forward-looking texture volatility: the count (normalized by unseen-deck size) of next-street cards that pair the board, make a flush newly possible/advance a suit, or enable a NEW straight window.

**Why:** Protection sizing, barrel planning, and charging draws depend on how violently the next card can change the nut landscape; every existing equity feature is CURRENT-RANK only, so 'static now but explosive next card' vs 'locked-texture' boards are indistinguishable without deep multi-hot reasoning.

**Final architecture** (verdict: pick): The dual_board arch (6 dims), tightened, tail append: per board 3 scalars, each a count of unseen cards (unseen = 52 − |hero hole| − |visible A| − |visible B|, from the existing visible_count tensor) divided by the unseen-deck size: [0] pair_outs = Σ over ranks r on this board of unseen copies of r; [1] flush_advance_outs = Σ over suits s with board suit count ∈ {2, 3} of unseen copies of s (2→3 makes a flush newly possible; 3→4 is the flush-over-flush escalation the board-source arch missed; 4→5 excluded as noise); [2] straight_advance_outs = Σ over off-board ranks r with ≥1 unseen copy such that some window W containing r goes from exactly 2 board ranks to 3 (newly-possible window). All three forced to 0 at the river (no next card). A block then B block.

**Reviewer reasoning:** Pick the dual_board variant: the two archs are otherwise the same census, and the 3→4 flush-advance nuance is a real decision input (a made flush facing a 4th suit card is the classic dual-board barrel/shutdown trigger) that the board-source arch's exactly-2-only definition drops. Tightenings: normalize by the actual global unseen count rather than an assumed constant, pin 'newly possible window' to the 2→3 transition so already-possible windows don't double-fire, and make the river zeroing explicit. Nothing in the current obs is forward-looking, so this is a genuine axis, and it is the cheap unconditional base BRD-5 subtracts from. Checks: (a) [0,1] fractions; (b) board-only, seat-independent; (c) computable from the existing (N,13,4) presence/visibility tensors in both python paths + the ported Rust encoder, no engine change; (d) no rotation.

**All proposed architectures (pre-review):**

- *board_texture* (6 dims): Per board 3 scalars: unseen cards making a flush possible (copies of exactly-2 suits), enabling a new straight window (rank taking a window 2->3), pairing the board — each /unseen count. 3 x 2 = 6.
- *dual_board* (6 dims): Per board 3 scalars: pair_outs, flush_advance_outs (incl 3->4 flush-over-flush pressure), straight_advance_outs, each /unseen deck. 3 x 2 = 6.

**Overlap with existing dims:** MERGE of board 'board_threat_census' + dual_board 'board_arrival_volatility' (near-identical; dual adds the 3->4 flush-over-flush nuance). None forward-looking today; straight/flush_possible describe the present, MC dims are current-rank.  
**Cost:** Cheap numpy from existing (N,13,4) presence/visibility tensors; established 10-window loop; easy Rust.

### BRD-5 · hero_vulnerability_outs — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero-conditioned threat counts, per board: flush-enabling cards in suits where hero holds <2, board-pairing cards of ranks hero holds zero copies of, and straight-completing cards that do NOT complete a straight for hero.

**Why:** Fast-play vs trap with made-but-fragile hands (top set on two-tone, dry-side nut straight) needs 'how many next cards demote ME specifically' — the board volatility census minus the cards that also help hero; no existing dim subtracts hero's participation, so vulnerable-nuts vs redraw-heavy-nuts read identically.

**Final architecture** (verdict: agree): 6 dims, tail append, same unseen-deck-size normalization and river-zeroing as BRD-4: per board [0] danger-flush outs = Σ over suits s with board count == 2 AND hero suit count < 2 of unseen copies of s (flushes hero cannot participate in); [1] danger-pair outs = Σ over ranks r on this board with hero rank count == 0 of unseen copies of r (board pairings that never fill hero); [2] danger-straight outs = count of unseen cards c such that on board+c some straight is possible/completed for the field (a window reaches ≥3 board ranks) AND hero does NOT make a straight on board+c (reuses the exact makes-window predicate from _straight_flush_features on the augmented rank mask). Pure set algebra on the existing presence/visibility tensors — no re-evaluation.

**Reviewer reasoning:** Agree. This is the hero-conditioned complement of BRD-4 and carries the actual decision signal (fast-play vs trap = how many next cards demote ME); shipping both lets the net read the delta directly, and every existing draw/outs feature measures only hero's upside, never downside exposure. Tightening: dim [2]'s 'does not complete for hero' is pinned to the same L ⊆ H_W / nH ≥ 2 / nL ≤ 2 makes-rule already implemented in scalar and batched form, so all three encoders share one predicate (Rust encoder port required, as for every accepted dim, since PLO5_RUST_ENCODER is now enabled and pinned). Checks: (a) [0,1] fractions; (b) hero+board only, seat-independent, HU irrelevant; (c) numpy from existing masks, cheap Rust; (d) no rotation.

**Overlap with existing dims:** BRD-4 is the unconditional superset; existing draw-outs describe hero's upside only, never the downside exposure.  
**Cost:** Cheap numpy; reuses straight_out_cands and suit/rank masks from the SF block; easy Rust.

### BRD-6 · straight_out_union — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board: hero's DISTINCT straight-out count (union of straight_out_cands across all windows weighted by unseen copies, deduped) and the subset of those outs after which hero holds the NUT straight.

**Why:** Semi-bluff stack-off and call-price decisions hinge on total wrap size and nut-vs-dirty out quality; the existing per-window outs double-count ranks shared between overlapping windows (overstating wraps) and there is NO straight analog of nut_flush_draw_outs, so a 20-out nut wrap and a 13-out dirty wrap are hard to tell apart.

**Final architecture** (verdict: agree): 4 dims, tail append: per board [0] union straight outs = Σ over the deduped union of straight_out_cands ranks across all 10 windows of (4 − global visible count of r), raw (0-20); [1] nut-straight outs = subset count of those out CARDS after which hero holds the top straight on board+c — i.e. hero makes some window W' on board+c and no strictly higher window is board-possible (≥3 board ranks) on board+c: straight_nut_distance == 0 re-evaluated on the augmented board, straight-type-local (ignores flush/boat, symmetric with nut_flush_draw_outs). Zero when no outs / river. A then B.

**Reviewer reasoning:** Agree. Two verified defects in the existing block motivate it: straight_outs_per_window double-counts ranks shared by overlapping windows (a wrap's true out count is unreadable without a dedup the net can't do across 10 slots), and nut-quality grading exists for flush draws (nut_flush_draw_outs) but has no straight analog — so 20-out nut wraps and 13-out sucker wraps read alike, which is exactly the multiway PLO disaster case. The union is a cheap dedup over straight_out_cands sets the scalar helper already caches (line ~741); the nut check is a bounded second pass (≤20 candidate ranks × 10 windows) on rank masks — implementable in scalar numpy, batched (augment the rank mask per candidate), and Rust ('moderate' as the cost note says, but mechanical). Checks: (a) raw bounded counts, house style; (b) board+hero only; (c) no engine change; (d) no rotation.

**Overlap with existing dims:** straight_outs_per_window carries raw material but double-counts and never grades outs by nut status; nut_flush_draw_outs is the flush-only analog; typed component of BRD-12's aggregate.  
**Cost:** Numpy: union is a dedup over already-computed straight_out_cands; nut check is an extra 10-window pass per out; moderate Rust.

### BRD-7 · boat_plus_outs — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, the count of unseen one-card completions that promote hero's best hand to full-house-or-better (0 when hero already holds FH+) — set/two-pair/trips redraw inventory against straights and flushes.

**Why:** Calling/raising with sets/two-pair on wet boards is priced by the boat redraw, and the draw-out features stop at straight/flush/SF — there is no full-house-direction out count anywhere, so 'top set with 7 boat outs on a monotone board' vs 'bare aces' differ only through slow multi-hot composition.

**Final architecture** (verdict: agree): 2 dims, tail append: per board, raw count (0-~10) of unseen cards c such that hero's best category on board+c ≥ full house, 0 when hero already holds FH+ on that board, forced 0 at the river. Implementation: ENGINE-EMITTED like the hero_category features — a Rust free fn (rank-count case analysis over candidate ranks; only ranks matching hero's pairs/board pairs can qualify, so the candidate set is tiny) surfaced via observation_dict + a batched array + the Rust encoder, with the serial/batched python encoders doing pass-through copy. This sidesteps the 'fiddly in numpy' problem entirely — the parity surface is one Rust function, mirroring how hero_category_a/b already flow.

**Reviewer reasoning:** Agree, with the delivery mechanism pinned to the engine-emit pattern rather than tri-implementing fiddly case analysis (the cost note itself points there; hero_category and opp_outcome_fractions are the precedent — engine computes, encoders copy, one implementation to keep bit-exact). The gap is real and important: draw features stop at straight/flush/SF, so there is no FH-direction redraw inventory anywhere, and set-with-boat-outs vs bare-overpair on wet boards is a core stack-off distinction. Typed component of BRD-12's deduped aggregate, but not subsumed: the FH+ threshold is specifically 'beats the straights/flushes I'm worried about', which the aggregate's any-improvement count blurs. Checks: (a) raw bounded count; (b) hero+board only; (c) engine change declared in the cost note, small; (d) no rotation.

**Overlap with existing dims:** Board pair structure + pair-with-board counts + rank histogram contain the ingredients implicitly; typed component of BRD-12's aggregate; no explicit FH-direction count exists.  
**Cost:** Closed-form rank-count case analysis fiddly in numpy; recommended a small Rust free fn (category_for style) enumerating candidate ranks; cheapest engine-side.

### BRD-8 · fd_rank_quality — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, for hero's live flush draw (hero >=2 of a suit with exactly 2 on board): the high card rank of hero's draw suit /12, and the count of unseen higher cards of that suit (draw-suit nut distance) — zero when no draw.

**Why:** Semi-bluff selection and flush-over-flush payoff avoidance need K-high vs 8-high draw separation; nut_flush_draw_outs collapses to {full,1,0} the moment 2+ higher cards are unseen, so ALL clearly-non-nut draws read identically.

**Final architecture** (verdict: agree): 4 dims, tail append: per board, for hero's live flush draw (a suit with hero count ≥ 2 AND board count == 2; if two suits qualify — hero 2+2 double-suited on a two-two-tone board — take the suit with the higher hole_max, the nut-most draw): [0] (hole_max_per_suit[draw_suit] + 1)/13 (the +1 shift keeps a 3-high draw distinct from the 0.0 no-draw sentinel); [1] raw count of unseen higher cards of that suit (0-11, the nut-distance of the draw). Both 0 when no qualifying suit. A then B.

**Reviewer reasoning:** Agree with two tightenings: an explicit tie-break for the double-qualifying-suit case the proposal ignored (hero can hold 2-2 in two suits with a board 2-2, so 'the draw suit' is otherwise ill-defined — highest hole_max is the decision-relevant pick), and the +1/13 sentinel shift consistent with my BRD-1/BRD-11 rulings. The cost claim is verified in the source: the batched SF path already computes hole_max_per_suit (~line 1285) and the unseen-above-h1 blocker counts (~line 1381), so both dims are reads of existing intermediates; scalar and Rust ports are trivial. The value claim also holds: nut_flush_draw_outs collapses to {full, 1, 0} the moment 2+ higher cards are unseen, so K-high and 8-high draws — opposite semi-bluff and stack-off decisions in PLO — currently read identically. Checks: (a) bounded; (b) hero+board only; (c) no engine change; (d) no rotation.

**Overlap with existing dims:** nut_flush_draw_outs covers only the nut/near-nut end; raw multi-hots carry rank quality implicitly.  
**Cost:** Free in numpy — hole_max_per_suit and unseen-above-h1 blocker count already computed in the SF batched path; trivial Rust.

### BRD-9 · backdoor_draw_census — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Two-card-runout draw inventory, per board (flop-relevant): count of suits where hero >=2 and board==1 (backdoor flush draws), and count of straight windows needing exactly two more board ranks that hero fills with 2 hole ranks.

**Why:** Bomb pots START at the flop, so every hand's first decision prices backdoor equity for peels and floats — yet all draw features (flags, per-window outs, SF outs) are strictly one-card-to-come, leaving two-card draw potential completely unencoded.

**Final architecture** (verdict: agree): 4 dims, tail append, HARD-GATED to street == Flop (explicit zeros on turn/river — the raw predicates would still fire on the turn where the draws are dead, so the gate is load-bearing, not cosmetic): per board [0] BDFD count = number of suits with hero count ≥ 2 AND board count == 1, raw (0-2 realistically); [1] backdoor-straight window count = number of windows W with |W − B_W − H_W| == 2 AND |H_W| ≥ 2 AND |W − B_W| ≤ 4 (both missing ranks must arrive turn+river and hero already covers the rest under the exactly-2-hole makes-rule; one-card draws have |W − B − H| ≤ 1 and stay in straight_outs), raw (0-10). A then B.

**Reviewer reasoning:** Agree — the motivation is airtight for this variant (every hand's FIRST decision is a flop decision, and all existing draw features are strictly one-card-to-come, so backdoor equity for peels/floats is completely unencoded), and the cost is a suit-count comparison plus one extra predicate inside the existing 10-window loop in all three encoders. My tightenings: derived the window predicate precisely from the engine's makes-rule (the proposal's 'needs exactly two more board ranks that hero fills with 2 hole ranks' is ambiguous about hole ranks duplicating board ranks — the |L − H| == 2 form handles it) and made the flop gate explicit rather than 'decay to 0 by the turn', which the BDFD definition does NOT naturally do (board count == 1 fires on turn boards where the draw is dead). Checks: (a) small raw counts; (b) hero+board only; (c) no engine change; (d) no rotation.

**Overlap with existing dims:** None — every existing draw dim requires the draw to complete with one card.  
**Cost:** Cheap numpy: suit-count comparison + one predicate inside the existing 10-window loop; easy Rust.

### BRD-10 · future_nut_flush_blocker — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, a flag/count over suits with exactly 2 on board where hero holds the highest-ranked card of that suit NOT on the board (the As-on-two-tone class: hero blocks the future nut flush without necessarily having the draw).

**Why:** Turn-barrel bluff selection when the third flush card arrives is classically keyed to holding the future nut-flush blocker; _blocker_features is explicitly zero until a flush is ALREADY possible (board>=3 of suit), so this signal appears one street too late for the decision it drives.

**Final architecture** (verdict: agree): 2 dims, tail append: per board, raw count (0-2) of suits s with board suit count == 2 where hero holds the KEY card of s: scanning ranks descending, skipping ranks whose (r,s) card is visible on EITHER board (those are unholdable by anyone — visibility-aware via the existing visible_count tensor, deliberately stronger than the proposal's board-local rule), the first remaining card is either in hero's hand (suit qualifies) or unseen (it does not). Forced 0 at the river (no third flush card can arrive). Note the deliberate divergence from _blocker_features' board-local convention documented at the offset: this block answers 'who can hold the future nut flush card', so cross-board visibility is the correct universe.

**Reviewer reasoning:** Agree with one substantive tightening: visibility-aware key-card resolution. Under the proposed board-local rule, As sitting on board B makes hero's Ks on a two-tone board A read as a non-blocker even though Ks IS the effective future nut-flush blocker there — on a dual-board game that misfire is common, and visible_count is already built in every encoder so the fix is free. The gap claim is verified in source: _blocker_features returns early unless the board already has ≥3 of a suit, so the As-on-two-tone barrel-selection signal genuinely appears one street too late today. River-zeroing added since a 2-suit board at the river has no future. Checks: (a) 0-2 raw; (b) hero+board only; (c) suit counts + presence masks exist in all three paths, no engine change; (d) no rotation.

**Overlap with existing dims:** blockers-to-nuts (v2 tail) is the post-arrival version; multi-hots implicit otherwise.  
**Cost:** Free in numpy from board suit counts + hero presence masks; trivial Rust.

### BRD-11 · turn/river card identity (deal order) — 20 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Which card ARRIVED on the turn and river, per board — the board multi-hot is a set and destroys deal order, so 'the flush card just hit' vs 'it was there on the flop' is currently invisible within a single observation.

**Why:** Range inference is order-dependent — villain's flop calls were made against the flop texture, so betting after a texture-changing turn differs from static-texture turns even when the final card SET is identical; action history records streets but never cards, so arrival order exists nowhere.

**Final architecture** (verdict: pick): The compact board_texture arch (20 dims), tightened, tail append: per board, per arrival slot {turn, river}: [(rank+1)/13 scalar, suit one-hot (4)] = 5 dims, all-zero until that card is dealt. The suit one-hot doubles as the dealt flag; the +1/13 rank shift keeps a deuce distinct from the undealt 0.0 sentinel. Order: A-turn(5), A-river(5), B-turn(5), B-river(5). Source is the deal-ordered board arrays — serial dict lists append turn/river, and the batched (N,5) u8 arrays carry turn at column 3 and river at column 4 — so this is a re-encode of existing inputs with no engine change; both boards deal their turn/river simultaneously, so dealt-ness is also implied by the street one-hot (harmless redundancy).

**Reviewer reasoning:** Pick the 20-dim compact form over the 68-dim full one-hot: the multi-hot already carries exact card identity, so this feature only needs to POINT at which set members arrived late — a rank scalar + suit one-hot binds to the multi-hot cheaply, and 68 dims of duplicate one-hot encoding is disproportionate for a pointer. The underlying gap is real and verified: boards are sets in the observation, action history records streets but never cards, so 'the flush card just hit' vs 'it was there all along' — which conditions every range-read of villain's earlier calls — exists nowhere. Checks: (a) bounded, one-hot + (0,1] scalar; (b) board-only, street-gated by construction; (c) deal order confirmed available in scalar dict, batched arrays, and the engine (Rust encoder reads its own deal-ordered state); (d) no rotation.

**All proposed architectures (pre-review):**

- *board_texture* (20 dims): Per board, per slot {turn,river}: rank/12 scalar (1) + suit one-hot (4) = 5; zero until dealt. 5 x 2 slots x 2 boards = 20.
- *dual_board* (68 dims): Per board: turn card 13-rank one-hot + 4-suit one-hot (17), river same (17). 34 x 2 boards = 68. (Heavier full-one-hot format.)

**Overlap with existing dims:** MERGE of board 'turn_river_identity' (compact scalar-rank) + dual_board 'board_card_arrival_order' (full one-hot). Card identities are in the multi-hots; arrival ORDER exists nowhere.  
**Cost:** Free in numpy (board arrays are deal-ordered in scalar dict and batched (N,5) arrays); trivial Rust.

### BRD-12 · hero_improve_outs_dedup — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board: the count of DISTINCT unseen cards that strictly improve hero's made-hand category (union across ALL types incl trips->boat, set->quads), plus a redundancy counter of how many of hero's 10 two-card combos achieve hero's best category.

**Why:** Semi-bluff selection and counterfeit-robust calling — total live improvement mass (not per-type) sets bluff-raise equity, and combo redundancy determines whether a board-pairing turn destroys hero's hand. Existing outs cover straight/flush/SF only, can double-count across types, and carry zero boat/quads info.

**Final architecture** (verdict: agree): 4 dims, tail append, ENGINE-EMITTED (Rust free fn + observation_dict key + batched array + Rust-encoder read; python encoders pass through, mirroring hero_category/opp_outcome_fractions): per board [0] improve_outs = count of DISTINCT unseen cards c such that hero's best hand CATEGORY on board+c strictly exceeds his current category on that board (union across all improvement types incl trips→boat, set→quads), normalized by the ACTUAL unseen-deck size (41 flop / 39 turn — not the proposal's fixed /45, keeping the convention consistent with BRD-4/5), forced 0 at river; [1] best_cat_combo_count = number of hero's C(5,2)=10 hole pairs achieving his current best category on that board, /10 (counterfeit-redundancy; well-defined at every street incl river). Order: outs_A, outs_B, combos_A, combos_B.

**Reviewer reasoning:** Agree with two tightenings: the unseen-count normalizer (a fixed /45 is wrong at every street and inconsistent with the other census features) and explicit river zeroing for the outs dims while the redundancy dims stay live. Scope note pinned: category-strict improvement only — a better two-pair doesn't count; that's the spec'd tradeoff, and the typed siblings (BRD-6 nut-quality, BRD-7 FH-threshold) carry the quality axes this aggregate deliberately blurs, so all three coexist without redundancy-rule violations. Cost is honest engine work (~2×41 hero re-evals per obs, deterministic, no RNG), correctly declared as not-practical-in-numpy and following the established engine-emit pattern so tri-encoder parity is one function. Checks: (a) [0,1]; (b) hero+board only; (c) engine change declared; (d) no rotation.

**Overlap with existing dims:** Deduped aggregate of which BRD-6 (straight) and BRD-7 (FH) are typed components; straight/flush/SF outs cover two types without dedup; redundancy appears nowhere.  
**Cost:** New Rust computation ~90 partial evals/obs (cheap, deterministic); not practical in numpy — engine-emitted like category features.

### BRD-13 · board_nut_ceiling_class — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, the best hand CATEGORY any holding could have right now (the nut ceiling): one-hot over 9 classes computed from board structure — SF-possible (suit-restricted window >=3, encoded nowhere), quads/boat/flush/straight-possible, else trips-floor.

**Why:** Polarization targeting and thin-value ceilings per board — what 'the nuts' IS on each board determines whether hero's flush is effectively top or crushed by boats, and the ceiling DELTA between boards tells which half villain's nutted range can attack. Today the net reconstructs it from scattered paired/flush/straight flags — a multi-input AND/max, and the SF-possible conjunction is genuinely absent.

**Final architecture** (verdict: better): 4 dims (replacing the proposed 18), tail append: per board [0] sf_possible = 1 iff some straight window W and suit s have ≥3 board cards of suit s at ranks in W (board-only, ignores hero blockers, consistent with the flush/straight_possible conventions — this conjunction is the genuinely-absent bit; computable by running the existing window machinery on the per-suit board masks, which the batched path already builds as board_rank_suit (N,13,4)); [1] ceiling_class = ordinal scalar /8 of the best achievable category from board structure: SF(8) if sf_possible, else quads(7) if the board is paired (a pocket pair of the paired rank makes quads under the exactly-2 rule, so paired ⇒ quads dominates FH and FH never wins the argmax), else flush(5) if any flush_possible, else straight(4) if any straight_possible window, else trips(3) — the ordinal form makes the A-vs-B ceiling DELTA (which half villain's nutted range can attack) a linear readout instead of a 9-way one-hot comparison. Empty/preflop board → 0/0. Pure mask algebra in all three encoders, no engine change.

**Reviewer reasoning:** The proposed 18-dim one-hot is mostly re-packaging of dims the net already has in nearly-linear form: paired/double/tripled/quadded (813-821) pins the quads/FH tier, flush_possible_per_suit (856-860/894-898) and straight_possible_per_window pin the middle tiers, and the priority-argmax over four binary flags is a couple of hidden units — the only genuinely new information in the whole idea is the suit-restricted SF-possible conjunction, which no existing dim carries and which decides whether hero's nut flush is actually the board ceiling. So: extract the new bit explicitly (2 dims) and keep the digest as a compact ordinal scalar (2 dims) that preserves the idea's cross-board ceiling-delta rationale at 1/4 the width and with better geometry (ordinal subtraction beats one-hot comparison for a delta). Checks: (a) binary + bounded ordinal in [3/8,1]; (b) board-only; (c) window×suit masks exist in all paths; (d) no rotation.

**Overlap with existing dims:** flush_possible_per_suit, straight_possible_per_window, board pair-structure are the ingredients; the argmax digest and SF-possible conjunction are new.  
**Cost:** Cheap numpy mask algebra, fully vectorizable; easy Rust. No engine change.


## Double-board structure

### DUAL-1 · Split-adjusted price ladder — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Pot odds re-denominated to the split-pot outcome ladder: the call price if hero wins exactly one board (to_call/(0.5*pot+to_call)) and if quartered (to_call/(0.25*pot+to_call)), raw and effective-stack-capped.

**Why:** The canonical double-board leak is calling at full-pot odds when the hand realistically wins only half, and the modal outcome per the engine's own counters IS win-one; existing pot-odds/bet-faced/eff-price all use the SCOOP denominator. Lets the net read 'price vs my likely share' directly (pairs with the joint outcome / win-one-tie-both MC dims).

**Final architecture** (verdict: better): 2 floats, encoder-computed identically in serial numpy, batched numpy, and the Rust encoder, appended to the obs tail: [0] eff_half_price = eff_to_call/(0.5*pot + eff_to_call); [1] eff_quarter_price = eff_to_call/(0.25*pot + eff_to_call); eff_to_call = min(max(bet_to_call - hero_street_commit, 0), hero EFFECTIVE stack) — the exact operand already built for _EFF_PRICE_OFF dim 0. Both dims 0 when eff_to_call == 0. Range (0,1) by construction, no clip, no saturation at 300bb (denominators strictly positive: pot >= ante pot postflop). HU / all-in / fold states need no special casing (pure scalar arithmetic on values every encoder already has).

**Reviewer reasoning:** The merge of the two sources is right, but neither listed arch is: the 4-dim ladder double-encodes each rung as raw AND effective, and the obs-v2 lesson (the eff-price block at 1007 was added precisely because raw pot odds at 780 overstate the price when a PL bet covers hero) says new price features should be effective-stack-capped only — the raw form is the known-inferior format and already exists at 780/990 for the full-pot denominator. Keep both rungs (0.5 = the modal win-one outcome per the engine's own counters; 0.25 = quartered), drop the raw twins: 2 dims. Division is exactly what MLPs compose poorly, so this redundant re-denomination is justified despite being arithmetic on existing dims, and it pairs directly with the win-one/tie-both dims at 997-998.

**All proposed architectures (pre-review):**

- *dual_board* (4 dims): 4 dims: half_price=to_call/(0.5*pot+to_call), quarter_price=to_call/(0.25*pot+to_call), eff_half_price, eff_quarter_price (eff_to_call variants); 0 when no bet.
- *stack_geometry* (1 dims): 1 dim: eff_to_call/(pot/2+eff_to_call) [0,1] (the half-pot / win-one price alone, stack-capped); 0 when no bet.

**Overlap with existing dims:** MERGE of dual 'split_adjusted_price_ladder' + stack 'halfpot_price_dual_board' (the fuller ladder includes the exact half-pot dim). Pot odds (780), bet-faced (990), eff-price (1007) — same ingredients, split denominators new. Explicitly welcomed redundant format.  
**Cost:** Free: pure numpy scalar arithmetic on values already computed in all three encoders.

### DUAL-2 · Best-hand card usage / cross-board coverage — 10 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** How hero's 5-card hand distributes across the two boards: which exactly-2 hole cards form hero's best hand on each board, and the resulting overlap (0/1/2 shared cards) plus count of distinct hole cards working across both.

**Why:** Dual-board commitment needs card-overlap: a set on A and a straight on B using DISJOINT pairs is a far more robust scoop threat than both leaning on the same two cards, and folding a card load-bearing on both boards differs from folding danglers. Made-hand one-hots say WHAT hero has per board but never WHICH cards make it; the cross-board block covers same-pair straight/flush coincidences only, leaving sets/two-pair/boats uncovered.

**Final architecture** (verdict: pick): The board_texture arch: 10 binary dims, hero-only (no rotation needed). Per board {A,B}, 5 slots aligned to hero's hole cards sorted by card INDEX descending (pin this as the canonical multiset order, matching the project's fixed-ordering contract); slot i = 1 iff hole card i is one of the exactly-2 cards of hero's best holding on that board. Engine plumbing (honest in the cost note): the evaluate_plo5_partial pair iteration additionally tracks the argmax pair, tie-broken to the lexicographically smallest (lo_index, hi_index) pair among rank-ties — pin the tie-break in a parity test; expose via observation_dict + a new batched array + the Rust encoder. Exactly two bits set per board whenever a board exists (always, in bomb pots); all-zero only in degenerate no-board states. Fold/all-in/HU independent (hero-only feature).

**Reviewer reasoning:** The raw masks strictly subsume the 4-dim overlap digest — |intersection| in {0,1,2} and distinct-card count are one-layer reductions (elementwise AND + sum) of the two 5-bit masks — while additionally joining card IDENTITY with everything else keyed on those cards (blockers-to-nuts, draw suits, rank histogram): 'the two cards making my hand on A are also my blockers to B's nuts' is readable only from masks. Engine plumbing cost is identical for either arch (both need the argmax-pair accessor), so take the richer format for +6 dims. The tie-break canonicalization is the one correctness trap; it must be pinned or serial/batched/Rust parity breaks.

**All proposed architectures (pre-review):**

- *board_texture* (10 dims): Per board 5 binary dims, slot i=1 iff hero's i-th hole card (canonical descending order) is in the best 2-card holding. 5 x 2 = 10 raw masks; net intersects/unions the two masks.
- *dual_board* (4 dims): 4 dims: 3-way one-hot of |best_pair_A ∩ best_pair_B| in {0,1,2} + (distinct hole cards used across both)/4. (Pre-computed overlap digest.)

**Overlap with existing dims:** MERGE of board 'best_hand_usage_mask' (raw masks) + dual 'best_combo_board_coverage' (overlap digest derivable from the masks). hero_cat gives class per board; cross-board straight/flush made-both flags cover two same-pair cases only.  
**Cost:** ENGINE: eval already iterates all C(5,2) pairs x board triples — expose the argmax pair (new accessor via observation_dict/arrays); tie-break canonicalization (lowest card-index pair) must be pinned; numpy encoders scatter it.

### DUAL-3 · Exact nut-lock / freeroll flags — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Crisp exact indicators per board from the k=2 exhaustive pass: pure nuts (behind==0 AND tie==0), nut-or-chop (behind==0), near-nuts (behind<=1%), plus the dual-board aggregates 'locked both' (scoop lock), 'locked at least one', and freeroll (cannot lose one board while live on the other).

**Why:** PLO dual-board aggression pivots on exact-zero risk — freeroll (lock one board, gamble the other) is THE double-board raising license, and folding is never correct on a locked half. The per-board ahead/tie/behind fractions (991-999) carry this only as an exact-zero threshold on a float, and a smooth MLP cannot represent the EV cliff between behind=0.000 and 0.002 sharply. Directly relevant to the observed lock-fold probe behavior.

**Final architecture** (verdict: pick): The dual_board 6-dim arch, computed ENCODER-SIDE in all three encoders (no engine change): gate active = (obs[991]+obs[992]+obs[993]) > 0.5 (the per-board fractions sum to 1 when the k=2 pass ran; all-zero at terminal); then pure_nut_A = active AND behind_A==0.0 AND tie_A==0.0; nut_or_chop_A = active AND behind_A==0.0; same for B; locked_both = nut_or_chop_A AND nut_or_chop_B; locked_at_least_one = OR. The ==0.0 tests are EXACT: the fractions are integer-counter x (1/n) in f32, so 0.0 iff counter==0 (no epsilon needed, no underflow at n<=820). 6 binary dims, zeros when inactive. Freeroll-ish = locked_at_least_one AND NOT locked_both falls out as a conjunction of two emitted dims.

**Reviewer reasoning:** Beats the equity_mc 5-dim arch because (a) the strict-nuts vs nut-or-chop distinction matters in a split-pot game (chop-or-better = never fold; pure nuts = freeroll-raise license) and the equity arch collapses it, and (b) near_nuts (behind<=1%) is a SOFT threshold of the continuous behind_frac already at 993/996 — nets learn soft thresholds fine; only the exact-zero EV cliff is unrepresentable, which is the entire justification for these flags. Beats the board 4-dim arch by carrying the dual-board aggregates. Near-free, no engine work, and it is the single most decision-relevant item in this chunk given the observed lock-fold probe behavior (deep locks folding 36-54%). The mandatory tightening is the activity gate — naive ==0 tests would read terminal all-zeros as 'nuts'.

**All proposed architectures (pre-review):**

- *dual_board* (6 dims): 6 dims: pure_nut_A (behind_A==0 && tie_A==0), nut_or_chop_A (behind_A==0), pure_nut_B, nut_or_chop_B, locked_both, locked_at_least_one. Exact ==0 thresholding of integer per-board counters.
- *equity_mc* (5 dims): 5 dims: nuts_A (behind==0), nuts_B, near_nuts_A (behind<=1%), near_nuts_B, freeroll (behind==0 on one board while live on the other).
- *board_texture* (4 dims): 4 dims: strict nuts (behind==0 && tie==0) + chop-or-better (behind==0), per board (nuts_now_flags; proposer tagged equity_mc — refiled to dual_board here).

**Overlap with existing dims:** MERGE of dual 'per_board_nut_lock_flags' + equity 'nut_lock_freeroll_flags' + board 'nuts_now_flags' (all exact thresholds of the k=2 behind/tie counters; refiled the equity_mc-tagged one into dual_board). Deterministic function of per_board_outcome (991-999) — deliberate redundant sharpening; k=2 scoop fraction approaches 1 near the scoop lock.  
**Cost:** Near-free: numpy thresholding of already-computed per_board_outcome values (or emit from the same Rust counters for robustness); no new engine work.

### DUAL-4 · guaranteed_pot_share_block — 5 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Hero's worst-case and best-case fraction of the final pot over the k=2 exhaustive opponent universe at current rank (share = 0.5*win_A+0.25*tie_A+0.5*win_B+0.25*tie_B; g_min = min over combos, g_max = max), plus expected share and a price on the contested slice.

**Why:** Call/raise pricing when part of the pot is already decided — with the nut flush on A and air on B a pot-size call risks to_call to fight over only the contested half; hero should compute odds against pot*(g_max-g_min), not the full pot. Pot odds, eff-price, and SPR all use the FULL pot; per-board marginals cannot yield a min-over-combos (a min is not any marginal), so the guaranteed floor is absent.

**Final architecture** (verdict: better): 5 dims (drops E_share). Engine: inside the existing k=2 exhaustive loop compute per-combo share s = 0.5*[cmp_a==-1] + 0.25*[cmp_a==0] + 0.5*[cmp_b==-1] + 0.25*[cmp_b==0] and track g_min/g_max (two free trackers; dims 0..20 of the fused output stay byte-identical — pure append, P1-style pinning; cache key unchanged since g depends on the same inputs outcome_seed already hashes). Encoder dims: [0] g_min, [1] g_max (both quantized in {0,.25,.5,.75,1}); [2] flag g_min >= 0.5; [3] price_contested = eff_to_call/(pot*(g_max-g_min) + eff_to_call), set 0 when eff_to_call==0 OR g_max==g_min; [4] log1p(eff_hero_stack / (pot*(g_max-g_min))), set 0 when g_max==g_min (sentinel — avoids a meaningless ~15-nat blowup on fully-decided pots; the [2] flag plus DUAL-3 disambiguate that state). All five zero when the outcome block is inactive. log1p, unclipped, per the deep-tier saturation lesson.

**Reviewer reasoning:** The core insight is sound and verified against the Rust loop: a min over combos is not recoverable from any marginal, and g_min >= 0.5 is NOT equivalent to locked-at-least-one (hero can be unlocked on both boards yet never share-below-half), so the floor is genuinely new. But E_share is EXACTLY linear in dims 991-996 (0.5*aheadA + 0.25*tieA + 0.5*aheadB + 0.25*tieB) — a single linear layer recovers it perfectly, and the proposal itself admits this; dropping it saves a dim with zero information loss. The two sentinel rules for the degenerate g_max==g_min case are required — the proposed max(pot*dg,1) guard in raw chips (bb=10000) otherwise produces a wild outlier dimension exactly on locked pots.

**Overlap with existing dims:** E_share is exactly linear in dims 991-997; g_min at 0.5 co-fires with DUAL-3 nut-lock flags; price recombination overlaps eff-price (1007) in format only.  
**Cost:** Two extra counters in the already-running k=2 loop (no new evals, bit-identical existing outputs); trivial encoder mirror.

### DUAL-5 · villain_cross_board_coverage — 9 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Board-only (hero-independent) cross-board coverage: per suit, both boards have >=2 (one 2-flush-card holding draws at both) and both >=3 (one holding can flush both); per straight window, straight-possible on BOTH; plus the count of 2-rank pairs completing a straight on A and on B (villain scoop-straight density).

**Why:** Bluff-catch and value-sizing calibration to villain's scoop geometry — when a single holding class can cover both boards, villain's big raises are credibly scoop-polar and hero's win-one holdings shrink in value; texture-orthogonal boards cap large aggression to one half. The existing cross-board block describes HERO's coverage (requires hero holds >=2), never the field's.

**Final architecture** (verdict: better): 9 dims, board-only, hero-independent, encoder-computed (scalar path: suit counts + the 10-window loop; batched: the existing _PAIR_BITS_13 / _WINDOW_BITS_13 bitmask machinery; mechanical Rust-encoder port): [0..4) per suit s: 1 iff board_A count_s >= 2 AND board_B count_s >= 2 (one 2-card suited holding has flush-draw-or-better interest on both); [4..8) per suit s: 1 iff both counts >= 3 (a single holding can make flushes on both); [8] scoop-straight density = |{2-rank pairs completing a straight on A AND completing one on B}| / 78 (reuse the DUAL-cross pair-bitmask enumeration with per-board coverage tests, hero mask replaced by all-ones). Well-defined from the flop (both boards always >= 3 cards in this variant); no rotation, fold/all-in/HU independent; no engine change. Dropped from the proposal: the 10 per-window straight-possible-on-both one-hots.

**Reviewer reasoning:** The field-coverage concept is objective and genuinely absent (the 963-977 cross block is hero-conditioned), but 14 of the proposed 19 dims are single-AND compositions of existing aligned binaries — the 10 window dims are literally elementwise ANDs of straight_possible_per_window A (846-856) and B (884-894), the shallowest possible composition — so 19 dims is disproportionate sharpening. The genuinely deep parts survive: the ==2-both suit thresholds exist nowhere (flush_possible only fires at >=3), and the both-boards completing-pair count is a real combinatorial reduction. The per-suit >=3-both flags are kept despite being ANDs because they align the block with the hero-involved made-both layout at 963. /78 normalization cannot saturate (max is all 78 pairs, never reached).

**Overlap with existing dims:** Hero-involved cross flags (963-977) are the hero-conditioned special case; per-board flush/straight_possible masks supply the AND inputs; the both-boards pair-count is new.  
**Cost:** Pure numpy from existing masks in both python encoders (the batched pair-bitmask helper exists); mechanical Rust port.


## Monte-Carlo / equity extensions

### EQ-1 · nut_combo_density — 2 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, the fraction of unseen 2-card combos whose best hand achieves the maximum achievable combo rank (how many distinct holdings ARE the current nuts: 16-combo nut-straight boards vs 1-combo top-set-is-nuts boards).

**Why:** Bluff-catch pricing and raise-credibility depend on nut density — behind_frac tells hero's percentile but says nothing about how much combo mass sits AT the top, so 'villain repping the nuts' is plausible on JT-heavy boards and near-impossible on dry ones with no way to read the difference.

**Final architecture** (verdict: better): 2 dims, per board: RAW COUNT (not fraction) of unseen k=2 combos whose rank on that board equals the maximum combo rank — running (max_rank, count) trackers inside the existing k=2 exhaustive loop (if r > max {max=r; count=1} else if r == max {count+=1}); appended to the fused Rust output after dim 19 (dims 0..20 byte-identical, cache key unchanged — same inputs as outcome_seed). Typical range 1-16 (two-rank nut straights = 16 combos; paired-board nut boats/quads = 1-4), no clip, matching the raw-count convention of the straight-outs block (836-846). Zeros when the pass is inactive (terminal). Serial/batched encoders and the Rust encoder all read the widened slab pass-through.

**Reviewer reasoning:** The idea is right and near-free (verified: the k=2 loop already computes every combo's per-board rank at line ~1444 of engine.rs), but the proposed [0,1] fraction normalization is defective: nut-combo mass tops out around 16/820 ~ 0.02, so the feature would live in [0.001, 0.02] — wasted dynamic range at the un-normalized input layer, the same class of silent information deletion as the [0,4] SPR clip. Raw count is the established convention for combinatorial inventories here. Genuinely new information: behind_frac gives hero's percentile, never the mass AT the maximum.

**Overlap with existing dims:** per_board ahead/tie/behind gives hero-relative fractions, not the density at the maximum.  
**Cost:** ENGINE (Rust): marginal extension of the fused exhaustive k=2 pass (running max combo rank + count of sharers over the stored ~1081 ranks); near-free relative to the existing pass.

### EQ-2 · nut_retention_count — 2 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, the fraction of unseen next cards c after which hero still holds the chop-or-better nuts on board+c (full re-evaluation: hero may improve, the nut class may rise) — the exact one-street lock-durability of hero's holding.

**Why:** The fast-play-vs-trap decision with current nuts (THE recurring spot: nuts on A, marginal on B) is exactly 'how many turns keep me unbeatable', which the current-rank equity dims cannot see and the threat censuses only approximate (they count card types, not whether hero survives them).

**Final architecture** (verdict: better): 2 dims, per board: fraction of unseen next cards c after which hero still holds the chop-or-better nuts on board+c, i.e. hero_rank(board+c) >= max combo rank(board+c). Rust-only, exhaustive over unseen c (~41-45): per candidate, incremental hero re-eval PLUS a CONSTRUCTIVE nut-rank (case ladder SF -> quads -> boat -> flush -> straight -> trips over board+c structure, ~10-30 candidate evals per card, exhaustively parity-pinned against brute C(n,2) enumeration in tests) instead of the naive per-card full universe. Cost ~2-3x the current fused pass at flop. Zeros at the river (DUAL-3 covers the terminal street) and when the pass is inactive. Emitted in the fused slab; cache key unchanged.

**Reviewer reasoning:** The decision it targets (fast-play vs trap with current-but-fragile nuts) is real and forward-looking equity durability exists nowhere. But the arch as proposed is uncosted for good reason: the naive form is ~40-50x the current pass (each candidate card needs its own ~C(44,2) nut determination), against an MC pass already measured at ~32% of rollout wall-clock. Even my constructive-nuts redesign is ~2-3x the pass AND introduces a correctness-critical new Rust module (an exact nut constructor) whose parity burden is high.

**⚠ Remove recommended:** Cost grossly disproportionate to the decision it changes: naive form 40-50x the fused pass (which is ~32% of rollout), and even the constructive-nut redesign is 2-3x plus a high-risk exact-nuts Rust module. The signal is approximated near-free by DUAL-3 (locks now), BRD-4/BRD-5 (card-type threat/vulnerability censuses), and EQ-1 (nut density). Revisit only with profiled headroom after the EQ-4 family lands.

**Overlap with existing dims:** BRD-4/BRD-5 are cheap card-type approximations; DUAL-3 is the zero-lookahead special case; nothing existing looks forward.  
**Cost:** ENGINE (Rust), priciest of the board proposals: ~41 candidate cards x (hero re-eval + max-combo-rank check) per board — tens of times the k=2 pass unless incrementalized; must be costed/profiled before adoption.

### EQ-3 · joint_board_outcome_matrix — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The full 3x3 joint distribution of (hero ahead/tie/behind on A) x (on B) over the k=2 exhaustive universe — 9 cells, of which 4 dof are genuinely new: win-A-lose-B vs win-B-lose-A attribution, and the tie-one-win-other vs tie-one-lose-other (3/4 vs 1/4 quarter geometry) cells.

**Why:** Which board carries the barrel, and quarter-avoidance — calling when hero's likely outcome is 3/4 (tie A, win B) is mandatory, at 1/4 (tie A, lose B) a disaster; WHICH board hero's win-one lives on couples with per-board volatility. Existing dims give only marginals + AA (scoop) + TT (tie-both) + SUM win-one; attribution and tie-cross cells are unrecoverable from any linear combination.

**Final architecture** (verdict: better): 6 dims: the six OFF-DIAGONAL cells of the hero-perspective 3x3 joint, as fractions of the k=2 exhaustive universe — [WT, WL, TW, TL, LW, LT] where cell XY = P(hero X on board A, hero Y on board B), X,Y in {W=ahead, T=tie, L=behind}; the diagonal WW/TT/LL is OMITTED because those cells already exist verbatim (WW = scoop_hero at dim 980, LL = scoop_opp at 978, TT = tie_both at 998). Free counters inside the existing k=2 loop (a 3x3 tally indexed by (cmp_a, cmp_b)); appended to the fused slab, dims 0..20 byte-identical, cache key unchanged. Self-check: the 6 new dims + the 3 existing diagonals sum to 1 when active. All-zero when the pass is inactive (terminal). Encoders are pass-through.

**Reviewer reasoning:** The concept is right but the proposal's '4 dof genuinely new' claim is wrong — I verified by linear elimination that given the marginals (991-996), the k=2 joint row (quarter_hero/quarter_opp at 979/981), win-one (997), and tie-both (998), the full matrix has exactly ONE free dof (any single cross cell determines the rest). So the value here is mostly FORMAT: handing the net every attribution (which board carries the win; 3/4-vs-1/4 tie geometry per side) directly instead of via 10-term signed linear recombination. Given that, the 9-cell version wastes 3 dims on exact duplicates; the 6 off-diagonal cells keep full directness at zero duplication and zero compute (counters only). Worth shipping: 1 real dof + direct-read format, free.

**Overlap with existing dims:** Marginals = 991-997; AA cell = k=2 scoop; TT = tie_both (998); AB+BA sum = win-one (997). 4 cross-attribution dof new. Current-rank version of EQ-12 (runout joint).  
**Cost:** Free counters inside the existing Rust k=2 loop (exactly how obs-v2 P1 dims were added); encoder side is a copy.

### EQ-4 · River-runout equity vs one opponent — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Showdown-projected (not current-rank) equity dealing out the remaining turn/river of BOTH boards vs one uniform random k=2 holding: per-board win probability by the river and a signed delta vs the current-rank ahead-share; the fuller bundle also emits the joint scoop/win-one/lose-both distribution and expected pot share.

**Why:** Call-vs-fold at a price and semibluff-raise vs give-up need showdown equity, but every existing equity dim is current-rank dominance ('no runout on flop/turn'); a fragile made hand and a monster draw read identically, and pot-building is only correct when SCOOP-BY-RIVER is likely, not when hero is merely ahead now. The delta hands the net the made-vs-drawing axis directly.

**Final architecture** (verdict: pick): The equity_mc 4-float arch: [equity_river_A, delta_A, equity_river_B, delta_B]. New Rust runout sub-pass inside outcome_features_mc: R joint runouts (deal the remaining turn/river cards for BOTH boards from the shared unseen deck, disjoint across boards) x M opp k=2 combos sampled disjoint from the post-runout deck, fresh per runout; per (runout, combo) full-board evals; equity_river_b = mean(win + 0.5*tie) on board b; delta_b = equity_river_b - (ahead_b + 0.5*tie_b) from the existing exhaustive counters. Forked seed = outcome_seed ^ ARM1 constant so dims 0..20 stay byte-identical; batched cache key UNCHANGED (depends on exactly the fields outcome_seed hashes). Budgets: serial/UI R=16,M=16; batched knob (e.g. R=8,M=16) mirroring the existing 1024-vs-256 split. RIVER RULE (pinned): skip the sub-pass; equity_b := ahead_b + 0.5*tie_b exactly, delta := 0. Zeros when inactive.

**Reviewer reasoning:** Picks the lean 4-dim arch over the 8-dim dual_board bundle: the bundle's joint-outcome dims are exactly EQ-12 (ruled separately, rides this same pass) and its share metrics are linear recombinations, so bundling only obscures the budget accounting. This is the single largest genuine gap in the obs — every equity dim is current-rank dominance, so a fragile made hand and a monster draw are indistinguishable, and delta hands the net the made-vs-drawing axis directly. Cost is the real concern: ~2.5-5x the fused pass depending on budget, against a pass that is ~32% of rollout — ship behind a profiled budget knob and treat EQ-6/EQ-7/EQ-12 as free riders on this pass. The exact-at-river shortcut avoids pointless MC noise where the exhaustive answer is already computed.

**All proposed architectures (pre-review):**

- *equity_mc* (4 dims): 4 floats: [equity_river_A, delta_A, equity_river_B, delta_B], raw [0,1] (delta signed). New runout sub-pass in the fused Rust outcome_features_mc: R=16 joint both-board runouts x M=16 disjoint opp k=2 combos; forked ChaCha8 seed outcome_seed ^ ARM1 so existing dims stay byte-identical.
- *dual_board* (8 dims): 8 floats: eq_A, eq_B (river win share), P_scoop_river, P_win_exactly_one_river, P_lose_both_river, P_tie_involved_river, E_share_river, P_share_zero_river. ~1024 (opp,joint-runout) samples, hero re-eval per runout. (Bundles the per-board equity + joint outcomes.)

**Overlap with existing dims:** MERGE of equity 'river_equity_mean_delta' + dual 'runout_scoop_equity_mc' (the bundle also emits the joint distribution that EQ-12 carries standalone, and shares its sub-pass with EQ-6/EQ-7). Per-board ahead/tie/behind (991-996) is the current-rank version; delta = new mean - existing share.  
**Cost:** Rust engine work, the dominant new cost: ~8k evals at R=16/M=16 (~4-5x the current outcome pass; ~2x at R8/M8). Deterministic under forked seed; batched MC cache key unchanged.

### EQ-5 · Current-rank field dominance — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Current-rank dominance vs the ACTUAL live field size m (non-folded opponents, m disjoint random k=2 holdings, hero must beat ALL): P(best on A), P(best on B), P(best both / scoops field), P(best exactly one), expected pot share with n-way tie splits, P(zero share).

**Why:** Multiway value-bet thinness, c-bet bluff frequency, continue-vs-fold in 4-6-way pots — a bomb pot puts EVERY seat on the flop, yet all 20 existing outcome dims are 1-vs-1, and P(best of 5) vs P(best of 1) differ enormously (0.8^5≈0.33) with card-removal coupling; the net must synthesize this deep in the torso from pairwise dominance x active mask.

**Final architecture** (verdict: pick): The equity_mc 6-float arch, tightened: [P_best_A, P_best_B, P_scoop_field, P_win_exactly_one_field, E_share_field, P_zero_share]. ~256 seeded draws of m disjoint k=2 combos, m = live non-hero non-folded seats INCLUDING all-in seats (they contest showdown); m=1 must reproduce the 1v1 marginals (sanity anchor). Each opp's per-board rank = one lookup in the existing tab_a/tab_b pair tables (no new evals; 2m <= 10 <= unseen always). Flags use STRICT > over the field max per board; E_share uses per-board tie-splitting (hero gets 0.5/(1+t) of a board when tied with t opps at the max); P_zero_share = strictly beaten on both boards. Seed = outcome_seed ^ ARM2 ^ m, and the BATCHED MC CACHE KEY IS EXTENDED WITH m (only the count matters — holdings are exchangeable draws from the unseen deck, so hashing the full fold mask would cause spurious cache misses). Zeros when inactive. Fused-slab append; encoders pass-through.

**Reviewer reasoning:** Beats the dual_board 5-float variant because best_exactly_one and P_zero_share are the decision-facing cells (barrel-target selection and continue-vs-fold multiway) while P_lose_both is recoverable as a residual. Cheapest high-value engine add in the chunk — verified against engine.rs that the pair tables make this pure lookups, comparable to the existing k=4 arm. The cache-key caveat is real and load-bearing: outcome_seed (engine.rs:1300) hashes only hero seat/street/hole/boards, so without the m extension two states differing only in live count would silently share stale values; my tightening reduces the extension to m rather than the mask, which is exactly the feature's true dependence. Bomb pots are structurally multiway and all 20 existing outcome dims are 1v1 — this is the priority multiway fix.

**All proposed architectures (pre-review):**

- *equity_mc* (6 dims): 6 floats: [best_A, best_B, best_both, best_exactly_one, E_share_field, P_zero_share]. ~256 draws of m disjoint k=2 combos, each opponent's per-board rank = ONE lookup in the k=2 pair tables (no new evals); forked seed outcome_seed ^ ARM2 ^ hash(fold mask).
- *dual_board* (5 dims): 5 floats: [P_best_A, P_best_B, P_scoop_field, P_lose_both_vs_field, E_share_multiway]; at n_live==2 reproduces the k=2 marginals. ~1024 draws via the existing pair tables (lookups, not evals).

**Overlap with existing dims:** MERGE of equity 'field_best_now' + dual 'multiway_outcome_mc' (both current-rank vs the live field). Active mask (160-168) + 1v1 per-board/joint dims are the ingredients (nonlinear composition); EQ-8 is the runout analog.  
**Cost:** Cheapest high-value engine add (~256 x m x 2 table lookups ≈ the k=4 arm). CAVEAT: depends on the live-fold mask, so the batched MC cache key (outcome_seed) MUST be extended with the fold mask.

### EQ-6 · river_equity_volatility — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, dispersion of the per-runout river equity distribution: std across runouts, plus fraction of runouts landing near-locked (equity>=0.9) and near-dead (equity<=0.1).

**Why:** Raise-now-vs-realize-later and protection sizing: two hands with equal mean equity play oppositely when one is locked and the other volatile (charge draws/deny equity vs pot-control). NOTHING in the current obs carries second-moment equity information; outs counts are a one-street, hero-side-only proxy.

**Final architecture** (verdict: agree): 6 floats: per board [std_runout_equity, P(eq >= 0.9), P(eq <= 0.1)], A then B. Per-runout equity = (wins + 0.5*ties)/M over that runout's own M combos from EQ-4's sub-pass; std = population std over the R runouts; threshold dims = counter fractions over R (granularity 1/R, acceptable at R=16). Zero incremental evals — counter accumulation on EQ-4's fixed deterministic sample scheme, same forked stream. RIVER RULE (pinned, matching EQ-4's): std := 0 and the threshold flags become exact thresholds of the exhaustive current-rank equity (ahead + 0.5*tie) — consistent and still informative. All-zero when inactive. Ships ONLY with EQ-4.

**Reviewer reasoning:** Agree: second-moment equity information exists nowhere in the obs, and the locked-vs-volatile distinction at equal mean equity is exactly the charge-draws vs pot-control axis; the horizon separation from EQ-11 (showdown vs one-card) is correctly argued. The arch is well-posed as free counters on EQ-4's pass — the only tightening needed is pinning the per-runout equity estimator to that runout's own M combos and the degenerate river behavior. Worthless standalone (meaningless without the runout pass), so its fate is coupled to EQ-4's — which I rule keep.

**Overlap with existing dims:** None existing carries equity dispersion; distinct HORIZON from EQ-11 (showdown vs one-card); straight/flush outs correlate weakly (hero-side counts).  
**Cost:** Zero incremental engine evals given EQ-4's pass (counter accumulation only); meaningless without it. Same forked-seed determinism.

### EQ-7 · river_class_and_nut_arrival — 8 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, probability that hero's RIVER hand reaches strength classes: P(>=straight), P(>=flush), P(>=full house), plus P(hero's river hand beats all M sampled combos on that runout). Probability-space hand-arrival, including 2-card backdoors.

**Why:** Semibluff selection and deep-SPR implied-odds calls: the outs/nut-distance blocks are single-street, count-space, flush/straight-only — backdoors (0 outs now, real probability by river), boat/quads arrival, and 'when I hit, do I win' are all missing. The deep tier got unclipped log1p SPR but has no nut-potential signal to pair with it.

**Final architecture** (verdict: agree): 8 floats: per board [P(river cat >= straight), P(>= flush), P(>= full house), P(hero beats all M sampled combos on that runout)], A then B. Hero's river category = the category byte of the cached hero river rank (rank >> 20, the existing category extraction), computed once per runout in EQ-4's pass; the beats-all dim is a per-runout indicator over the same M combos. Zero incremental evals; same forked stream; slab pass-through in all three encoders. RIVER RULE (pinned): class dims = exact indicators of hero's CURRENT category; beats-all := 1 iff behind_frac == 0 on that board (the exhaustive limit of the sampled definition). All-zero when inactive. Ships ONLY with EQ-4.

**Reviewer reasoning:** Agree: probability-space hand arrival including 2-card backdoors and the boat/quads direction is genuinely uncovered (the outs blocks are one-card, count-space, straight/flush/SF-only — verified against the 834-910 layout), and 'when I arrive, am I best' pairs the deep tier's unclipped SPR with an actual nut-potential signal. Free rider on EQ-4's cached hero river evals. The tightenings are the pinned river degeneration and pinning the category threshold set {straight, flush, full house} to the existing 9-class enum indices so all encoders agree.

**Overlap with existing dims:** straight/flush/SF outs (834-910) and blockers-to-nuts (999-1007) cover current-street flush/straight arrival in count-space; nothing covers boats, backdoors, or when-I-hit-am-I-good.  
**Cost:** Zero incremental engine evals on top of EQ-4's runout sub-pass (category extraction + counters). Rust-only; slab passthrough.

### EQ-8 · field_share_at_river — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Check-down equity vs the live field: MC expectation of hero's pot share at showdown against m disjoint random k=2 combos over full runouts, plus P(scoop the field at river), P(zero share), and share std — the actual payoff function marginalized over cards.

**Why:** Every close continue/stack-off decision multiway: it composes runout equity, field size, and dual-board half-pot splits — three things the net must otherwise multiply through approximations. Existing dims give current-rank 1v1 pieces only; nothing showdown-anchored exists.

**Final architecture** (verdict: agree): 4 floats [E_share_river_field, P_scoop_field_river, P_zero_share_river, std_share_river]. Extends EQ-4's sub-pass: per runout, sample F fields of m disjoint k=2 combos from the post-runout deck (m = live non-hero seats incl. all-in, same definition as EQ-5); opp evals on the runout boards are REAL evals (pair tables cover only the current board); per-sample share = per-board n-way tie-split (0.5/(1+t)) summed over boards, in [0,1]; moments over the R*F samples. DEFAULT BUDGET TIGHTENED to R=8 (reusing EQ-4's runouts), F=2 (~1.6k pair-set evals at m=5, about 1x the current pass) with (R,F) as knobs. Seed fork = outcome_seed ^ ARM4 ^ m; the batched cache key needs the SAME m extension as EQ-5. Zeros when inactive; at river, tallied over F fields on the degenerate empty runout (exact board, sampled fields). Ships only with EQ-4.

**Reviewer reasoning:** Agree with a halved default budget: this is the actual payoff function (runout equity x field size x dual-board split composed), and the proposal is honest that it is the composition of EQ-4 and EQ-5 — the question is whether the net multiplies those pieces adequately. Given bomb pots are structurally multiway and every close stack-off decision is this quantity, I keep it, but it is explicitly the FIRST candidate to cut if the rollout profile objects, since its marginal information over shipped EQ-4 + EQ-5 is the interaction term only. The cache-key m extension is mandatory, same reasoning as EQ-5.

**Overlap with existing dims:** Pure composition target of EQ-4 + EQ-5 plus the dual-board split rule; runout analog of EQ-5; no existing dim approaches it.  
**Cost:** Rust engine work: opponent evals on runout boards are real evals (pair table only covers the current board): R=16 x F=4 x m x ~20 ≈ +6k evals (~3-4x the current pass on top of EQ-4). Budget knob (R,F); cache key must include fold mask.

### EQ-9 · top_quantile_dominance — 4 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, hero ahead-fraction restricted to the STRONGEST current k=2 combos: vs the top 25% and top 10% of the exhaustively enumerated pair universe, ranked by their own current hand rank on that board. Objective strength-conditioning, no range modeling.

**Why:** Value-raise vs bluff-raise classification and thin-value sizing: money goes in against continuation regions, which are strength-censored — random-combo dominance systematically overstates equity vs the hands that put in raises. The k=3/4 arms shift density toward strong hands but only coarsely and only in the joint block.

**Final architecture** (verdict: agree): 4 floats [ahead_vs_top25_A, ahead_vs_top10_A, ahead_vs_top25_B, ahead_vs_top10_B]. During the k=2 exhaustive loop, additionally collect each combo's per-board rank into a Vec (or re-iterate the existing tab_a/tab_b tables over the C(n,2) index pairs); post-pass per board: sort descending (deterministic; rank-ties straddling the quantile boundary are harmless — tied combos have identical hero-ahead indicators, so the count is invariant to which are included), qualifier count q = max(1, floor(quantile * n)), feature = |{qualifiers with rank < hero_rank}| / q (STRICT ahead, matching the pb 'ahead' convention in the fused pass). Exhaustive, exact, seedless; appended to the fused slab, dims 0..20 byte-identical, cache key unchanged. Zeros when inactive. Encoders pass-through.

**Reviewer reasoning:** Agree: strength-censored dominance is the right objective proxy for equity-vs-continuing-range without any range modeling, and the k=3/4 arms genuinely don't decompose per board. Cost claim verified against the Rust loop — the ranks are already computed and stored per pair (engine.rs:1450-1451), so this is one O(n log n) select + sweep over ~820 values per board, near-free. My only tightenings: the max(1, floor(.)) floor for degenerate small universes (study-mode duplicates), the strict-ahead pin, and the observation that quantile-boundary tie-breaking provably cannot affect the output (equal ranks contribute equally), which kills the one parity worry.

**Overlap with existing dims:** k=3/4 joint outcome rows (978-990) are the density-shifted cousins — coarser, joint-only, no per-board decomposition.  
**Cost:** Near-free: no new evals, one O(n log n) select + counter sweep over 820 stored ranks per board inside the existing fused pass.

### EQ-10 · call_equity_margin — 3 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Signed break-even margins composing equity with the price: (share_now_1v1 - eff_price), (share_river_1v1 - eff_price), (share_river_field - eff_price), where eff_price is the effective-stack-capped to_call/(pot+to_call). Sign = call is +EV when action closes.

**Why:** The fold/call boundary directly — it IS sign(margin) for pot-closing calls; the operands all exist (or will) but the net must learn the division-and-compare composition, and a signed margin makes the decision boundary linear.

**Final architecture** (verdict: agree): 3 signed floats, pure encoder arithmetic in all three encoders (identical one-liners): [margin_now_1v1, margin_river_1v1, margin_river_field], each = share_estimate - eff_price where eff_price = obs[1007] (eff_to_call/(pot+eff_to_call)). share_now_1v1 = 0.5*aheadA + 0.25*tieA + 0.5*aheadB + 0.25*tieB from dims 991-996; share_river_1v1 = 0.5*(equity_river_A + equity_river_B) from EQ-4's slab; share_river_field = E_share_river_field from EQ-8's slab. ALL THREE dims forced to 0 when eff_to_call == 0 (the sign semantics exist only facing a bet; unconditional emission would just duplicate the share dims). Range ~[-1, 1], no clip. Dims 2-3 ship only if EQ-4 / EQ-8 ship (size the block to what exists); margin 1 is standalone and free today.

**Reviewer reasoning:** Agree: the sign of the margin IS the fold/call boundary for pot-closing calls, the operands exist or are being added, and a divide-and-compare composition is exactly what an MLP represents poorly — making the decision boundary linear in one feature is the cheapest kind of win. The list's rule-3 separation from DUAL-1 (price re-denomination vs equity-minus-price gap) is correct and preserved. My tightenings: pin the share_now formula (the proposal left it implicit), the zero-when-no-bet gate, and consistent eff-capping on the price side so the margin is exact for the all-in-closing case.

**Overlap with existing dims:** KEPT SEPARATE from DUAL-1 (price re-denomination) — this is the equity-minus-price GAP, a different quantity (rule 3). Pot odds (780), eff-price (1007-1012), outcome fractions carry the operands, never the margin.  
**Cost:** Zero engine cost (arithmetic on existing + new slab dims); margins 2-3 exist only if EQ-4/EQ-8 ship (margin 1 is standalone and free today).

### EQ-11 · next_card_equity_swing — 8 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, one-street lookahead distribution: over candidate NEXT cards on that board, the [mean, std, min, max] of hero's next-street ahead-fraction vs random k=2 combos. Min = worst card (counterfeit/board-pair severity), max = best, std = immediate re-decision risk.

**Why:** Flop protection sizing and turn-barrel planning: distinct from showdown-horizon volatility because the NEXT card re-prices the pot and reopens action — 'how bad is the worst turn card for my two-pair' is a one-card question. Existing draw flags/outs are hero-side counts; board dims are static; nothing measures opponent-side distributional next-card impact.

**Final architecture** (verdict: agree): 8 floats: per board [mean, std, min, max] of next-street ahead-share (ahead + 0.5*tie vs random k=2 combos) over candidate next cards, A then B; zeros at river and when inactive. Rust MC sub-pass, forked seed = outcome_seed ^ ARM3 (cache key unchanged — hero/boards/street only): FLOP: C=8 sampled candidate cards per board x M=64 combos per card, with min/max DOCUMENTED as over the sampled candidates (a soft quantile, not the true worst card — min of noisy per-card estimates also biases low; do not present it as exact counterfeit severity). TURN: exhaustive over unseen cards (~44) with M capped at 32. Per candidate: hero re-eval on board+c plus M current-rank combo evals (pair tables inapplicable — the board changed). Budget (C, M) is an independent knob.

**Reviewer reasoning:** The arch as proposed is essentially right (it already specs sampled-flop/exhaustive-turn and a budget knob), so agree with the soft-min documentation and budget pins — but I recommend removal on cost grounds. Verified arithmetic: ~3-6x the current fused pass depending on street/budget, on top of the EQ-4 family, against a pass already ~32% of rollout wall-clock.

**⚠ Remove recommended:** Cost grossly disproportionate for a second-order signal: EQ-6 (free on EQ-4) carries equity dispersion at the showdown horizon, BRD-4/BRD-5 carry the card-type next-card threat census near-free, and the headline min-over-cards degrades to a soft quantile under C=8 sampling anyway (the exact worst-card read it promises isn't what ships). Revisit only with profiled headroom after the EQ-4 family and EQ-5 land.

**Overlap with existing dims:** EQ-6 is the sibling — different HORIZON (showdown vs one-card), both formats intended; draw flags (799-803) and outs blocks cover hero improvement counts only.  
**Cost:** Rust engine work: ~5k evals at flop budgets (~3x the current outcome pass; ~7x on turn if cards exhausted — cap combos). Second-largest new cost after the runout family; independent budget knob.

### EQ-12 · river_joint_outcome_dist — 6 dims

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** The runout-anchored analog of the current-rank joint block, vs one random k=2 opponent held fixed across both boards: [P(scoop), P(quarter opp: win one + tie one), P(win exactly one), P(chop both), P(lose both)] at the river.

**Why:** Pot-build vs pot-control on half-pot hands: dual-board terminal payoff is set by the JOINT river outcome, and 'I nearly always win exactly one board' dictates small-ball while river scoop mass licenses pot-building. The existing joint block and win-one/tie-both dims are current-rank only; runout chop arrival is invisible.

**Final architecture** (verdict: better): 6 floats, free counters over EQ-4's (runout, combo) samples using the SAME tally rules as the current-rank pass (tally_joint + the win-one/tie-both counters): [scoop_opp_river, quarter_opp_river, scoop_hero_river, quarter_hero_river, win_exactly_one_river, tie_both_river] — the exact runout-anchored mirror of dims 978-981 + 997-998. Zero incremental evals; same forked stream; slab append; encoders pass-through. RIVER RULE (pinned): equals the current-rank cells exactly (copy, not re-sample). All-zero when inactive. Ships ONLY with EQ-4.

**Reviewer reasoning:** The concept (runout-anchored joint outcome = the terminal payoff distribution) is right, but the proposed 5-cell set is asymmetric and invents a new convention: it carries hero-quarters-opp but drops opp-quarters-hero into the residual, and splits chop-both oddly. Mirroring the existing current-rank format cell-for-cell is strictly better: the net can read per-cell now-vs-river deltas (the made/draw axis per outcome class), no new vocabulary, and the layout self-documents against 978-981/997-998. Same cost (free counters), +1 dim.

**Overlap with existing dims:** Current-rank joint fractions k=2 row (978-982) and win-one/tie-both (997-998) are the now-versions; EQ-4's dual_board arch also emits this joint distribution (bundled).  
**Cost:** Zero incremental engine evals given EQ-4's pass (joint tallying only); meaningless without it.

### EQ-13 · clean_outs_equity — 4 dims  ⚠ REMOVE-REC

**Decision:** ☐ keep ☐ modify ☐ drop

**What:** Per board, out quality in probability space: fraction of next cards that IMPROVE hero's hand-rank class, and the mean next-street ahead-fraction CONDITIONAL on improving — 'when I hit, am I actually good', draw-type-agnostic.

**Why:** Semibluff hand selection multiway: non-nut draws (dominated flush draws, sucker-end wraps) are the classic bomb-pot disaster, and today only nut_flush_draw_outs covers cleanliness (flushes only, count-space, no conditional equity). High improve-fraction with low conditional equity is precisely the 'stop semibluffing' signal.

**Final architecture** (verdict: agree): 4 floats: per board [P(next card improves hero's hand-rank CLASS), E(ahead-share next street | improved)], A then B; zeros at river and when inactive. Free conditional tallying over EQ-11's per-(card, combo) samples, same forked stream: 'improves' = hero category on board+c strictly exceeds hero's current category (category byte of the rank, the existing extraction); the conditional mean is over improving samples only, PINNED to 0 when no sampled candidate improves (the 0/0 case — must be explicit for parity). Ships ONLY with EQ-11.

**Reviewer reasoning:** The arch is well-posed as free counters and the dirty-draw signal (high improve-fraction, low conditional equity) is real — but it is structurally chained to EQ-11, which I recommend removing for cost, so this inherits the recommendation.

**⚠ Remove recommended:** Rides EQ-11 (recommended for removal — 3-6x the fused MC pass); cannot ship alone. Its decision content is substantially covered at zero-or-cheap cost: EQ-7's P(>= class) + P(beats-all) give arrival-and-am-I-good at the showdown horizon (including backdoors, which matter more in a flop-start game), nut_flush_draw_outs (864/902) covers flush cleanliness, and BRD-6 grades straight-out quality. Re-propose only together with EQ-11 under profiled headroom.

**Overlap with existing dims:** nut_flush_draw_outs (864-868/902-906) covers flush cleanliness in count-space; straight outs carry no quality signal at all.  
**Cost:** Zero incremental engine evals given EQ-11's sub-pass (conditional tallying); ships only with it.


## Merge notes (dedup log, drops, per-category tallies)

PER-CATEGORY COUNTS (ideas after merge): position 7, history 14, stack_geometry 11, board_texture 13, dual_board 5, equity_mc 13. TOTAL 63 merged ideas (from 73 input ideas across 6 lenses; 9 merge groups collapsed 10 entries).

TOTAL PROPOSED DIMS (sum of each idea's FIRST-arch dims): 746. By category: position 42, history 453 (dominated by U4=80, HIST-13=128, HIST-14=64), stack_geometry 49, board_texture 92, dual_board 45, equity_mc 65.

MERGES PERFORMED (9). Each keeps every distinct architecture in the archs array:
1. STK-1 = position 'Money still to act' (4, raw eff-stack) + stack 'behind_raise_exposure' (2, stack-minus-owed raise capacity) — same pending set, near-identical max/sum aggregate.
2. STK-8 = stack 'side_pot_structure' (3, current-commit eligibility) + dual 'eligible_pot_sidepot_block' (3, full-reach eligibility) — two definitions of hero's winnable pot under all-in layering.
3. BRD-4 = board 'board_threat_census' (6) + dual 'board_arrival_volatility' (6) — same forward-looking next-card texture census; dual arch adds 3->4 flush-over-flush.
4. BRD-11 = board 'turn_river_identity' (20, scalar rank + suit one-hot) + dual 'board_card_arrival_order' (68, full rank one-hot + suit one-hot) — same deal-order card identity.
5. DUAL-1 = dual 'split_adjusted_price_ladder' (4) + stack 'halfpot_price_dual_board' (1) — split-pot call price; the ladder contains the exact half-pot dim.
6. DUAL-2 = board 'best_hand_usage_mask' (10, raw per-board masks) + dual 'best_combo_board_coverage' (4, overlap digest) — which hole cards make each board's best hand.
7. DUAL-3 = dual 'per_board_nut_lock_flags' (6) + equity 'nut_lock_freeroll_flags' (5) + board 'nuts_now_flags' (4) — all exact ==0/near-zero thresholds of the k=2 behind/tie counters.
8. EQ-4 = equity 'river_equity_mean_delta' (4) + dual 'runout_scoop_equity_mc' (8) — one both-boards-to-river MC vs a random opponent; the dual bundle also emits the joint distribution.
9. EQ-5 = equity 'field_best_now' (6) + dual 'multiway_outcome_mc' (5) — current-rank dominance vs the live m-opponent field.

CROSS-CATEGORY MOVE: nuts_now_flags was tagged equity_mc by its proposer but is the same exact-threshold-of-k2-counters mechanism as the two dual_board nut-lock ideas, and the freeroll/scoop-lock framing is inherently dual-board — refiled into DUAL-3. This shifts one idea out of equity_mc into dual_board relative to the raw category tags.

NO DROPS. Every idea is objective/deterministic (counts, fractions, and seeded-ChaCha8 MC that is a pure function of state — the MC ideas do NOT read hidden opponent cards, they enumerate/sample the unseen deck). No idea's own overlap note declares it a strict duplicate of an EXISTING encoded dim; the self-described 'redundant formats' (DUAL-1 half-pot = monotone transform of pot odds; DUAL-3/nuts_now = exact threshold of per_board_outcome; HIST-10 decayed vs U4; POS-5 field vs POS-4 mask) are explicitly WELCOMED redundant formats, not drops.

KEPT SEPARATE despite thematic closeness (rule 3), with rationale:
- POS-4 (acts-after-hero binary mask) vs POS-5 (acting-order scalar field): the field carries opponent-vs-opponent adjacency + closing-option argmax the binary hero-relative mask cannot express — different quantities though the field is a redundant-format superset.
- BRD-6 (straight_out_union) vs BRD-7 (boat_plus_outs) vs BRD-12 (hero_improve_outs_dedup): typed straight/FH out counts vs the deduped ALL-type aggregate — BRD-12 is the aggregate of which BRD-6/BRD-7 are typed components.
- EQ-3 (joint_board_outcome_matrix, current-rank 3x3) vs EQ-12 (river_joint_outcome_dist, runout joint): now-vs-river versions of the same five/nine events; EQ-4's dual arch also emits the runout joint (noted overlap).
- EQ-5 (current-rank field) vs EQ-8 (field_share_at_river, runout field): horizon differs (current rank vs full showdown).
- EQ-6 (river_equity_volatility, showdown dispersion) vs EQ-11 (next_card_equity_swing, one-card dispersion): sibling formats, different lookahead horizon.
- DUAL-1 (price re-denominated to split pot) vs EQ-10 (call_equity_margin, equity-minus-price gap): this is exactly the rule-3 example ('pot odds in bb' vs 'equity vs price gap' are different quantities).

STRICT-SUBSET OVERLAPS kept as cheaper/standalone alternatives (noted per-idea, not merged into the mega-blocks):
- HIST-2 (Prior-street aggressor trail, 16) is contained in U4's per-street final-aggressor flags but is a far cheaper standalone add.
- HIST-13 (Seat-street last-gate matrix, 128) is the maximalist alternative that subsumes HIST-6, HIST-12's fold-street one-hots, and U4's aggressor flags — explicitly proposed as 'this OR the curated digests'; left for the panel to choose granularity.
- STK-4 (seat_commitment_ratio, 8) equals U4's per-seat commitment-fraction column but U4 is an 80-dim history mega-block and STK-4 is a focused stack_geometry feature; not a strict dup, so kept.

SHARED SUB-DIMS across separate ideas (flagged in overlap fields so the panel can dedupe at build time if desired, but the parent blocks differ): dead-money fraction appears in HIST-12 and STK-8; the ante pot in HIST-9 and STK-10; caller-count-of-current-bet in POS-7 and HIST-11; raise-counts (street/hand) in HIST-3 (log1p) and HIST-9 (raw); hero-was-raised/facing-checkraise in HIST-3 and HIST-11.

ENGINE-CHANGE FOOTPRINT (for the panel's cost budgeting): the biggest new compute is the equity_mc runout family — EQ-4 is the dominant standalone cost (~2-5x the current outcome MC pass depending on R/M), and EQ-6/EQ-7/EQ-12 ride EQ-4's sub-pass for free; EQ-8 and EQ-11 each add a comparable second pass; EQ-2 (nut_retention) must be profiled before adoption. Cheap-but-high-value engine adds: EQ-3, EQ-9, DUAL-3, DUAL-4 (free counters in the existing k=2 loop) and EQ-5 (pair-table lookups). Several ideas require exposing the engine's acted_this_street bit (U3, HIST-3, HIST-6, STK-1) or the argmax 2-card subset (DUAL-2) — shared one-time state-exposure changes. Note: EQ-5/EQ-8 make the batched MC cache key depend on the live-fold mask (cache-key extension flagged).
