//! Criterion benchmarks of the engine's hot kernels (TEST-026).
//!
//!   bash scripts/rust_bench.sh                    every benchmark (a few minutes)
//!   bash scripts/rust_bench.sh encode             only names matching a regex
//!   bash scripts/rust_bench.sh eval -- --save-baseline before
//!   bash scripts/rust_bench.sh eval -- --baseline before     after a change
//!
//! All on ONE rayon thread (per-core cost). Groups:
//! - `eval`: the 5-card lookup, PLO (2 hole + 3 board of 5-card holes on a
//!   river), NLH best-5-of-7;
//! - `features`: the per-row hand features (hero_board_v3, board_draw_v3,
//!   hero categories) on a flop/turn/river mix;
//! - `mc`: the opp-outcome Monte-Carlo block (384 samples, the training
//!   budget) without and with the shared per-street pair table, and the table;
//! - `rules`: a deal, the legal mask, a whole random hand;
//! - `payouts`: EV settlement of flop all-ins (64 runouts) and a showdown;
//! - `encode`: one rollout step's observation work for 256 tables — the full
//!   and minimal packers and encoders with cold caches, and compact packing.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use plo5bp_engine::bindings::bench_api::{obs_dim, pack_rows, Table};
use plo5bp_engine::cards::Card;
use plo5bp_engine::hand_eval::{evaluate_5, evaluate_nlh, evaluate_plo};
use plo5bp_engine::state::{GameState, Street};
use plo5bp_engine_bench::*;

const MC: usize = 384;

fn eval(c: &mut Criterion) {
    one_thread();
    let fives: Vec<[Card; 5]> = (0..1024u64)
        .map(|s| cards(s, 5).try_into().unwrap())
        .collect();
    let plo: Vec<(Vec<Card>, [Card; 5])> = (0..256u64)
        .map(|s| {
            let v = cards(s, 10);
            (v[..5].to_vec(), v[5..].try_into().unwrap())
        })
        .collect();
    let nlh: Vec<(Vec<Card>, Vec<Card>)> = (0..1024u64)
        .map(|s| {
            let v = cards(s, 7);
            (v[..2].to_vec(), v[2..].to_vec())
        })
        .collect();
    let mut g = c.benchmark_group("eval");
    g.throughput(Throughput::Elements(fives.len() as u64));
    g.bench_function("evaluate_5", |b| {
        b.iter(|| fives.iter().fold(0u32, |a, h| a ^ evaluate_5(black_box(h))))
    });
    g.throughput(Throughput::Elements(plo.len() as u64));
    g.bench_function("evaluate_plo_plo5_river", |b| {
        b.iter(|| {
            plo.iter().fold(0u32, |a, (h, bd)| {
                a ^ evaluate_plo(black_box(h), black_box(bd))
            })
        })
    });
    g.throughput(Throughput::Elements(nlh.len() as u64));
    g.bench_function("evaluate_nlh", |b| {
        b.iter(|| {
            nlh.iter().fold(0u32, |a, (h, bd)| {
                a ^ evaluate_nlh(black_box(h), black_box(bd))
            })
        })
    });
    g.finish();
}

fn features(c: &mut Criterion) {
    one_thread();
    let states = states_in_progress(&plo5_6max(), 256, 10, 7);
    let mut g = c.benchmark_group("features");
    g.throughput(Throughput::Elements(states.len() as u64));
    g.bench_function("hero_board_v3", |b| {
        b.iter(|| {
            states
                .iter()
                .fold(0u8, |a, s| a ^ black_box(s).hero_board_v3()[0])
        })
    });
    g.bench_function("board_draw_v3", |b| {
        b.iter(|| {
            states
                .iter()
                .fold(0u8, |a, s| a ^ black_box(s).board_draw_v3()[0])
        })
    });
    g.bench_function("hero_category_both_boards", |b| {
        b.iter(|| {
            states.iter().fold(0u8, |a, s| {
                let seat = s.current_actor().unwrap_or(0);
                a ^ s.hero_category(seat, 0) ^ s.hero_category(seat, 1)
            })
        })
    });
    g.finish();
}

fn mc(c: &mut Criterion) {
    one_thread();
    let config = plo5_6max();
    let mut g = c.benchmark_group("mc");
    for (name, street) in [("flop", Street::Flop), ("turn", Street::Turn)] {
        let states: Vec<GameState> = states_on(&config, 16, street, 11);
        let tables: Vec<_> = states.iter().map(|s| s.board_pair_table()).collect();
        g.throughput(Throughput::Elements(states.len() as u64));
        g.bench_function(format!("outcome_features_mc_{name}"), |b| {
            b.iter(|| {
                states
                    .iter()
                    .map(|s| black_box(s).outcome_features_mc(MC)[0])
                    .sum::<f32>()
            })
        });
        g.bench_function(format!("outcome_features_mc_shared_{name}"), |b| {
            b.iter(|| {
                states
                    .iter()
                    .zip(&tables)
                    .map(|(s, t)| black_box(s).outcome_features_mc_shared(MC, t.as_ref())[0])
                    .sum::<f32>()
            })
        });
        g.bench_function(format!("board_pair_table_{name}"), |b| {
            b.iter(|| {
                states
                    .iter()
                    .filter(|s| black_box(s).board_pair_table().is_some())
                    .count()
            })
        });
    }
    g.finish();
}

fn rules(c: &mut Criterion) {
    one_thread();
    let config = plo5_6max();
    let states = states_in_progress(&config, 256, 10, 3);
    let mut g = c.benchmark_group("rules");
    g.throughput(Throughput::Elements(256));
    g.bench_function("new_hand_6max", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            (0..256).fold(0usize, |a, i| {
                seed += 1;
                a ^ GameState::new_hand(config.clone(), black_box(seed), i % 6).pot as usize
            })
        })
    });
    g.bench_function("redeal_6max", |b| {
        let mut tables: Vec<GameState> = (0..256u64)
            .map(|i| GameState::new_hand(config.clone(), i, (i % 6) as usize))
            .collect();
        let mut seed = 1_000u64;
        b.iter(|| {
            tables.iter_mut().enumerate().fold(0usize, |a, (i, t)| {
                seed += 1;
                t.redeal(&config, black_box(seed), i % 6);
                a ^ t.pot as usize
            })
        })
    });
    g.bench_function("legal_action_mask", |b| {
        b.iter(|| {
            states
                .iter()
                .filter(|s| black_box(s).legal_action_mask()[1])
                .count()
        })
    });
    g.throughput(Throughput::Elements(64));
    g.bench_function("random_hand_to_the_end", |b| {
        let mut seed = 0u64;
        b.iter(|| {
            (0..64).fold(0usize, |a, i| {
                seed += 1;
                let mut rng = Rng::new(seed);
                let mut s = GameState::new_hand(config.clone(), seed, i % 6);
                while !s.is_terminal() {
                    random_action(&mut s, &mut rng);
                }
                a ^ s.pot as usize
            })
        })
    });
    g.finish();
}

fn payouts(c: &mut Criterion) {
    one_thread();
    let all_ins = flop_all_ins(16, 5);
    let showdowns: Vec<GameState> = states_on(&plo5_6max(), 64, Street::River, 9)
        .into_iter()
        .map(|mut s| {
            while !s.is_terminal() {
                s.apply(plo5bp_engine::actions::Action::CheckCall);
            }
            s
        })
        .collect();
    let mut g = c.benchmark_group("payouts");
    g.throughput(Throughput::Elements(all_ins.len() as u64));
    g.bench_function("payouts_ev_flop_all_in_64", |b| {
        b.iter(|| {
            all_ins
                .iter()
                .enumerate()
                .map(|(i, s)| black_box(s).payouts_ev(64, i as u64)[0])
                .sum::<i64>()
        })
    });
    g.throughput(Throughput::Elements(showdowns.len() as u64));
    g.bench_function("payouts_showdown_6max", |b| {
        b.iter(|| {
            showdowns
                .iter()
                .map(|s| black_box(s).payouts()[0])
                .sum::<i64>()
        })
    });
    g.finish();
}

fn encode(c: &mut Criterion) {
    one_thread();
    let table = Table::new(plo5_6max(), 256, 10, MC, 21);
    let rows = table.rows();
    let (full, minimal) = (obs_dim(true), obs_dim(false));
    let mut out_full = vec![0f32; rows * full];
    let mut out_min = vec![0f32; rows * minimal];
    // The compact layout: every column that holds only exact 0/1 in these rows
    // is a flag (what the encoder's flag mask marks), the rest are reals.
    table.clear_caches();
    table.encode(true, &mut out_full);
    let (flags, reals): (Vec<usize>, Vec<usize>) = (0..full).partition(|&c| {
        (0..rows).all(|r| matches!(out_full[r * full + c].to_bits(), 0 | 0x3F80_0000))
    });
    let mut bits = vec![0u8; rows * flags.len().div_ceil(8)];
    let mut real = vec![0f32; rows * reals.len()];
    table.encode(false, &mut out_min);
    let (mflags, mreals): (Vec<usize>, Vec<usize>) = (0..minimal).partition(|&c| {
        (0..rows).all(|r| matches!(out_min[r * minimal + c].to_bits(), 0 | 0x3F80_0000))
    });
    let mut mbits = vec![0u8; rows * mflags.len().div_ceil(8)];
    let mut mreal = vec![0f32; rows * mreals.len()];

    let mut g = c.benchmark_group("encode");
    g.throughput(Throughput::Elements(rows as u64));
    g.sample_size(20);
    g.bench_function("pack_full_cold", |b| {
        b.iter(|| {
            table.clear_caches();
            table.pack_full()
        })
    });
    g.bench_function("encode_full_cold", |b| {
        b.iter(|| {
            table.clear_caches();
            table.encode(true, &mut out_full);
            out_full[0]
        })
    });
    g.bench_function("encode_full_warm", |b| {
        b.iter(|| {
            table.encode(true, &mut out_full);
            out_full[0]
        })
    });
    g.bench_function("pack_minimal", |b| b.iter(|| table.pack_minimal()));
    g.bench_function("encode_minimal", |b| {
        b.iter(|| {
            table.encode(false, &mut out_min);
            out_min[0]
        })
    });
    g.bench_function("compact_pack_full_rows", |b| {
        b.iter(|| {
            pack_rows(&out_full, full, &flags, &reals, &mut bits, &mut real);
            bits[0]
        })
    });
    g.bench_function("compact_pack_minimal_rows", |b| {
        b.iter(|| {
            pack_rows(&out_min, minimal, &mflags, &mreals, &mut mbits, &mut mreal);
            mbits[0]
        })
    });
    g.finish();
}

criterion_group!(benches, eval, features, mc, rules, payouts, encode);
criterion_main!(benches);
