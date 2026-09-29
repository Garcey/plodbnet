//! Bakes a hash of the engine's sources into the module (`_engine.SOURCE_HASH`),
//! so the test suite can tell a stale build from a fresh one (tests/conftest.py,
//! TEST-036): feature probes like `hasattr(GameState, "reset_with_deck")` would
//! otherwise turn a missing rebuild into quiet skips.
//!
//! FNV-1a 64 over every file under `src/` (path-sorted, as "src/<path>"), then
//! `Cargo.toml` and `../Cargo.lock`: for each, the path, a NUL, the bytes with CRLF
//! normalised to LF, a NUL. `tests/conftest.py::engine_source_hash` is the Python
//! twin — change both together. No effect on the engine's code or outputs.
use std::fs;
use std::path::{Path, PathBuf};

fn walk(dir: &Path, base: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, base, out);
        } else if path.is_file() {
            let rel = path.strip_prefix(base).unwrap_or(&path);
            let parts: Vec<String> = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            out.push((format!("src/{}", parts.join("/")), path));
        }
    }
}

fn main() {
    let root = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let src = root.join("src");
    let mut files = Vec::new();
    walk(&src, &src, &mut files);
    files.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    files.push(("Cargo.toml".to_string(), root.join("Cargo.toml")));
    files.push((
        "../Cargo.lock".to_string(),
        root.join("..").join("Cargo.lock"),
    ));

    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (rel, path) in &files {
        let Ok(data) = fs::read(path) else { continue };
        let mut norm = Vec::with_capacity(data.len());
        let mut i = 0;
        while i < data.len() {
            if data[i] == b'\r' && i + 1 < data.len() && data[i + 1] == b'\n' {
                i += 1;
                continue;
            }
            norm.push(data[i]);
            i += 1;
        }
        feed(rel.as_bytes());
        feed(&[0]);
        feed(&norm);
        feed(&[0]);
    }
    println!("cargo:rustc-env=PLO5_SOURCE_HASH={h:016x}");
    println!("cargo:rerun-if-changed=src");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../Cargo.lock");
    println!("cargo:rerun-if-changed=build.rs");
}
