/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Native frontend probes with inspectable PCM/features, without a model or GPU.

use flare_helios::frontend::audio::{self, AudioLimits, LogMelPlan};
use std::{fs, io::Write, sync::atomic::AtomicBool};

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 || !matches!(args[1].as_str(), "decode" | "features") {
        return Err("usage: test_audio {decode|features} INPUT OUTPUT_F32LE".into());
    }
    let cancelled = AtomicBool::new(false);
    let size = fs::metadata(&args[2])?.len();
    let limit = if args[1] == "decode" {
        25 * 1024 * 1024
    } else {
        600 * 16000 * 4
    };
    if size > limit {
        return Err("input exceeds probe capacity".into());
    }
    let bytes = fs::read(&args[2])?;
    let values = if args[1] == "decode" {
        audio::decode(&bytes, AudioLimits::default(), &cancelled)?
    } else {
        if bytes.len() % 4 != 0 {
            return Err("PCM input must contain complete f32 samples".into());
        }
        let samples: Vec<_> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();
        LogMelPlan::new()?.features(&samples, &cancelled)?
    };
    let mut output = fs::File::create_new(&args[3])?;
    for value in &values {
        output.write_all(&value.to_le_bytes())?;
    }
    println!(
        "{{\"operation\":\"{}\",\"elements\":{}}}",
        args[1],
        values.len()
    );
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
