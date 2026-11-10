/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Native frontend probes with inspectable PCM/features, without a model or GPU.

use flare_helios::frontend::audio::{self, AudioLimits, AudioProcessor, LogMelPlan};
use std::{fs, io::Write, sync::atomic::AtomicBool};

#[cfg(feature = "runtime-package")]
mod audio_plan;

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    #[cfg(feature = "runtime-package")]
    if args.len() == 8 && args[1] == "plan" {
        return audio_plan::run(&args[2..]);
    }
    if args.len() == 5 && args[1] == "fetch" {
        let settings = serde_json::from_slice(&fs::read(&args[4])?)?;
        let processor = AudioProcessor::new(settings, 1)?;
        let bytes = processor.fetch_https(&args[2], std::sync::Arc::new(AtomicBool::new(false)))?;
        fs::File::create_new(&args[3])?.write_all(&bytes)?;
        println!("{}", serde_json::json!({"operation":"fetch", "file_bytes":bytes.len()}));
        return Ok(());
    }
    if args.len() == 5 && args[1] == "split" {
        let size = fs::metadata(&args[2])?.len();
        if size > 600 * 16000 * 4 || size % 4 != 0 { return Err("invalid split PCM extent".into()); }
        let bytes = fs::read(&args[2])?;
        let samples: Vec<_> = bytes.chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap())).collect();
        let settings = serde_json::from_slice(&fs::read(&args[4])?)?;
        let processor = AudioProcessor::new(settings, 1)?;
        let ranges = processor.transcription_ranges(&samples, &AtomicBool::new(false))?;
        let ranges: Vec<_> = ranges.iter().map(|range| [range.start, range.end]).collect();
        let output = fs::File::create_new(&args[3])?;
        serde_json::to_writer(output, &ranges)?;
        println!("{}", serde_json::json!({"operation":"split", "chunks":ranges.len()}));
        return Ok(());
    }
    if args.len() != 4 || !matches!(args[1].as_str(), "decode" | "features" | "features-fma") {
        return Err("usage: test_audio {decode|features|features-fma} INPUT OUTPUT_F32LE; or split INPUT_F32LE OUTPUT_JSON SETTINGS_JSON; or fetch HTTPS_URL OUTPUT_AUDIO SETTINGS_JSON; or plan PACKAGE PUBLISHERS_JSON TP INPUTS_JSON OUTPUT_JSONL ACCEPTED_NAMESPACE (requires runtime-package)".into());
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
        let numerics = if args[1] == "features-fma" {
            flare_helios::model::contract::AudioNumerics::Fp32Fma
        } else { flare_helios::model::contract::AudioNumerics::Legacy };
        LogMelPlan::with_numerics(numerics)?.features(&samples, &cancelled)?
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
        let mut source = error.source();
        while let Some(cause) = source {
            eprintln!("caused by: {cause:?}");
            source = cause.source();
        }
        std::process::exit(1);
    }
}
