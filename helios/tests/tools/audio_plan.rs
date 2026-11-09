/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! CPU plans from authenticated audio metadata, using the serving frontend itself.

use flare_helios::{
    frontend::{AudioSettings, audio::AudioProcessor},
    model::contract::AudioCapacity,
    package::{protection::{self, Selection}, reader::Metadata},
};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, io::{BufWriter, Write}, path::Path, sync::atomic::AtomicBool};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    id: String,
    path: std::path::PathBuf,
    samples: usize,
    pcm_f32le_sha256: String,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn emit(output: &mut impl Write, value: Value) -> Result<()> {
    serde_json::to_writer(&mut *output, &value)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

fn plan(input: &Input, processor: &AudioProcessor, capacity: &AudioCapacity,
        settings: &AudioSettings) -> Result<Value> {
    if fs::metadata(&input.path)?.len() > settings.max_file_bytes as u64 {
        return Err("encoded audio exceeds the signed frontend limit".into());
    }
    let bytes = fs::read(&input.path)?;
    let cancelled = AtomicBool::new(false);
    let pcm = processor.decode(&bytes, &cancelled)?;
    let mut hash = Sha256::new();
    for value in &pcm { hash.update(value.to_le_bytes()); }
    let decoded_hash = hex(&hash.finalize());
    if pcm.len() != input.samples || decoded_hash != input.pcm_f32le_sha256 {
        return Err("native decoded PCM differs from the frozen samples or identity".into());
    }
    let mut chunks = Vec::new();
    for range in processor.transcription_ranges(&pcm, &cancelled)? {
        // Normalize the entire service chunk once. Encoder windows are views
        // into that matrix, never independent STFT/normalization inputs.
        let (audio, identity) = processor.prepare_pcm(&pcm[range.clone()], capacity, &cancelled)?;
        let windows: Vec<_> = audio.windows().iter().map(|window| json!({
            "frames": [window.frames.start, window.frames.end],
            "features": [window.features.start, window.features.end],
        })).collect();
        chunks.push(json!({"samples": [range.start, range.end],
            "semantic_identity": hex(&identity), "features": audio.feature_count(),
            "frames": audio.windows().last().ok_or("missing audio window")?.frames.end,
            "windows": windows}));
        // Drop the Mel matrix before preparing the following service chunk.
    }
    Ok(json!({"kind": "input", "status": "passed", "id": input.id,
        "samples": pcm.len(), "pcm_f32le_sha256": decoded_hash,
        "encoded_sha256": hex(&protection::sha256(&bytes)), "chunks": chunks}))
}

pub fn run(args: &[String]) -> Result<()> {
    let metadata = Metadata::open(Path::new(&args[0]))?;
    let publishers: Vec<flare_helios::runtime_package::Publisher> =
        serde_json::from_slice(&fs::read(&args[1])?)?;
    let tp: u32 = args[2].parse()?;
    if tp == 0 { return Err("tensor-parallel size must be positive".into()); }
    let catalog = protection::verify(&metadata.envelope, &publishers, Selection::Catalog)?;
    let mut matches = Vec::new();
    for index in 0..catalog.deployment_count() {
        if catalog.deployment_body(index)?.tensor_parallel == tp { matches.push(index); }
    }
    let [deployment] = matches.as_slice() else {
        return Err("requested tensor-parallel size must select exactly one signed deployment".into());
    };
    let verified = protection::verify(&metadata.envelope, &publishers,
        Selection::Deployment { deployment: *deployment, rank: None })?;
    if verified.claim().namespace != args[5] {
        return Err("publisher namespace acceptance does not match the signed namespace".into());
    }
    let settings: AudioSettings = serde_json::from_value(serde_json::to_value(
        verified.package().frontend.as_ref().and_then(|frontend| frontend.audio.as_ref())
            .ok_or("package has no typed audio frontend")?)?)?;
    let capacity: AudioCapacity = serde_json::from_value(serde_json::to_value(
        verified.deployment().and_then(|body| body.contract.as_ref())
            .and_then(|contract| contract.model_contract.as_ref()).and_then(|model| model.audio.as_ref())
            .ok_or("deployment has no typed audio capacity")?)?)?;
    let processor = AudioProcessor::new(settings.clone(), 1)?;
    if fs::metadata(&args[3])?.len() > 32 * 1024 * 1024 {
        return Err("input plan manifest exceeds 32 MiB".into());
    }
    let inputs: Vec<Input> = serde_json::from_slice(&fs::read(&args[3])?)?;
    if inputs.is_empty() { return Err("input plan manifest is empty".into()); }
    let mut ids = BTreeSet::new();
    let mut duplicates = BTreeSet::new();
    for input in &inputs {
        if !ids.insert(&input.id) { duplicates.insert(&input.id); }
    }
    let mut output = BufWriter::new(fs::File::create_new(&args[4])?);
    emit(&mut output, json!({"kind": "header", "schema": 1,
        "scope": "authenticated metadata and actual CPU frontend plans only; no payload, CUDA, or model execution validation",
        "envelope_sha256": hex(&verified.envelope_digest()), "signer": verified.claim().signer,
        "model_identity": verified.claim().model_identity, "deployment": deployment,
        "tensor_parallel": tp, "settings": settings, "capacity": capacity, "inputs": inputs.len()}))?;
    let mut failures = 0usize;
    for (ordinal, input) in inputs.iter().enumerate() {
        let valid = !input.id.is_empty() && !duplicates.contains(&input.id) && input.samples > 0
            && input.samples <= settings.max_samples && input.pcm_f32le_sha256.len() == 64
            && input.pcm_f32le_sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        let planned = if valid { plan(input, &processor, &capacity, &settings) }
            else { Err("input plan row has invalid or duplicate identities".into()) };
        let mut value = match planned {
            Ok(value) => value,
            Err(error) => {
                failures += 1;
                json!({"kind": "input", "status": "failed", "id": input.id, "error": error.to_string()})
            }
        };
        value["ordinal"] = ordinal.into();
        emit(&mut output, value)?;
    }
    emit(&mut output, json!({"kind": "summary", "inputs": inputs.len(), "failed": failures}))?;
    if failures > 0 { return Err("one or more native frontend plans failed; all input results retained".into()); }
    Ok(())
}
