/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Bounded native audio decoding and request-local FP32 Slaney Log-Mel.

use crate::{Error, Failure, FrontendError, Result};
use std::{
    ffi::{CStr, c_char, c_int, c_void},
    ptr::NonNull,
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const RATE: usize = 16_000;
const FFT: usize = 400;
const HOP: usize = 160;
const BINS: usize = FFT / 2 + 1;
const MELS: usize = 128;

#[derive(Clone, Debug)]
pub struct AudioWindow {
    pub frames: std::ops::Range<usize>,
    pub features: std::ops::Range<usize>,
}

/// Immutable request input. Encoder windows partition one already-normalized
/// Mel matrix; they do not introduce STFT or normalization boundaries.
#[derive(Clone)]
pub struct PreparedAudio {
    mel: std::sync::Arc<[f32]>,
    frames: usize,
    convolution_width: usize,
    frames_per_chunk: usize,
    features_per_chunk: usize,
    subsampling_factor: usize,
    frame_capacities: Vec<usize>,
    attention_window_frames: usize,
    windows: Vec<AudioWindow>,
}

pub struct AudioWindowBuffers {
    pub mel: Vec<f32>,
    pub frames: i32,
    pub total_input_frames: i32,
    pub convolution_width: i32,
    pub gather_indices: Vec<i32>,
    pub cumulative_lengths: [i32; 2],
    pub feature_count: i32,
    pub feature_offset: i32,
}

/// A single immutable audio input and its expanded feature-token positions.
#[derive(Clone)]
pub struct PreparedAudioInput {
    audio: std::sync::Arc<PreparedAudio>,
    token_span: std::ops::Range<usize>,
    identity: [u8; 32],
}

impl PreparedAudioInput {
    pub fn new(audio: std::sync::Arc<PreparedAudio>, token_span: std::ops::Range<usize>, identity: [u8; 32]) -> Result<Self> {
        if token_span.is_empty() || token_span.len() != audio.feature_count() {
            return Err(invalid("audio token span differs from its prepared feature count"));
        }
        Ok(Self { audio, token_span, identity })
    }

    pub fn audio(&self) -> &PreparedAudio { &self.audio }
    pub fn token_span(&self) -> &std::ops::Range<usize> { &self.token_span }
    pub fn identity(&self) -> [u8; 32] { self.identity }
}

pub fn semantic_identity(pcm: &[f32], settings: &super::AudioSettings) -> Result<[u8; 32]> {
    use sha2::{Digest, Sha256};
    settings.validate()?;
    if pcm.is_empty() || pcm.len() > settings.max_samples || pcm.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid("audio identity requires bounded finite decoded PCM"));
    }
    let mut hash = Sha256::new();
    hash.update(b"flare-conditioned-audio-centered-slaney-v1\0");
    hash.update(serde_json::to_vec(settings).map_err(|error| invalid(error.to_string()))?);
    hash.update((pcm.len() as u64).to_le_bytes());
    for sample in pcm { hash.update(sample.to_le_bytes()); }
    Ok(hash.finalize().into())
}

impl PreparedAudio {
    pub fn new(mel: Vec<f32>, frames: usize, capacity: &crate::model::contract::AudioCapacity) -> Result<Self> {
        if frames == 0 || frames > 600 * RATE / HOP
            || frames.checked_mul(MELS) != Some(mel.len()) || mel.iter().any(|value| !value.is_finite())
            || capacity.mel_bins != MELS || capacity.frames_per_chunk == 0
            || capacity.subsampling_factor == 0 || capacity.attention_window_frames == 0
            || capacity.frame_capacities.last() != Some(&capacity.attention_window_frames)
            || capacity.frame_capacities.windows(2).any(|pair| pair[0] >= pair[1])
            || capacity.frame_capacities.iter().any(|value| *value == 0 || !value.is_multiple_of(capacity.frames_per_chunk))
            || !capacity.attention_window_frames.is_multiple_of(capacity.frames_per_chunk)
            || capacity.frames_per_chunk.div_ceil(capacity.subsampling_factor) != capacity.features_per_chunk
        {
            return Err(invalid("audio features differ from the prepared frame layout"));
        }
        let mut windows = Vec::new();
        let mut offset = 0usize;
        for begin in (0..frames).step_by(capacity.attention_window_frames) {
            let end = begin.saturating_add(capacity.attention_window_frames).min(frames);
            let length = end - begin;
            let count = (length / capacity.frames_per_chunk).checked_mul(capacity.features_per_chunk)
                .and_then(|full| full.checked_add((length % capacity.frames_per_chunk).div_ceil(capacity.subsampling_factor)))
                .ok_or_else(|| invalid("audio feature count overflows"))?;
            let next = offset.checked_add(count).filter(|next| *next <= capacity.features_per_request)
                .ok_or_else(|| invalid("audio exceeds prepared feature storage"))?;
            windows.push(AudioWindow { frames: begin..end, features: offset..next });
            offset = next;
        }
        Ok(Self { mel: mel.into(), frames, convolution_width: frames.min(capacity.frames_per_chunk),
            frames_per_chunk: capacity.frames_per_chunk, features_per_chunk: capacity.features_per_chunk,
            subsampling_factor: capacity.subsampling_factor, frame_capacities: capacity.frame_capacities.clone(),
            attention_window_frames: capacity.attention_window_frames, windows })
    }

    pub fn windows(&self) -> &[AudioWindow] { &self.windows }

    pub fn feature_count(&self) -> usize { self.windows.last().unwrap().features.end }

    pub fn matches_layout(&self, capacity: &crate::model::contract::AudioCapacity) -> bool {
        capacity.mel_bins == MELS && capacity.frames_per_chunk == self.frames_per_chunk
            && capacity.features_per_chunk == self.features_per_chunk
            && capacity.subsampling_factor == self.subsampling_factor
            && capacity.frame_capacities == self.frame_capacities
            && capacity.attention_window_frames == self.attention_window_frames
            && capacity.features_per_request >= self.feature_count()
    }

    pub fn window_buffers(&self, index: usize, frame_capacity: usize) -> Result<AudioWindowBuffers> {
        let window = self.windows.get(index).ok_or_else(|| invalid("audio window index is outside its input"))?;
        if frame_capacity < window.frames.len() || !self.frame_capacities.contains(&frame_capacity) {
            return Err(invalid("audio window exceeds its graph frame capacity"));
        }
        let feature_capacity = (frame_capacity / self.frames_per_chunk).checked_mul(self.features_per_chunk)
            .filter(|value| *value <= i32::MAX as usize).ok_or_else(|| invalid("audio gather capacity overflows"))?;
        let extent = frame_capacity.checked_mul(MELS).filter(|value| *value <= MELS * (600 * RATE / HOP))
            .ok_or_else(|| invalid("audio encoder window allocation exceeds its bounded capacity"))?;
        let mut mel = vec![0.0; extent];
        for bin in 0..MELS {
            mel[bin * frame_capacity..bin * frame_capacity + window.frames.len()].copy_from_slice(
                &self.mel[bin * self.frames + window.frames.start..bin * self.frames + window.frames.end]);
        }
        let mut gather_indices = vec![-1; feature_capacity];
        let mut destination = 0;
        for (chunk, begin) in (0..window.frames.len()).step_by(self.frames_per_chunk).enumerate() {
            let length = self.frames_per_chunk.min(window.frames.len() - begin);
            for column in 0..length.div_ceil(self.subsampling_factor) {
                gather_indices[destination] = i32::try_from(chunk * self.features_per_chunk + column)
                    .map_err(|_| invalid("audio gather index overflows"))?;
                destination += 1;
            }
        }
        let integer = |value| i32::try_from(value).map_err(|_| invalid("audio metadata exceeds Int32"));
        Ok(AudioWindowBuffers { mel, frames: integer(window.frames.len())?,
            total_input_frames: integer(self.frames)?,
            convolution_width: integer(self.convolution_width)?, gather_indices,
            cumulative_lengths: [0, integer(destination)?], feature_count: integer(destination)?,
            feature_offset: integer(window.features.start)? })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

unsafe extern "C" {
    fn flare_audio_decode(
        bytes: *const u8,
        size: usize,
        output: *mut f32,
        capacity: usize,
        written: *mut usize,
        cancelled: unsafe extern "C" fn(*mut c_void) -> c_int,
        context: *mut c_void,
        message: *mut c_char,
        message_capacity: usize,
    ) -> c_int;
    fn flare_audio_fft_new() -> *mut c_void;
    fn flare_audio_fft_run(plan: *mut c_void, input: *mut f32, output: *mut Complex);
    fn flare_audio_fft_free(plan: *mut c_void);
}

fn invalid(message: impl Into<String>) -> Error {
    Error::frontend(FrontendError::Input(Failure::new(message.into())))
}

#[derive(Clone, Copy, Debug)]
pub struct AudioLimits {
    pub encoded_bytes: usize,
    pub samples: usize,
    pub decode_timeout: Duration,
}

impl Default for AudioLimits {
    fn default() -> Self {
        Self {
            encoded_bytes: 25 * 1024 * 1024,
            samples: 600 * RATE,
            decode_timeout: Duration::from_secs(60),
        }
    }
}

struct DecodeControl<'a> {
    cancelled: &'a AtomicBool,
    deadline: Instant,
}

unsafe extern "C" fn interrupted(context: *mut c_void) -> c_int {
    // The synchronous decoder borrows this stack value until its last callback.
    let control = unsafe { &*context.cast::<DecodeControl<'_>>() };
    (control.cancelled.load(Ordering::Acquire) || Instant::now() >= control.deadline) as c_int
}

pub fn decode(bytes: &[u8], limits: AudioLimits, cancelled: &AtomicBool) -> Result<Vec<f32>> {
    if bytes.is_empty() || bytes.len() > limits.encoded_bytes {
        return Err(invalid("audio is empty or exceeds the encoded-file limit"));
    }
    if limits.samples == 0
        || limits.samples > 600 * RATE
        || limits.decode_timeout.is_zero()
        || limits.encoded_bytes > 25 * 1024 * 1024
    {
        return Err(invalid("invalid audio decode capacity or deadline"));
    }
    let capacity = limits.samples + 1;
    let mut output = Vec::new();
    output
        .try_reserve_exact(capacity)
        .map_err(|e| invalid(format!("audio allocation failed: {e}")))?;
    output.resize(capacity, 0.0);
    let deadline = Instant::now()
        .checked_add(limits.decode_timeout)
        .ok_or_else(|| invalid("audio decode deadline overflows"))?;
    let mut control = DecodeControl {
        cancelled,
        deadline,
    };
    let mut count = 0;
    let mut message = [0 as c_char; 256];
    // C receives exactly this allocation's extent, including its over-limit
    // sentinel, and releases every FFmpeg owner before this call returns.
    let status = unsafe {
        flare_audio_decode(
            bytes.as_ptr(),
            bytes.len(),
            output.as_mut_ptr(),
            capacity,
            &mut count,
            interrupted,
            (&mut control as *mut DecodeControl<'_>).cast(),
            message.as_mut_ptr(),
            message.len(),
        )
    };
    if status < 0 {
        let detail = unsafe { CStr::from_ptr(message.as_ptr()) }.to_string_lossy();
        return Err(invalid(format!("audio decode failed: {detail}")));
    }
    if cancelled.load(Ordering::Acquire) || Instant::now() >= deadline {
        return Err(invalid("audio decode cancelled or timed out"));
    }
    if count == 0 || count > limits.samples {
        return Err(invalid(
            "decoded audio must be nonempty and fit the duration limit",
        ));
    }
    output.truncate(count);
    if output.iter().any(|sample| !sample.is_finite()) {
        return Err(invalid("decoded audio contains nonfinite samples"));
    }
    Ok(output)
}

/// One worker owns one transform plan and its mutable FFT scratch. Construct
/// plans before admitting work; do not share a plan between concurrent calls.
pub struct LogMelPlan {
    fft: NonNull<c_void>,
    hann: [f32; FFT],
    filters: Vec<f32>,
    input: [f32; FFT],
    spectrum: [Complex; BINS],
}

// SAFETY: the C plan owns only an AVTXContext and its function pointer; it has
// no thread-local state. Moving its exclusive owner between CPU workers is safe.
// LogMelPlan is not Sync: every transform requires &mut self and private scratch.
unsafe impl Send for LogMelPlan {}

/// Startup-created FFT plans bound concurrent feature preparation. HTTP's shared
/// frontend admission bounds decoding; this pool retains plans across requests.
pub struct AudioProcessor {
    settings: super::AudioSettings,
    plans: std::sync::Mutex<Vec<LogMelPlan>>,
    fetcher: fetch::Fetcher,
}

mod fetch;
mod cache;
pub(crate) mod encoded;

struct PlanLease<'a> {
    pool: &'a std::sync::Mutex<Vec<LogMelPlan>>,
    plan: Option<LogMelPlan>,
}

impl Drop for PlanLease<'_> {
    fn drop(&mut self) {
        if let Some(plan) = self.plan.take() {
            self.pool.lock().unwrap_or_else(|error| error.into_inner()).push(plan);
        }
    }
}

impl AudioProcessor {
    pub fn new(settings: super::AudioSettings, workers: usize) -> Result<Self> {
        settings.validate()?;
        if !(1..=64).contains(&workers) {
            return Err(invalid("audio preparation requires between 1 and 64 workers"));
        }
        let plans = (0..workers).map(|_| LogMelPlan::new()).collect::<Result<Vec<_>>>()?;
        Ok(Self { settings, plans: std::sync::Mutex::new(plans), fetcher: fetch::Fetcher::new(workers) })
    }

    pub fn fetch_https(&self, url: &str, cancelled: std::sync::Arc<AtomicBool>) -> Result<Vec<u8>> {
        self.fetcher.fetch(url, self.settings.max_file_bytes, cancelled)
    }

    pub fn decode(&self, bytes: &[u8], cancelled: &AtomicBool) -> Result<Vec<f32>> {
        decode(bytes, AudioLimits {
            encoded_bytes: self.settings.max_file_bytes,
            samples: self.settings.max_samples,
            decode_timeout: Duration::from_millis(self.settings.decode_timeout_millis),
        }, cancelled)
    }

    /// Transcription service boundaries. Encoder windows are computed later,
    /// independently inside each service chunk; chat never calls this method.
    pub fn transcription_ranges(&self, pcm: &[f32], cancelled: &AtomicBool)
        -> Result<Vec<std::ops::Range<usize>>> {
        if pcm.is_empty() || pcm.len() > self.settings.max_samples || pcm.iter().any(|value| !value.is_finite()) {
            return Err(invalid("transcription splitting requires bounded finite PCM"));
        }
        let chunk = self.settings.service_chunk_samples;
        let search = self.settings.silence_search_samples;
        let window = self.settings.silence_window_samples;
        let mut ranges = Vec::new();
        let mut begin = 0;
        while begin < pcm.len() {
            if cancelled.load(Ordering::Acquire) { return Err(invalid("audio splitting cancelled")); }
            let boundary = begin.saturating_add(chunk).min(pcm.len());
            let mut end = boundary;
            if boundary < pcm.len() {
                let first = boundary - search;
                end = first;
                let mut lowest = f32::INFINITY;
                // The reference range excludes the final candidate even when
                // the search region is an exact multiple of the energy window.
                for offset in (0..search.saturating_sub(window)).step_by(window) {
                    let at = first + offset;
                    let mean = pairwise_square_sum(&pcm[at..at + window]) / window as f32;
                    let energy = mean.sqrt();
                    if !energy.is_finite() { return Err(invalid("nonfinite audio split energy")); }
                    if energy < lowest { lowest = energy; end = at; }
                }
                if end <= begin { end = boundary; }
            }
            ranges.push(begin..end);
            begin = end;
        }
        Ok(ranges)
    }

    pub fn prepare_pcm(&self, pcm: &[f32], capacity: &crate::model::contract::AudioCapacity,
                       cancelled: &AtomicBool) -> Result<(std::sync::Arc<PreparedAudio>, [u8; 32])> {
        self.prepare_pcm_cached(pcm, capacity, cancelled, None)
    }

    pub fn prepare_pcm_cached(&self, pcm: &[f32], capacity: &crate::model::contract::AudioCapacity,
                             cancelled: &AtomicBool, store: Option<&flare_selene::input::InputStore>)
        -> Result<(std::sync::Arc<PreparedAudio>, [u8; 32])> {
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio preparation cancelled")); }
        let identity = semantic_identity(pcm, &self.settings)?;
        let key = cache::key(identity);
        let _source = store.map(|store| store.source_guard(key));
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio preparation cancelled")); }
        let frames = match self.settings.sample_padding {
            crate::model::contract::AudioSamplePadding::Unpadded => pcm.len() / HOP,
            crate::model::contract::AudioSamplePadding::HopMultiple => pcm.len().div_ceil(HOP),
        };
        let prepared_samples = match self.settings.sample_padding {
            crate::model::contract::AudioSamplePadding::Unpadded => pcm.len(),
            crate::model::contract::AudioSamplePadding::HopMultiple => frames * HOP,
        };
        if prepared_samples <= FFT / 2 || prepared_samples > 600 * RATE {
            return Err(invalid("invalid finite PCM extent for centered audio STFT"));
        }
        if let Some(mel) = store.and_then(|store| cache::lookup(store, key, frames)) {
            let audio = std::sync::Arc::new(PreparedAudio::new(mel, frames, capacity)?);
            if cancelled.load(Ordering::Acquire) { return Err(invalid("audio preparation cancelled")); }
            return Ok((audio, identity));
        }
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio preparation cancelled")); }
        let plan = self.plans.lock().unwrap_or_else(|error| error.into_inner()).pop()
            .ok_or_else(|| Error::admission(crate::AdmissionError::Capacity,
                "native audio feature preparation is at capacity"))?;
        let mut lease = PlanLease { pool: &self.plans, plan: Some(plan) };
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio preparation cancelled")); }
        let features = lease.plan.as_mut().unwrap().prepared_features(pcm, &self.settings, cancelled)?;
        let frames = features.len() / MELS;
        let audio = std::sync::Arc::new(PreparedAudio::new(features, frames, capacity)?);
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio preparation cancelled")); }
        if let Some(store) = store { cache::insert(store, key, &audio)?; }
        if cancelled.load(Ordering::Acquire) { return Err(invalid("audio preparation cancelled")); }
        Ok((audio, identity))
    }
}

// Match the reference's FP32 square and NumPy pairwise mean reduction. The
// eight-lane, 128-element block order is defined in NumPy v2.2.6 loops_utils.h.src:
// https://github.com/numpy/numpy/blob/v2.2.6/numpy/_core/src/umath/loops_utils.h.src
fn pairwise_square_sum(input: &[f32]) -> f32 {
    if input.len() < 8 {
        return input.iter().fold(-0.0, |sum, value| sum + value * value);
    }
    if input.len() > 128 {
        let midpoint = (input.len() / 2) / 8 * 8;
        return pairwise_square_sum(&input[..midpoint]) + pairwise_square_sum(&input[midpoint..]);
    }
    let mut lanes: [f32; 8] = std::array::from_fn(|index| input[index] * input[index]);
    let full = input.len() / 8 * 8;
    for values in input[8..full].chunks_exact(8) {
        for (sum, value) in lanes.iter_mut().zip(values) { *sum += value * value; }
    }
    let mut sum = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3]))
        + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
    for value in &input[full..] { sum += value * value; }
    sum
}

impl LogMelPlan {
    pub fn new() -> Result<Self> {
        let fft = NonNull::new(unsafe { flare_audio_fft_new() })
            .ok_or_else(|| invalid("cannot prepare native 400-point audio FFT"))?;
        let hann = std::array::from_fn(|i| {
            (0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / FFT as f64).cos())) as f32
        });
        let log_step = 6.4_f64.ln() / 27.0;
        let last_mel = 15.0 + (8000.0_f64 / 1000.0).ln() / log_step;
        let edges: Vec<_> = (0..MELS + 2)
            .map(|i| {
                let mel = last_mel * i as f64 / (MELS + 1) as f64;
                if mel < 15.0 {
                    mel * (200.0 / 3.0)
                } else {
                    1000.0 * ((mel - 15.0) * log_step).exp()
                }
            })
            .collect();
        let mut filters = vec![0.0; MELS * BINS];
        for mel in 0..MELS {
            for bin in 0..BINS {
                let hz = bin as f64 * RATE as f64 / FFT as f64;
                let rising = (hz - edges[mel]) / (edges[mel + 1] - edges[mel]);
                let falling = (edges[mel + 2] - hz) / (edges[mel + 2] - edges[mel + 1]);
                filters[mel * BINS + bin] =
                    (rising.min(falling).max(0.0) * 2.0 / (edges[mel + 2] - edges[mel])) as f32;
            }
        }
        Ok(Self {
            fft,
            hann,
            filters,
            input: [0.0; FFT],
            spectrum: [Complex::default(); BINS],
        })
    }

    pub fn prepared_features(&mut self, pcm: &[f32], settings: &super::AudioSettings,
                             cancelled: &AtomicBool) -> Result<Vec<f32>> {
        settings.validate()?;
        if pcm.is_empty() || pcm.len() > settings.max_samples {
            return Err(invalid("audio exceeds the original decoded sample limit"));
        }
        match settings.sample_padding {
            crate::model::contract::AudioSamplePadding::Unpadded => self.features(pcm, cancelled),
            crate::model::contract::AudioSamplePadding::HopMultiple => {
                let extent = pcm.len().div_ceil(HOP) * HOP;
                if extent == pcm.len() { return self.features(pcm, cancelled); }
                let mut padded = Vec::with_capacity(extent);
                padded.extend_from_slice(pcm);
                padded.resize(extent, 0.0);
                self.features(&padded, cancelled)
            }
        }
    }

    /// Returns row-major [128, floor(samples/160)] features. Input is already
    /// mono 16 kHz. Reflection and normalization use this input alone; graph
    /// capacity padding is applied by the caller after these features exist.
    pub fn features(&mut self, pcm: &[f32], cancelled: &AtomicBool) -> Result<Vec<f32>> {
        if pcm.len() <= FFT / 2 || pcm.len() > 600 * RATE || pcm.iter().any(|v| !v.is_finite()) {
            return Err(invalid("invalid finite PCM extent for centered audio STFT"));
        }
        let frames = pcm.len() / HOP;
        let mut output = vec![0.0; MELS * frames];
        let mut maximum = f32::NEG_INFINITY;
        for frame in 0..frames {
            if cancelled.load(Ordering::Acquire) {
                return Err(invalid("audio features cancelled"));
            }
            for i in 0..FFT {
                let position = (frame * HOP + i) as isize - (FFT / 2) as isize;
                let reflected = if position < 0 {
                    -position
                } else if position >= pcm.len() as isize {
                    2 * pcm.len() as isize - 2 - position
                } else {
                    position
                };
                self.input[i] = pcm[reflected as usize] * self.hann[i];
            }
            unsafe {
                flare_audio_fft_run(
                    self.fft.as_ptr(),
                    self.input.as_mut_ptr(),
                    self.spectrum.as_mut_ptr(),
                )
            };
            let power: [f32; BINS] = std::array::from_fn(|i| {
                let magnitude = self.spectrum[i].re.hypot(self.spectrum[i].im);
                magnitude * magnitude
            });
            if power.iter().any(|value| !value.is_finite()) {
                return Err(invalid("nonfinite audio power spectrum"));
            }
            for mel in 0..MELS {
                let mut value = 0.0_f32;
                for (filter, power) in self.filters[mel * BINS..(mel + 1) * BINS]
                    .iter()
                    .zip(&power)
                {
                    value += filter * power;
                }
                if !value.is_finite() {
                    return Err(invalid("nonfinite audio Mel reduction"));
                }
                let value = value.max(1e-10).log10();
                if !value.is_finite() {
                    return Err(invalid("nonfinite audio Log-Mel feature"));
                }
                output[mel * frames + frame] = value;
                maximum = maximum.max(value);
            }
        }
        for value in &mut output {
            *value = (value.max(maximum - 8.0) + 4.0) / 4.0;
        }
        Ok(output)
    }
}

impl Drop for LogMelPlan {
    fn drop(&mut self) {
        unsafe { flare_audio_fft_free(self.fft.as_ptr()) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> super::super::AudioSettings {
        super::super::AudioSettings {
            sample_rate: 16000, mel_bins: 128, fft_size: 400, hop_samples: 160,
            max_file_bytes: 25 * 1024 * 1024, max_samples: 600 * RATE,
            decode_timeout_millis: 60000, service_chunk_samples: 480000,
            silence_search_samples: 16000, silence_window_samples: 1600,
            sample_padding: crate::model::contract::AudioSamplePadding::HopMultiple,
        }
    }

    #[test]
    fn declared_padding_changes_frame_count_without_changing_decode_duration() {
        let mut plan = LogMelPlan::new().unwrap();
        let cancelled = AtomicBool::new(false);
        let mut settings = settings();
        for (samples, frames) in [(161, 2), (200, 2), (201, 2), (7999, 50), (16001, 101)] {
            let result = plan.prepared_features(&vec![0.0; samples], &settings, &cancelled).unwrap();
            assert_eq!(result.len(), MELS * frames);
            assert!(result.iter().all(|value| *value == -1.5));
        }
        for samples in [1, 160] {
            assert!(plan.prepared_features(&vec![0.0; samples], &settings, &cancelled).is_err());
        }
        settings.sample_padding = crate::model::contract::AudioSamplePadding::Unpadded;
        assert_eq!(plan.prepared_features(&vec![0.0; 16001], &settings, &cancelled).unwrap().len(), MELS * 100);
        assert!(plan.prepared_features(&vec![0.0; 200], &settings, &cancelled).is_err());
    }

    fn capacity() -> crate::model::contract::AudioCapacity {
        use crate::model::contract::{AudioCapacity, StateView};
        let view = || StateView { buffer: String::new(), offset: 0, stride: 1, nbytes: 1 };
        AudioCapacity {
            feature_slots: 1, features_per_request: 8192, feature_width: 2048,
            frame_capacities: vec![100, 800], mel_bins: MELS, frames_per_chunk: 100,
            features_per_chunk: 13, subsampling_factor: 8, attention_window_frames: 800,
            features: view(), generations: view(), counts: view(),
        }
    }

    #[test]
    fn transcription_splits_preserve_first_minimum_and_exclude_final_candidate() {
        let processor = AudioProcessor::new(settings(), 1).unwrap();
        let cancelled = AtomicBool::new(false);
        for length in [201, 480000, 480001, 960001] {
            let ranges = processor.transcription_ranges(&vec![0.0; length], &cancelled).unwrap();
            assert_eq!(ranges.first().unwrap().start, 0);
            assert_eq!(ranges.last().unwrap().end, length);
            assert!(ranges.windows(2).all(|pair| pair[0].end == pair[1].start));
            assert!(ranges.iter().all(|range| !range.is_empty() && range.len() <= 480000));
            if length > 480000 { assert_eq!(ranges[0].end, 464000); }
        }
        let mut pcm = vec![0.5; 600000];
        pcm[478400..480000].fill(0.0); // excluded last candidate
        pcm[467200..468800].fill(0.25);
        assert_eq!(processor.transcription_ranges(&pcm, &cancelled).unwrap()[0].end, 467200);
        pcm[465600..467200].fill(0.25); // an earlier tie wins
        assert_eq!(processor.transcription_ranges(&pcm, &cancelled).unwrap()[0].end, 465600);
        pcm.fill(0.6);
        pcm[464000..465600].fill(0.5);
        pcm[464000] = 0.5000477;
        pcm[465600..467200].fill(0.5);
        let first = pairwise_square_sum(&pcm[464000..465600]) / 1600.;
        let second = pairwise_square_sum(&pcm[465600..467200]) / 1600.;
        assert!(first > second);
        assert_eq!(first.sqrt(), second.sqrt());
        assert_eq!(processor.transcription_ranges(&pcm, &cancelled).unwrap()[0].end, 464000);
        cancelled.store(true, Ordering::Release);
        assert!(processor.transcription_ranges(&pcm, &cancelled).is_err());
        assert!(processor.transcription_ranges(&[f32::NAN], &AtomicBool::new(false)).is_err());
        assert!(processor.transcription_ranges(&vec![f32::MAX; 600000], &AtomicBool::new(false)).is_err());
    }

    #[test]
    fn pooled_plans_return_after_rejection_cancellation_and_thread_reuse() {
        let processor = std::sync::Arc::new(AudioProcessor::new(settings(), 1).unwrap());
        let cancelled = AtomicBool::new(false);
        let pcm = processor.decode(&wav(16001, 2, 16000), &cancelled).unwrap();
        let (first, identity) = processor.prepare_pcm(&pcm, &capacity(), &cancelled).unwrap();
        assert_eq!(first.frames, 101);
        assert_eq!(first.feature_count(), 14);
        assert_eq!(processor.plans.lock().unwrap().len(), 1);
        let mut other = pcm.clone(); other[0] += 0.125;
        assert_ne!(identity, semantic_identity(&other, &settings()).unwrap());
        let mut different_settings = settings(); different_settings.sample_padding =
            crate::model::contract::AudioSamplePadding::Unpadded;
        assert_ne!(identity, semantic_identity(&pcm, &different_settings).unwrap());
        for invalid_pcm in [vec![0.0; 160], vec![f32::NAN; 201]] {
            assert!(processor.prepare_pcm(&invalid_pcm, &capacity(), &cancelled).is_err());
            assert_eq!(processor.plans.lock().unwrap().len(), 1);
        }
        cancelled.store(true, Ordering::Release);
        assert!(processor.prepare_pcm(&pcm, &capacity(), &cancelled).is_err());
        assert_eq!(processor.plans.lock().unwrap().len(), 1);
        let retained = processor.plans.lock().unwrap().pop().unwrap();
        assert!(matches!(processor.prepare_pcm(&pcm, &capacity(), &AtomicBool::new(false)),
            Err(Error::Admission(crate::AdmissionError::Capacity(_)))));
        processor.plans.lock().unwrap().push(retained);
        let owner = processor.clone();
        let repeated = std::thread::spawn(move || {
            let (audio, repeated_identity) = owner.prepare_pcm(&pcm, &capacity(), &AtomicBool::new(false)).unwrap();
            assert_eq!(identity, repeated_identity);
            audio
        }).join().unwrap();
        assert_eq!(first.mel.as_ref(), repeated.mel.as_ref());
        assert_eq!(processor.plans.lock().unwrap().len(), 1);
        assert!(AudioProcessor::new(settings(), 0).is_err());
        assert!(AudioProcessor::new(settings(), 65).is_err());
    }

    #[test]
    fn whole_input_feature_cache_skips_fft_and_revalidates_identity_capacity_and_records() {
        use flare_selene::input::{DiskBudget, InputStore, PreparedInputRecord};
        use std::sync::Arc;

        let store = InputStore::start(4 << 20, None, DiskBudget::new(0, 0)).unwrap();
        let processor = AudioProcessor::new(settings(), 1).unwrap();
        let cancelled = AtomicBool::new(false);
        let pcm = processor.decode(&wav(16001, 2, 16000), &cancelled).unwrap();
        let (first, identity) = processor.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)).unwrap();
        let key = cache::key(identity);
        assert!(store.get_blocking(key, true).is_some());
        let plan = processor.plans.lock().unwrap().pop().unwrap();
        assert!(matches!(processor.prepare_pcm(&pcm, &capacity(), &cancelled),
            Err(Error::Admission(crate::AdmissionError::Capacity(_)))));
        let (hit, repeated_identity) = processor.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)).unwrap();
        assert_eq!(identity, repeated_identity);
        assert_eq!(first.mel.as_ref(), hit.mel.as_ref());
        assert!(store.stats().payload_hits > 0);
        let mut other = pcm.clone(); other[0] += 0.125;
        assert!(matches!(processor.prepare_pcm_cached(&other, &capacity(), &cancelled, Some(&store)),
            Err(Error::Admission(crate::AdmissionError::Capacity(_)))));
        let mut smaller = capacity(); smaller.features_per_request = first.feature_count() - 1;
        let failure = processor.prepare_pcm_cached(&pcm, &smaller, &cancelled, Some(&store)).err().unwrap();
        assert!(failure.to_string().contains("exceeds prepared feature storage"));
        assert!(processor.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)).is_ok());
        cancelled.store(true, Ordering::Release);
        assert!(processor.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)).is_err());
        cancelled.store(false, Ordering::Release);
        processor.plans.lock().unwrap().push(plan);
        for kind in 0..3 {
            store.invalidate(key);
            assert!(store.get_blocking(key, true).is_none());
            let mut payload: Vec<_> = first.mel.iter().flat_map(|value| value.to_le_bytes()).collect();
            let mut metadata = b"flare-centered-slaney-f32-v1\0".to_vec();
            match kind {
                0 => metadata[0] ^= 1,
                1 => payload.truncate(payload.len() - 4),
                _ => payload[..4].copy_from_slice(&f32::NAN.to_le_bytes()),
            }
            store.insert(key, Arc::new(PreparedInputRecord { metadata: metadata.into(), payload: Some(Arc::new(payload)) }));
            assert!(store.get_blocking(key, true).is_some());
            let (recovered, recovered_identity) = processor.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)).unwrap();
            assert_eq!(identity, recovered_identity);
            assert_eq!(first.mel.as_ref(), recovered.mel.as_ref());
            assert_eq!(processor.plans.lock().unwrap().len(), 1);
        }
        let mut altered = settings(); altered.sample_padding = crate::model::contract::AudioSamplePadding::Unpadded;
        let altered = AudioProcessor::new(altered, 1).unwrap();
        let plan = altered.plans.lock().unwrap().pop().unwrap();
        assert!(matches!(altered.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)),
            Err(Error::Admission(crate::AdmissionError::Capacity(_)))));
        altered.plans.lock().unwrap().push(plan);
    }

    #[test]
    fn cached_features_retain_normalization_across_encoder_windows() {
        use flare_selene::input::{DiskBudget, InputStore};
        let store = InputStore::start(4 << 20, None, DiskBudget::new(0, 0)).unwrap();
        let processor = AudioProcessor::new(settings(), 1).unwrap();
        let cancelled = AtomicBool::new(false);
        let pcm: Vec<_> = (0..900 * HOP + 1).map(|i|
            (i as f32 * 0.01).sin() * if i < 800 * HOP { 0.0001 } else { 0.5 }).collect();
        let (original, identity) = processor.prepare_pcm(&pcm, &capacity(), &cancelled).unwrap();
        let (first, _) = processor.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)).unwrap();
        assert_eq!(original.mel.as_ref(), first.mel.as_ref());
        assert!(store.get_blocking(cache::key(identity), true).unwrap().payload.as_ref().unwrap().len() > 800 * MELS * 4);
        let plan = processor.plans.lock().unwrap().pop().unwrap();
        let (hit, repeated_identity) = processor.prepare_pcm_cached(&pcm, &capacity(), &cancelled, Some(&store)).unwrap();
        assert_eq!(identity, repeated_identity);
        assert_eq!(hit.windows.len(), 2);
        for index in 0..2 {
            let before = original.window_buffers(index, 800).unwrap();
            let after = hit.window_buffers(index, 800).unwrap();
            assert_eq!(before.mel, after.mel);
            assert_eq!(before.gather_indices, after.gather_indices);
            assert_eq!(before.feature_offset, after.feature_offset);
        }
        processor.plans.lock().unwrap().push(plan);
    }

    #[test]
    fn feature_cache_does_not_admit_invalid_short_or_nonfinite_pcm() {
        use flare_selene::input::{DiskBudget, InputStore, PreparedInputRecord};
        use std::sync::Arc;

        let store = InputStore::start(1 << 20, None, DiskBudget::new(0, 0)).unwrap();
        for padding in [crate::model::contract::AudioSamplePadding::Unpadded,
            crate::model::contract::AudioSamplePadding::HopMultiple] {
            let mut settings = settings(); settings.sample_padding = padding;
            let processor = AudioProcessor::new(settings.clone(), 1).unwrap();
            for samples in [1, 160, 161, 200, 201] {
                let pcm = vec![0.; samples];
                let identity = semantic_identity(&pcm, &settings).unwrap();
                let key = cache::key(identity);
                let frames = if padding == crate::model::contract::AudioSamplePadding::HopMultiple {
                    samples.div_ceil(HOP)
                } else { samples / HOP };
                store.insert(key, Arc::new(PreparedInputRecord {
                    metadata: Arc::from(&b"flare-centered-slaney-f32-v1\0"[..]),
                    payload: Some(Arc::new(vec![0; frames * MELS * 4])),
                }));
                assert!(store.get_blocking(key, true).is_some());
                let valid = samples > 200 || padding == crate::model::contract::AudioSamplePadding::HopMultiple && samples > 160;
                assert_eq!(processor.prepare_pcm_cached(&pcm, &capacity(), &AtomicBool::new(false), Some(&store)).is_ok(), valid);
                store.invalidate(key);
                assert!(store.get_blocking(key, true).is_none());
            }
            assert!(processor.prepare_pcm_cached(&[f32::NAN; 201], &capacity(), &AtomicBool::new(false), Some(&store)).is_err());
            assert_eq!(store.stats().memory_records, 0);
        }
    }

    #[test]
    fn encoder_windows_keep_full_input_normalization_and_convolution_width() {
        let source: Vec<_> = (0..MELS * 801).map(|index| (index % 801) as f32).collect();
        let audio = PreparedAudio::new(source, 801, &capacity()).unwrap();
        assert_eq!(audio.feature_count(), 105);
        assert_eq!(audio.windows.len(), 2);
        let tail = audio.window_buffers(1, 100).unwrap();
        assert_eq!((tail.frames, tail.feature_count, tail.feature_offset, tail.convolution_width), (1, 1, 104, 100));
        assert_eq!(tail.cumulative_lengths, [0, 1]);
        assert_eq!(tail.gather_indices[0], 0);
        assert!(tail.gather_indices[1..].iter().all(|value| *value == -1));
        for bin in 0..MELS {
            assert_eq!(tail.mel[bin * 100], 800.0);
            assert!(tail.mel[bin * 100 + 1..(bin + 1) * 100].iter().all(|value| *value == 0.0));
        }
        assert!(audio.window_buffers(0, 100).is_err());
        let short = PreparedAudio::new(vec![0.0; MELS * 99], 99, &capacity()).unwrap();
        assert_eq!(short.window_buffers(0, 100).unwrap().convolution_width, 99);
        let crossing = PreparedAudio::new(vec![0.0; MELS * 101], 101, &capacity()).unwrap();
        assert_eq!(crossing.feature_count(), 14);
        assert_eq!(crossing.window_buffers(0, 800).unwrap().gather_indices[13], 13);
    }

    fn wav(samples: usize, channels: usize, rate: usize) -> Vec<u8> {
        let size = samples * channels * 2;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&((36 + size) as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&(channels as u16).to_le_bytes());
        bytes.extend_from_slice(&(rate as u32).to_le_bytes());
        bytes.extend_from_slice(&((rate * channels * 2) as u32).to_le_bytes());
        bytes.extend_from_slice(&((channels * 2) as u16).to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(size as u32).to_le_bytes());
        for _ in 0..samples {
            for _ in 0..channels {
                bytes.extend_from_slice(&8192i16.to_le_bytes());
            }
        }
        bytes
    }

    #[test]
    fn decode_exact_limit_and_equal_channel_mean() {
        let cancelled = AtomicBool::new(false);
        let limits = AudioLimits {
            samples: 16000,
            ..AudioLimits::default()
        };
        for channels in [1, 2] {
            let pcm = decode(&wav(16000, channels, 16000), limits, &cancelled).unwrap();
            assert_eq!(pcm.len(), 16000);
            assert!(pcm.iter().all(|value| *value == 0.25));
        }
        for (second, expected) in [(-8192i16, 0.0), (16384i16, 0.375)] {
            let mut stereo = wav(16000, 2, 16000);
            for frame in stereo[44..].chunks_exact_mut(4) {
                frame[2..].copy_from_slice(&second.to_le_bytes());
            }
            assert!(
                decode(&stereo, limits, &cancelled)
                    .unwrap()
                    .iter()
                    .all(|v| *v == expected)
            );
        }
        let resampled = decode(&wav(48000, 1, 48000), limits, &cancelled).unwrap();
        assert_eq!(resampled.len(), 16000);
        assert!(resampled.iter().all(|v| v.is_finite()));
        assert!(decode(&wav(16001, 1, 16000), limits, &cancelled).is_err());
    }

    #[test]
    fn reject_short_corrupt_and_cancelled_audio() {
        let cancelled = AtomicBool::new(false);
        assert!(decode(b"corrupt", AudioLimits::default(), &cancelled).is_err());
        assert_eq!(decode(&wav(200, 1, 16000), AudioLimits::default(), &cancelled).unwrap().len(), 200);
        assert!(LogMelPlan::new().unwrap().features(&vec![0.0; 200], &cancelled).is_err());
        assert_eq!(
            decode(&wav(201, 1, 16000), AudioLimits::default(), &cancelled)
                .unwrap()
                .len(),
            201
        );
        cancelled.store(true, Ordering::Release);
        assert!(decode(&wav(16000, 1, 16000), AudioLimits::default(), &cancelled).is_err());
    }

    #[test]
    fn fft_impulse_and_silence_have_independent_exact_references() {
        let mut plan = LogMelPlan::new().unwrap();
        plan.input[0] = 1.0;
        unsafe {
            flare_audio_fft_run(
                plan.fft.as_ptr(),
                plan.input.as_mut_ptr(),
                plan.spectrum.as_mut_ptr(),
            )
        };
        assert!(
            plan.spectrum
                .iter()
                .all(|value| value.re == 1.0 && value.im == 0.0)
        );
        let result = plan
            .features(&vec![0.0; 16000], &AtomicBool::new(false))
            .unwrap();
        assert_eq!(result.len(), 128 * 100);
        assert!(result.iter().all(|value| *value == -1.5));
    }

    #[test]
    fn sequential_requests_do_not_share_normalization_or_fft_history() {
        let mut plan = LogMelPlan::new().unwrap();
        let input: Vec<_> = (0..16001)
            .map(|i| ((i as f32) * 0.19).sin() * 0.1)
            .collect();
        let cancelled = AtomicBool::new(false);
        let first = plan.features(&input, &cancelled).unwrap();
        plan.features(&vec![0.0; 201], &cancelled).unwrap();
        assert_eq!(first, plan.features(&input, &cancelled).unwrap());
        assert!(plan.features(&vec![f32::NAN; 201], &cancelled).is_err());
        assert!(plan.features(&vec![f32::MAX; 201], &cancelled).is_err());
        assert!(plan.features(&vec![0.0; 200], &cancelled).is_err());
    }
}
