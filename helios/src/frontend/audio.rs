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
    if count <= FFT / 2 || count > limits.samples {
        return Err(invalid(
            "decoded audio requires more than 200 samples and must fit the duration limit",
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
        assert!(decode(&wav(200, 1, 16000), AudioLimits::default(), &cancelled).is_err());
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
