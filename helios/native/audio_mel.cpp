/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

// Fixed FP32 frontend arithmetic. The legacy FFmpeg transform remains in
// audio.c for packages that do not declare these numerical semantics.
#include <cstddef>
#include <cstdint>

#if defined(__linux__) && defined(__aarch64__) && defined(__ARM_FEATURE_SVE)
#define POCKETFFT_CACHE_SIZE 1
#include "pocketfft_hdronly.h"
#include <arm_sve.h>
#include <array>
#include <cmath>
#include <memory>
#include <vector>

extern "C" svfloat32_t Sleef_cosfx_u10sve(svfloat32_t);
extern "C" svfloat32_t Sleef_log10fx_u10sve(svfloat32_t);

namespace {
constexpr size_t fft = 400, bins = 201, mels = 128, hop = 160;
struct MelPlan {
    std::array<float, fft> hann;
    std::array<float, mels * bins> filters;
    // Pin the cached immutable transform. Each call owns separate bounded
    // input/output scratch; no request can modify another request's plan.
    std::shared_ptr<pocketfft::detail::pocketfft_r<float>> transform;

    explicit MelPlan(const float *coefficients)
        : transform(pocketfft::detail::get_plan<pocketfft::detail::pocketfft_r<float>>(fft)) {
        std::copy_n(coefficients, filters.size(), filters.begin());
        const float step = static_cast<float>(6.283185307179586476925286766559 / fft);
        for (size_t i = 0; i < fft; ++i) hann[i] = static_cast<float>(i) * step;
        for (size_t i = 0; i < fft; i += svcntw()) {
            auto active = svwhilelt_b32(i, fft);
            svst1(active, hann.data() + i, Sleef_cosfx_u10sve(svld1(active, hann.data() + i)));
        }
        for (auto &value : hann) {
            // These are two distinct FP32 nodes in the checkpoint frontend.
            volatile float half = value * -0.5f;
            value = half + 0.5f;
        }
    }
};
}

extern "C" void *flare_audio_mel_new(const float *filters) noexcept {
    if (!filters) return nullptr;
    try { return new MelPlan(filters); }
    catch (...) { return nullptr; }
}

extern "C" int flare_audio_mel_run(void *opaque, const float *pcm, size_t samples,
    float *output, int (*cancelled)(void *), void *context) noexcept {
    if (!opaque || !pcm || !output || !cancelled || samples <= 200 || samples > 600 * 16000)
        return -1;
    try {
        auto &plan = *static_cast<MelPlan *>(opaque);
        const size_t frames = samples / hop, stft_frames = frames + 1;
        std::vector<float> input(stft_frames * fft);
        std::vector<std::complex<float>> spectrum(stft_frames * bins);
        for (size_t frame = 0; frame < stft_frames; ++frame) {
            if (cancelled(context)) return -2;
            for (size_t i = 0; i < fft; ++i) {
                ptrdiff_t at = static_cast<ptrdiff_t>(frame * hop + i) - static_cast<ptrdiff_t>(fft / 2);
                if (at < 0) at = -at;
                else if (static_cast<size_t>(at) >= samples) at = static_cast<ptrdiff_t>(2 * samples - 2) - at;
                input[frame * fft + i] = pcm[at] * plan.hann[i];
            }
        }
        // The final STFT frame is discarded only after the transform. Its
        // presence controls the final vector/scalar FFT group in the reference.
        pocketfft::r2c<float>({stft_frames, fft}, {fft * 4, 4}, {bins * 8, 8}, {1},
            true, input.data(), spectrum.data(), 1.0f, 1);
        float maximum = -INFINITY;
        for (size_t frame = 0; frame < frames; ++frame) {
            if (cancelled(context)) return -2;
            std::array<float, bins> power;
            for (size_t bin = 0; bin < bins; ++bin) {
                auto value = spectrum[frame * bins + bin];
                float magnitude = hypotf(value.real(), value.imag());
                power[bin] = magnitude * magnitude;
                if (!std::isfinite(power[bin])) return -3;
            }
            for (size_t mel = 0; mel < mels; ++mel) {
                float sum = 0;
                for (size_t bin = 0; bin < bins; ++bin)
                    sum = fmaf(plan.filters[mel * bins + bin], power[bin], sum);
                if (!std::isfinite(sum)) return -3;
                output[mel * frames + frame] = fmaxf(sum, 1e-10f);
            }
        }
        for (size_t i = 0; i < frames * mels; i += svcntw()) {
            auto active = svwhilelt_b32(i, frames * mels);
            svst1(active, output + i, Sleef_log10fx_u10sve(svld1(active, output + i)));
        }
        for (size_t i = 0; i < frames * mels; ++i) {
            if (!std::isfinite(output[i])) return -3;
            maximum = fmaxf(maximum, output[i]);
        }
        for (size_t i = 0; i < frames * mels; ++i)
            output[i] = (fmaxf(output[i], maximum - 8.0f) + 4.0f) / 4.0f;
        return cancelled(context) ? -2 : 0;
    } catch (...) { return -4; }
}

extern "C" void flare_audio_mel_free(void *opaque) noexcept {
    delete static_cast<MelPlan *>(opaque);
}
#else
extern "C" void *flare_audio_mel_new(const float *) noexcept { return nullptr; }
extern "C" int flare_audio_mel_run(void *, const float *, size_t, float *,
    int (*)(void *), void *) noexcept { return -1; }
extern "C" void flare_audio_mel_free(void *) noexcept {}
#endif
