/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

/* FFmpeg owns opaque codec/transform contexts; Rust owns the bounded buffers. */
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/intreadwrite.h>
#include <libavutil/mem.h>
#include <libavutil/tx.h>
#include <libswresample/swresample.h>

typedef int (*FlareCancelled)(void *);
typedef struct {
    const uint8_t *bytes;
    size_t size, position;
    FlareCancelled cancelled;
    void *cancel_context;
} FlareAudioInput;

typedef struct {
    int sample_rate;
    enum AVSampleFormat sample_format;
    AVChannelLayout layout;
} FlareAudioGeometry;

static int interrupted(void *opaque) {
    FlareAudioInput *input = opaque;
    return input->cancelled(input->cancel_context);
}

static int read_bytes(void *opaque, uint8_t *destination, int size) {
    FlareAudioInput *input = opaque;
    if (interrupted(opaque)) return AVERROR_EXIT;
    size_t count = input->size - input->position;
    if (!count) return AVERROR_EOF;
    if (count > (size_t)size) count = (size_t)size;
    memcpy(destination, input->bytes + input->position, count);
    input->position += count;
    return (int)count;
}

static int64_t seek_bytes(void *opaque, int64_t offset, int whence) {
    FlareAudioInput *input = opaque;
    if (interrupted(opaque)) return AVERROR_EXIT;
    if (whence == AVSEEK_SIZE) return (int64_t)input->size;
    whence &= ~AVSEEK_FORCE;
    int64_t base;
    if (whence == SEEK_SET) base = 0;
    else if (whence == SEEK_CUR) base = (int64_t)input->position;
    else if (whence == SEEK_END) base = (int64_t)input->size;
    else return AVERROR(EINVAL);
    if (offset < -base || offset > (int64_t)input->size - base) return AVERROR(EINVAL);
    input->position = (size_t)(base + offset);
    return (int64_t)input->position;
}

static int deny_external(AVFormatContext *context, AVIOContext **io, const char *url,
                         int flags, AVDictionary **options) {
    (void)context; (void)io; (void)url; (void)flags; (void)options;
    return AVERROR(EPERM);
}

/* A demuxer can return clean EOF when a truncated WAV ends exactly on a packet
 * boundary. Check the declared RIFF and chunk extents before decoding it. */
static int validate_riff(const uint8_t *bytes, size_t size) {
    if (size < 4) return 0;
    int extended = !memcmp(bytes, "RF64", 4) || !memcmp(bytes, "BW64", 4);
    if (!extended && memcmp(bytes, "RIFF", 4) && memcmp(bytes, "RIFX", 4)) return 0;
    if (size < 12 || memcmp(bytes + 8, "WAVE", 4)) return AVERROR_INVALIDDATA;
    int little = extended || !memcmp(bytes, "RIFF", 4);
    uint64_t declared = little ? AV_RL32(bytes + 4) : AV_RB32(bytes + 4);
    uint64_t data_size = 0;
    if (extended) {
        if (size < 48 || declared != UINT32_MAX || memcmp(bytes + 12, "ds64", 4)) return AVERROR_INVALIDDATA;
        uint32_t ds_size = AV_RL32(bytes + 16);
        if (ds_size < 28 || ds_size > size - 20
                || (uint64_t)AV_RL32(bytes + 44) * 12 > ds_size - 28) return AVERROR_INVALIDDATA;
        declared = AV_RL64(bytes + 20);
        data_size = AV_RL64(bytes + 28);
        if (data_size > size) return AVERROR_INVALIDDATA;
    }
    if (declared < 4 || declared != size - 8) return AVERROR_INVALIDDATA;
    size_t end = size;
    int data_seen = 0;
    for (size_t offset = 12; offset < end;) {
        if (end - offset < 8) return AVERROR_INVALIDDATA;
        uint64_t chunk = little ? AV_RL32(bytes + offset + 4) : AV_RB32(bytes + offset + 4);
        if (extended && !memcmp(bytes + offset, "data", 4)) {
            if (data_seen++ || chunk != UINT32_MAX) return AVERROR_INVALIDDATA;
            chunk = data_size;
        } else if (chunk == UINT32_MAX) {
            return AVERROR_INVALIDDATA;
        }
        uint64_t extent = 8 + chunk + (chunk & 1);
        if (extent > end - offset) return AVERROR_INVALIDDATA;
        offset += (size_t)extent;
    }
    if (extended && !data_seen) return AVERROR_INVALIDDATA;
    return 0;
}

static int receive_frames(AVCodecContext *codec, AVFrame *frame, SwrContext *resampler,
                           const FlareAudioGeometry *geometry, FlareAudioInput *input,
                           float *output, size_t capacity,
                           size_t *written) {
    for (;;) {
        if (interrupted(input)) return AVERROR_EXIT;
        int result = avcodec_receive_frame(codec, frame);
        if (result == AVERROR(EAGAIN) || result == AVERROR_EOF) return 0;
        if (result < 0) return result;
        if ((frame->flags & AV_FRAME_FLAG_CORRUPT) || frame->decode_error_flags
                || frame->sample_rate != geometry->sample_rate
                || frame->format != geometry->sample_format
                || av_channel_layout_compare(&frame->ch_layout, &geometry->layout)) {
            av_frame_unref(frame);
            return AVERROR_INVALIDDATA;
        }
        size_t remaining = capacity - *written;
        if (!remaining) { av_frame_unref(frame); return AVERROR(ENOSPC); }
        uint8_t *destination = (uint8_t *)(output + *written);
        result = swr_convert(resampler, &destination, (int)remaining,
                             (const uint8_t **)frame->extended_data, frame->nb_samples);
        av_frame_unref(frame);
        if (result < 0) return result;
        *written += (size_t)result;
        if (*written == capacity) return AVERROR(ENOSPC);
    }
}

/* capacity includes one sentinel sample, so exact duration limits are accepted
 * and longer decoded audio fails rather than returning a truncated success. */
int flare_audio_decode(const uint8_t *bytes, size_t size, float *output, size_t capacity,
                       size_t *written, FlareCancelled cancelled, void *cancel_context,
                       char *message, size_t message_capacity) {
    AVFormatContext *format = NULL;
    AVCodecContext *codec = NULL;
    AVIOContext *io = NULL;
    AVPacket *packet = NULL;
    AVFrame *frame = NULL;
    SwrContext *resampler = NULL;
    AVDictionary *options = NULL;
    AVDictionary *probe_options[16] = {0};
    int result = AVERROR(EINVAL);
    FlareAudioInput input = {bytes, size, 0, cancelled, cancel_context};
    FlareAudioGeometry geometry = {0};
    *written = 0;
    if (!size || size > INT_MAX || !capacity || capacity > INT_MAX || !cancelled) goto cleanup;
    result = validate_riff(bytes, size);
    if (result < 0) goto cleanup;
    uint8_t *buffer = av_malloc(32768);
    if (!buffer) { result = AVERROR(ENOMEM); goto cleanup; }
    io = avio_alloc_context(buffer, 32768, 0, &input, read_bytes, NULL, seek_bytes);
    if (!io) { av_free(buffer); result = AVERROR(ENOMEM); goto cleanup; }
    format = avformat_alloc_context();
    if (!format) { result = AVERROR(ENOMEM); goto cleanup; }
    format->pb = io;
    format->flags |= AVFMT_FLAG_CUSTOM_IO;
    format->interrupt_callback = (AVIOInterruptCB){interrupted, &input};
    format->io_open = deny_external;
    format->max_streams = 16;
    av_dict_set(&options, "format_whitelist", "wav,flac,mp3,ogg,aac,mov,matroska,webm", 0);
    av_dict_set(&options, "probesize", "2097152", 0);
    av_dict_set(&options, "analyzeduration", "5000000", 0);
    av_dict_set(&options, "max_probe_packets", "2500", 0);
    result = avformat_open_input(&format, NULL, NULL, &options);
    if (result >= 0 && av_dict_count(options)) result = AVERROR(EINVAL);
    av_dict_free(&options);
    if (result < 0) goto cleanup;
    int stream = -1;
    if (format->nb_streams > 16) { result = AVERROR_INVALIDDATA; goto cleanup; }
    for (unsigned i = 0; i < format->nb_streams; i++) {
        if (format->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO) continue;
        if (stream >= 0) { result = AVERROR_INVALIDDATA; goto cleanup; }
        stream = (int)i;
    }
    if (stream < 0) { result = AVERROR_STREAM_NOT_FOUND; goto cleanup; }
    const AVCodec *decoder = avcodec_find_decoder(format->streams[stream]->codecpar->codec_id);
    if (!decoder) { result = AVERROR_DECODER_NOT_FOUND; goto cleanup; }
    /* Stream discovery can open decoders itself. Do not let an unused video
     * stream allocate frames before the bounded audio decoder is configured. */
    format->codec_whitelist = av_strdup(decoder->name);
    if (!format->codec_whitelist) { result = AVERROR(ENOMEM); goto cleanup; }
    for (unsigned i = 0; i < format->nb_streams; i++) {
        if ((int)i != stream) format->streams[i]->discard = AVDISCARD_ALL;
        if (av_dict_set(&probe_options[i], "threads", "1", 0) < 0
                || av_dict_set(&probe_options[i], "max_samples", "8388608", 0) < 0
                || av_dict_set(&probe_options[i], "max_pixels", "1", 0) < 0) {
            result = AVERROR(ENOMEM); goto cleanup;
        }
    }
    result = avformat_find_stream_info(format, probe_options);
    if (result < 0) goto cleanup;
    for (unsigned i = 0; i < format->nb_streams; i++) {
        if ((int)i != stream && format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            result = AVERROR_INVALIDDATA; goto cleanup;
        }
    }
    if (format->streams[stream]->codecpar->codec_id != decoder->id) {
        result = AVERROR_INVALIDDATA; goto cleanup;
    }
    codec = avcodec_alloc_context3(decoder);
    if (!codec) { result = AVERROR(ENOMEM); goto cleanup; }
    result = avcodec_parameters_to_context(codec, format->streams[stream]->codecpar);
    if (result < 0) goto cleanup;
    if (codec->sample_rate < 1 || codec->sample_rate > 768000
            || codec->ch_layout.nb_channels < 1 || codec->ch_layout.nb_channels > 64) {
        result = AVERROR_INVALIDDATA; goto cleanup;
    }
    codec->thread_count = 1;
    codec->max_samples = 8 * 1024 * 1024;
    result = avcodec_open2(codec, decoder, NULL);
    if (result < 0) goto cleanup;
    if (codec->sample_rate < 1 || codec->sample_rate > 768000
            || codec->ch_layout.nb_channels < 1 || codec->ch_layout.nb_channels > 64) {
        result = AVERROR_INVALIDDATA; goto cleanup;
    }
    geometry.sample_rate = codec->sample_rate;
    geometry.sample_format = codec->sample_fmt;
    result = av_channel_layout_copy(&geometry.layout, &codec->ch_layout);
    if (result < 0) goto cleanup;
    AVChannelLayout mono = AV_CHANNEL_LAYOUT_MONO;
    result = swr_alloc_set_opts2(&resampler, &mono, AV_SAMPLE_FMT_FLT, 16000,
                                &geometry.layout, geometry.sample_format, geometry.sample_rate, 0, NULL);
    if (result < 0) goto cleanup;
    double matrix[64];
    for (int i = 0; i < codec->ch_layout.nb_channels; i++) matrix[i] = 1.0 / codec->ch_layout.nb_channels;
    result = swr_set_matrix(resampler, matrix, codec->ch_layout.nb_channels);
    if (result < 0) goto cleanup;
    result = swr_init(resampler);
    if (result < 0) goto cleanup;
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    if (!packet || !frame) { result = AVERROR(ENOMEM); goto cleanup; }
    unsigned packets = 0;
    while ((result = av_read_frame(format, packet)) >= 0) {
        if (++packets > 1000000 || interrupted(&input)) { result = AVERROR_EXIT; goto cleanup; }
        if (packet->stream_index == stream) {
            if (packet->flags & AV_PKT_FLAG_CORRUPT) { result = AVERROR_INVALIDDATA; goto cleanup; }
            result = avcodec_send_packet(codec, packet);
            if (result < 0) goto cleanup;
            result = receive_frames(codec, frame, resampler, &geometry, &input, output, capacity, written);
            if (result < 0) goto cleanup;
        }
        av_packet_unref(packet);
    }
    if (result != AVERROR_EOF) goto cleanup;
    if (io->error < 0 && io->error != AVERROR_EOF) { result = io->error; goto cleanup; }
    result = avcodec_send_packet(codec, NULL);
    if (result < 0) goto cleanup;
    result = receive_frames(codec, frame, resampler, &geometry, &input, output, capacity, written);
    if (result < 0) goto cleanup;
    do {
        if (interrupted(&input)) { result = AVERROR_EXIT; goto cleanup; }
        size_t remaining = capacity - *written;
        if (!remaining) { result = AVERROR(ENOSPC); goto cleanup; }
        uint8_t *destination = (uint8_t *)(output + *written);
        result = swr_convert(resampler, &destination, (int)remaining, NULL, 0);
        if (result < 0) goto cleanup;
        *written += (size_t)result;
        if (*written == capacity) { result = AVERROR(ENOSPC); goto cleanup; }
    } while (result);
cleanup:
    if (result < 0 && message_capacity) av_strerror(result, message, message_capacity);
    for (unsigned i = 0; i < 16; i++) av_dict_free(&probe_options[i]);
    av_dict_free(&options);
    av_packet_free(&packet);
    av_frame_free(&frame);
    swr_free(&resampler);
    av_channel_layout_uninit(&geometry.layout);
    avcodec_free_context(&codec);
    avformat_close_input(&format);
    if (io) { av_freep(&io->buffer); avio_context_free(&io); }
    return result;
}

typedef struct { AVTXContext *context; av_tx_fn transform; } FlareAudioFft;

void *flare_audio_fft_new(void) {
    FlareAudioFft *plan = av_mallocz(sizeof(*plan));
    if (!plan) return NULL;
    float scale = 1.0f;
    if (av_tx_init(&plan->context, &plan->transform, AV_TX_FLOAT_RDFT, 0, 400,
                    &scale, AV_TX_UNALIGNED) < 0) { av_free(plan); return NULL; }
    return plan;
}

void flare_audio_fft_run(void *opaque, float *input, AVComplexFloat *output) {
    FlareAudioFft *plan = opaque;
    plan->transform(plan->context, output, input, sizeof(float));
}

void flare_audio_fft_free(void *opaque) {
    FlareAudioFft *plan = opaque;
    av_tx_uninit(&plan->context);
    av_free(plan);
}
