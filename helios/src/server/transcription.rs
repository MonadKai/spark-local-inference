/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Bounded non-streaming parents submit ordinary generations to the shared queue.

use super::{HttpState, Generation, GenerationEvent, runtime, request, http, protocol::Protocol};
use crate::{AdmissionError, Error, Result, frontend::IncrementalDecoder, output::Termination};
use axum::{Extension, Json, extract::{FromRequest, Multipart, Request, State},
    http::StatusCode, response::{IntoResponse, Response}};
use futures_util::{FutureExt, future::BoxFuture};
use serde_json::{Map, Value, json};
use std::{ops::Range, sync::{Arc, atomic::AtomicBool}, task::Poll};
use tokio::sync::OwnedSemaphorePermit;

struct Parameters {
    context: String,
    language: Option<String>,
    sampling: Map<String, Value>,
}

fn parameters(mut fields: Map<String, Value>) -> Result<Parameters> {
    if fields.remove("stream").is_some_and(|value| value != false)
        || fields.remove("n").is_some_and(|value| value != 1) {
        return Err(super::inference::unsupported("transcription requires non-streaming n=1"));
    }
    if fields.remove("response_format").is_some_and(|value| value != "json" && value != "text") {
        return Err(super::inference::unsupported("transcription response_format must be json or text"));
    }
    let context = fields.remove("prompt").map_or_else(String::new, |value| value.as_str().unwrap().to_owned());
    let language = fields.remove("language").map(|value| value.as_str().unwrap().to_owned());
    fields.entry("temperature").or_insert(json!(0));
    Ok(Parameters { context, language, sampling: fields })
}

async fn multipart(request: Request, state: &Arc<HttpState>, maximum: usize)
    -> std::result::Result<(Vec<u8>, Map<String, Value>), (StatusCode, String)> {
    if request.headers().get("content-encoding").is_some_and(|value| value != "identity") {
        return Err((StatusCode::UNSUPPORTED_MEDIA_TYPE, "multipart Content-Encoding must be identity".into()));
    }
    let mut form = Multipart::from_request(request, state).await
        .map_err(|error| (error.status(), error.to_string()))?;
    let mut file = None;
    let mut fields = Map::new();
    while let Some(mut field) = form.next_field().await.map_err(|error| (error.status(), error.to_string()))? {
        let name = field.name().ok_or((StatusCode::BAD_REQUEST, "multipart field requires a name".into()))?.to_owned();
        if name == "file" {
            if file.is_some() { return Err((StatusCode::BAD_REQUEST, "transcription requires exactly one file".into())); }
            let mut bytes = Vec::new();
            while let Some(chunk) = field.chunk().await.map_err(|error| (error.status(), error.to_string()))? {
                if chunk.len() > maximum - bytes.len() {
                    return Err((StatusCode::PAYLOAD_TOO_LARGE, "audio file exceeds its encoded byte limit".into()));
                }
                bytes.extend_from_slice(&chunk);
            }
            if bytes.is_empty() { return Err((StatusCode::BAD_REQUEST, "audio file is empty".into())); }
            file = Some(bytes);
            continue;
        }
        if !["model", "language", "prompt", "response_format", "stream", "n", "seed",
            "temperature", "top_p", "top_k", "min_p", "max_completion_tokens"].contains(&name.as_str()) {
            return Err((StatusCode::BAD_REQUEST, format!("transcription does not support {name}")));
        }
        if fields.contains_key(&name) { return Err((StatusCode::BAD_REQUEST, format!("duplicate transcription field {name}"))); }
        let text = field.text().await.map_err(|error| (error.status(), error.to_string()))?;
        let value = if ["model", "language", "prompt", "response_format"].contains(&name.as_str()) {
            Value::String(text)
        } else {
            serde_json::from_str(&text).map_err(|_| (StatusCode::BAD_REQUEST, format!("invalid transcription {name}")))?
        };
        if name == "stream" && !value.is_boolean() || name != "stream"
            && !["model", "language", "prompt", "response_format"].contains(&name.as_str()) && !value.is_number() {
            return Err((StatusCode::BAD_REQUEST, format!("invalid transcription {name}")));
        }
        fields.insert(name, value);
    }
    Ok((file.ok_or((StatusCode::BAD_REQUEST, "transcription requires a file".into()))?, fields))
}

pub(super) async fn generate(State(state): State<Arc<HttpState>>,
    Extension(ingress): Extension<Arc<OwnedSemaphorePermit>>, request: Request) -> Response {
    let Some(settings) = state.frontend.audio_settings() else {
        return http::error(Protocol::Chat, StatusCode::BAD_REQUEST, "prepared model has no audio frontend");
    };
    if state.transcription.is_none() {
        return http::error(Protocol::Chat, StatusCode::BAD_REQUEST, "prepared frontend has no transcription syntax");
    }
    let (bytes, mut fields) = match multipart(request, &state, settings.max_file_bytes).await {
        Ok(value) => value,
        Err((status, message)) => return http::error(Protocol::Chat, status, message),
    };
    if let Some(model) = fields.remove("model") {
        let model = model.as_str().unwrap();
        if !model.is_empty() && model != state.model.config().model_id
            && !state.served_names.iter().any(|alias| alias == model) {
            return http::error(Protocol::Chat, StatusCode::NOT_FOUND, "requested model is not loaded");
        }
    }
    let parameters = match parameters(fields) {
        Ok(value) => Arc::new(value),
        Err(error) => return http::error(Protocol::Chat, StatusCode::BAD_REQUEST, error),
    };
    let cancelled = Arc::new(AtomicBool::new(false));
    let _cancellation = http::PreparationCancellation(cancelled.clone());
    match transcribe(state, ingress, bytes, parameters, cancelled).await {
        Ok((text, samples, generated, truncated)) => {
            // The pinned endpoint returns JSON for both declared format names.
            let mut response = Json(json!({"text":text,"usage":{"type":"duration","seconds":samples.div_ceil(16000)}})).into_response();
            response.headers_mut().insert("x-flare-generated-tokens", generated.to_string().parse().unwrap());
            response.headers_mut().insert("x-flare-finish-reason", if truncated { "length" } else { "stop" }.parse().unwrap());
            response
        }
        Err(error) => (super::status(&error), Json(Protocol::Chat.failure(&error))).into_response(),
    }
}

struct Child<'a> {
    generation: Generation,
    decoder: IncrementalDecoder<'a>,
    raw: String,
}
enum Progress {
    Admitted(usize, Result<()>),
    Event(usize, GenerationEvent),
    Prepared(Result<(usize, Generation)>),
}

async fn prepare_child(state: Arc<HttpState>, ingress: Arc<OwnedSemaphorePermit>, pcm: Arc<Vec<f32>>,
    range: Range<usize>, rendered: Arc<crate::frontend::RenderedPrompt>, parameters: Arc<Parameters>,
    sampling: crate::model::SamplingParameters, cancelled: Arc<AtomicBool>, index: usize)
    -> Result<(usize, Generation)> {
    let permit = state.frontend_slots.clone().acquire_owned().await
        .map_err(|_| Error::admission(AdmissionError::Unavailable, "native frontend is shutting down"))?;
    let preparation_state = state.clone();
    let prompt = tokio::task::spawn_blocking(move || -> Result<runtime::Prompt> {
        let _permit = permit;
        let _ingress = ingress;
        let state = preparation_state;
        let processor = state.audio.as_ref().ok_or("audio processor is unavailable")?;
        let config = state.model.config();
        let (audio, identity) = processor.prepare_pcm(&pcm[range], config.contract.audio.as_ref().unwrap(), &cancelled)?;
        let (prepared, input) = crate::model::prepare_audio_input(config, &rendered, audio, identity)?;
        let mut boundaries = rendered.message_boundaries.clone();
        request::expand_boundaries(&mut boundaries, &prepared.token_expansions);
        let max_new_tokens = request::output_limit(&parameters.sampling, config.max_new_tokens,
            prepared.tokens.len(), config.limits.max_sequence_tokens)?;
        Ok(runtime::Prompt { tokens: prepared.tokens, message_boundaries: boundaries, data: prepared.data,
            conditioned_inputs: Vec::new(), conditioned_audio: Some(input), conditioning: prepared.conditioning,
            sampling, max_new_tokens, reasoning_cut: None, stop_sequences: Vec::new(), stop_strings: Vec::new(),
            grammar: None, ignore_eos: false, logprobs: false, logprob_token_ids: Vec::new(), prompt_logprobs: false })
    }).await.map_err(|error| Error::from(format!("transcription preparation worker failed: {error}")))?
        .map_err(|error| error.admission_context(AdmissionError::Invalid))?;
    // Each child keeps the same request seed. Chunk index is only an ordering
    // key; it must not select the ordinary multi-choice seed increment.
    Ok((index, state.model.generate_when_available(prompt).await?))
}

async fn transcribe(state: Arc<HttpState>, ingress: Arc<OwnedSemaphorePermit>, bytes: Vec<u8>,
    parameters: Arc<Parameters>, cancelled: Arc<AtomicBool>) -> Result<(String, usize, u64, bool)> {
    let renderer = state.transcription.as_ref().ok_or("transcription syntax is unavailable")?;
    let separator = renderer.separator(parameters.language.as_deref())
        .map_err(|error| error.admission_context(AdmissionError::Invalid))?.to_owned();
    let config = state.model.config();
    let sampling = request::sampling_parameters(&parameters.sampling, &config.sampling_defaults, &config.sampling_ranges)
        .map_err(|error| error.admission_context(AdmissionError::Invalid))?;
    let permit = state.frontend_slots.clone().acquire_owned().await
        .map_err(|_| Error::admission(AdmissionError::Unavailable, "native frontend is shutting down"))?;
    let preparation_state = state.clone();
    let preparation_ingress = ingress.clone();
    let preparation_cancelled = cancelled.clone();
    let preparation_parameters = parameters.clone();
    let (pcm, ranges, rendered) = tokio::task::spawn_blocking(move || {
        let _permit = permit;
        let _ingress = preparation_ingress;
        let processor = preparation_state.audio.as_ref().ok_or("audio processor is unavailable")?;
        let renderer = preparation_state.transcription.as_ref().ok_or("transcription syntax is unavailable")?;
        let rendered = renderer.render(&preparation_state.frontend, &preparation_parameters.context,
            preparation_parameters.language.as_deref(), &preparation_cancelled)?;
        let pcm = processor.decode(&bytes, &preparation_cancelled)?;
        let ranges = processor.transcription_ranges(&pcm, &preparation_cancelled)?;
        Ok::<_, Error>((Arc::new(pcm), ranges, Arc::new(rendered)))
    }).await.map_err(|error| Error::from(format!("audio decode worker failed: {error}")))?
        .map_err(|error| error.admission_context(AdmissionError::Invalid))?;
    let mut children: Vec<Option<Child<'_>>> = (0..ranges.len()).map(|_| None).collect();
    let mut output: Vec<Option<String>> = (0..ranges.len()).map(|_| None).collect();
    let mut filling: Option<BoxFuture<'static, Result<(usize, Generation)>>> = None;
    let (mut next, mut complete, mut cursor) = (0usize, 0usize, 0usize);
    let mut pending: Option<usize> = None;
    let (mut generated, mut truncated) = (0u64, false);
    while complete < ranges.len() {
        if filling.is_none() && pending.is_none() && next < ranges.len() {
            filling = Some(prepare_child(state.clone(), ingress.clone(), pcm.clone(), ranges[next].clone(),
                rendered.clone(), parameters.clone(), sampling.clone(), cancelled.clone(), next).boxed());
            next += 1;
        }
        let has_filling = filling.is_some();
        let has_children = children.iter().any(Option::is_some);
        let progress = tokio::select! {
            result = async { filling.as_mut().unwrap().await }, if has_filling => Progress::Prepared(result),
            progress = std::future::poll_fn(|cx| {
                if let Some(index) = pending {
                    let child: &mut Child<'_> = children[index].as_mut().unwrap();
                    if let Poll::Ready(result) = child.generation.poll_admission(cx) {
                        return Poll::Ready(Progress::Admitted(index, result));
                    }
                }
                for offset in 0..children.len() {
                    let index = (cursor + offset) % children.len();
                    if let Some(child) = &mut children[index] {
                        match child.generation.events.poll_recv(cx) {
                            Poll::Ready(Some(event)) => return Poll::Ready(Progress::Event(index, event)),
                            Poll::Ready(None) => return Poll::Ready(Progress::Event(index,
                                GenerationEvent::Error(child.generation.failure()))),
                            Poll::Pending => {}
                        }
                    }
                }
                Poll::Pending
            }), if has_children => progress,
        };
        match progress {
            Progress::Prepared(result) => {
                let (index, generation) = result?;
                filling = None;
                children[index] = Some(Child { generation,
                    decoder: state.frontend.incremental_decoder(true), raw: String::new() });
                pending = Some(index);
            }
            Progress::Admitted(index, result) => {
                result?;
                debug_assert_eq!(pending, Some(index));
                pending = None;
            }
            Progress::Event(index, event) => {
                cursor = (index + 1) % children.len();
                match event {
                    GenerationEvent::Token(token) => {
                        let child = children[index].as_mut().unwrap();
                        if let Some(text) = child.decoder.push(&[token])? { child.raw.push_str(&text); }
                    }
                    GenerationEvent::Finished { generated_tokens, termination, .. } => {
                        if pending == Some(index) { return Err("transcription finished before admission".into()); }
                        let mut child = children[index].take().unwrap();
                        child.raw.push_str(&child.decoder.finish()?);
                        output[index] = Some(renderer.clean_output(&child.raw).to_owned());
                        generated += u64::from(generated_tokens);
                        truncated |= termination == Termination::OutputLimit;
                        complete += 1;
                        // Dropping the reader returns only its host credit. GPU
                        // inputs/features/KV remain owned through native retirement.
                    }
                    GenerationEvent::AdmissionRejected(error) | GenerationEvent::Error(error) => return Err(error),
                    _ => return Err("unexpected transcription generation event".into()),
                }
            }
        }
    }
    Ok((output.into_iter().map(Option::unwrap).collect::<Vec<_>>().join(&separator), pcm.len(), generated, truncated))
}
