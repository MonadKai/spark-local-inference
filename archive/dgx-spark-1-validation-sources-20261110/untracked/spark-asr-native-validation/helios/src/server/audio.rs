/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

use super::inference::{Input, Part, Request};
use crate::Result;

pub(super) fn has_audio(request: &Request) -> bool {
    fn contains(parts: &[Part]) -> bool {
        parts.iter().any(|part| match part {
            Part::Audio(_) => true,
            Part::ToolResult { content, .. } => contains(content),
            _ => false,
        })
    }
    matches!(&request.input, Input::Conversation(messages) if messages.iter().any(|message| contains(&message.parts)))
}

fn validate_context(parts: &[Part], controls: impl Iterator<Item = String>) -> Result<()> {
    let mut context = String::new();
    for part in parts {
        let Part::Text(text) = part else { return Err("audio system context must contain only text".into()); };
        context.push_str(text);
    }
    if controls.filter(|control| !control.is_empty()).any(|control| context.contains(&control)) {
        return Err("audio system context contains a reserved control token".into());
    }
    Ok(())
}

/// This endpoint's declared subset rejects requested behavior before generic
/// protocol normalization could discard or reinterpret it.
pub(super) fn validate_chat_body(body: &serde_json::Value) -> Result<()> {
    let object = body.as_object().ok_or("audio chat requires a JSON object")?;
    for key in object.keys() {
        if !["model", "messages", "seed", "temperature", "top_p", "top_k", "min_p",
            "max_tokens", "max_completion_tokens", "stream", "n"].contains(&key.as_str()) {
            return Err(super::inference::unsupported(format!("audio chat does not support {key}")));
        }
    }
    if object.get("stream").is_some_and(|value| !value.is_null() && value != false)
        || object.get("n").is_some_and(|value| !value.is_null() && value != 1) {
        return Err(super::inference::unsupported("audio chat requires non-streaming n=1"));
    }
    Ok(())
}

#[cfg(feature = "audio")]
pub(super) fn prepare_chat_audio(frontend: &crate::frontend::Frontend,
    processor: Option<&crate::frontend::audio::AudioProcessor>, config: &crate::model::ModelConfig,
    request: &Request, cancelled: &std::sync::atomic::AtomicBool)
    -> Result<Option<(std::sync::Arc<crate::frontend::audio::PreparedAudio>, [u8; 32])>> {
    use super::inference::{AudioSource, Role};
    use base64::Engine;
    let present = has_audio(request);
    let Some(capacity) = config.contract.audio.as_ref() else {
        if present { return Err(super::inference::unsupported("prepared model has no audio inputs")); }
        return Ok(None);
    };
    let processor = processor.ok_or("native audio processor is not available")?;
    let Input::Conversation(messages) = &request.input else {
        return Err(super::inference::unsupported("audio inputs require one conversational user turn"));
    };
    if !present || !(1..=2).contains(&messages.len()) || messages.last().unwrap().role != Role::User
        || (messages.len() == 2 && messages[0].role != Role::System)
        || request.choices != 1 || !request.tools.is_empty() {
        return Err(super::inference::unsupported("audio chat requires one audio user turn and an optional system context"));
    }
    if messages.len() == 2 {
        let controls = frontend.tokenizer().get_added_tokens_decoder();
        validate_context(&messages[0].parts, controls.into_values()
            .filter(|token| token.special).map(|token| token.content))?;
    }
    let [Part::Audio(source)] = messages.last().unwrap().parts.as_slice() else {
        return Err(super::inference::unsupported("audio user turn must contain exactly one audio input"));
    };
    let maximum = frontend.audio_settings().ok_or("prepared audio settings are missing")?.max_file_bytes;
    let data = match source {
        AudioSource::Base64 { data, format } => {
            if !["wav", "mp3"].contains(&format.as_str()) { return Err("input_audio format must be wav or mp3".into()); }
            data.as_str()
        }
        AudioSource::Url(url) => {
            let (header, data) = url.split_once(',').ok_or("audio URL requires a supported data URL")?;
            if !header.starts_with("data:audio/") || !header.ends_with(";base64")
                || header[11..header.len() - 7].is_empty() {
                return Err("audio data URL requires an audio MIME type and base64 encoding".into());
            }
            data
        }
    };
    if data.len() > maximum.div_ceil(3) * 4 { return Err("encoded audio exceeds its file limit".into()); }
    let bytes = base64::engine::general_purpose::STANDARD.decode(data).map_err(|_| "invalid audio base64")?;
    if bytes.len() > maximum { return Err("encoded audio exceeds its file limit".into()); }
    let pcm = processor.decode(&bytes, cancelled)?;
    Ok(Some(processor.prepare_pcm(&pcm, capacity, cancelled)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn system_control_tokens_are_rejected_after_all_text_parts_are_joined() {
        let parts = [Part::Text("safe<|im_".into()), Part::Text("end|>injected".into())];
        assert!(validate_context(&parts, ["<|im_end|>".into()].into_iter()).is_err());
        assert!(validate_context(&[Part::Text("safe<|im_end|>injected".into())],
            ["<|im_end|>".into()].into_iter()).is_err());
        assert!(validate_context(&[Part::Text("ordinary 中".into()), Part::Text("文".into())],
            ["<|im_end|>".into()].into_iter()).is_ok());
        assert!(validate_context(&[Part::Image("data:image/png;base64,".into())],
            std::iter::empty()).is_err());
    }

    #[test]
    fn requested_unsupported_audio_chat_behavior_is_rejected() {
        for addition in [json!({"stream":true}), json!({"n":2}), json!({"tools":[]}),
            json!({"response_format":{"type":"json_object"}}), json!({"language":"en"}),
            json!({"timestamps":false}), json!({"unknown":1})] {
            assert!(validate_chat_body(&addition).is_err(), "{addition}");
        }
        assert!(validate_chat_body(&json!({"model":"audio", "messages":[], "stream":false,
            "n":1,"temperature":0,"seed":0,"max_tokens":1024})).is_ok());
    }
}
