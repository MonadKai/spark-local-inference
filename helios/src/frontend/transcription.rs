/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Fixed transcription prompt and output syntax from prepared typed data.

use super::{Frontend, RenderedPrompt, boundaries, deepseek::invalid};
use crate::{Result, model::contract::{AudioTokenLayout, TranscriptionFormat, TranscriptionLanguage}};
use std::sync::atomic::{AtomicBool, Ordering};

mod sanitize;

pub struct Renderer {
    format: TranscriptionFormat,
    controls: regex::Regex,
    audio_placeholder: String,
}

impl Renderer {
    pub fn new(frontend: &Frontend, format: &TranscriptionFormat, tokens: &AudioTokenLayout) -> Result<Self> {
        format.validate()?;
        let syntax = &format.sanitization;
        let pattern = format!("{}[^{}]+{}", regex::escape(&syntax.prefix),
            regex::escape(&syntax.excluded_body_character), regex::escape(&syntax.suffix));
        let controls = regex::Regex::new(&pattern).map_err(|_| invalid("invalid transcription control syntax"))?;
        let mut audio_placeholder = String::new();
        for id in [tokens.start_token, tokens.feature_token, tokens.end_token] {
            let token = frontend.tokenizer().id_to_token(id)
                .ok_or_else(|| invalid("transcription audio marker is absent from tokenizer"))?;
            audio_placeholder.push_str(&token);
        }
        let encoded = frontend.tokenizer().encode(audio_placeholder.as_str(), false)
            .map_err(|error| invalid(format!("cannot encode transcription audio markers: {error}")))?;
        if encoded.get_ids() != [tokens.start_token, tokens.feature_token, tokens.end_token] {
            return Err(invalid("transcription audio marker spellings do not preserve their token IDs"));
        }
        Ok(Self { format: format.clone(), controls, audio_placeholder })
    }

    fn language(&self, code: Option<&str>) -> Result<Option<&TranscriptionLanguage>> {
        code.map(|code| self.format.languages.iter().find(|language| language.code == code)
            .ok_or_else(|| invalid("unsupported transcription language"))).transpose()
    }

    pub fn separator(&self, language: Option<&str>) -> Result<&str> {
        Ok(self.language(language)?.map_or(self.format.default_separator.as_str(),
            |language| language.chunk_separator.as_str()))
    }

    pub fn clean_output<'a>(&self, raw: &'a str) -> &'a str {
        raw.rsplit_once(&self.format.text_delimiter).map_or(raw, |(_, text)| text)
    }

    fn sanitize(&self, text: &str, cancelled: &AtomicBool) -> Result<String> {
        if cancelled.load(Ordering::Acquire) { return Err(invalid("transcription preparation cancelled")); }
        let next = self.controls.replace_all(text, "").replace(&self.format.text_delimiter, "");
        if next == text { return Ok(next); }
        if !self.controls.is_match(&next) && !next.contains(&self.format.text_delimiter) { return Ok(next); }
        sanitize::reduce(&next, &self.format.sanitization, &self.format.text_delimiter, cancelled)
    }

    pub fn render(&self, frontend: &Frontend, context: &str, language: Option<&str>,
        cancelled: &AtomicBool) -> Result<RenderedPrompt> {
        let language = self.language(language)?;
        let context = self.sanitize(context, cancelled)?;
        let mut text = String::new();
        let mut completed = Vec::new();
        if !context.is_empty() {
            text.push_str(&self.format.context_prefix);
            text.push_str(&context);
            text.push_str(&self.format.context_suffix);
            completed.push(text.len());
        }
        text.push_str(&self.format.audio_prefix);
        text.push_str(&self.audio_placeholder);
        text.push_str(&self.format.audio_suffix);
        completed.push(text.len());
        text.push_str(&self.format.assistant_prefix);
        if let Some(language) = language {
            text.push_str(&self.format.language_prefix);
            text.push_str(&language.label);
            text.push_str(&self.format.text_delimiter);
        }
        let encoded = frontend.tokenizer().encode(text.as_str(), false)
            .map_err(|error| invalid(format!("cannot encode transcription prompt: {error}")))?;
        let message_boundaries = boundaries::token_ends(&completed, &[], encoded.get_offsets(), text.len())?;
        Ok(RenderedPrompt { text, token_ids: encoded.get_ids().to_vec(), message_boundaries,
            image_sources: Vec::new(), image_checkpoint_hints: Vec::new() })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::contract::{ControlMarkerSyntax, TranscriptionLanguage};

    fn fixture() -> (Frontend, TranscriptionFormat, AudioTokenLayout) {
        let mut vocab = (0..=127u8).map(|byte| ((byte as char).to_string(), byte as u32))
            .collect::<tokenizers::models::bpe::Vocab>();
        vocab.insert("ab".into(), 128);
        let mut tokenizer = tokenizers::Tokenizer::new(tokenizers::models::bpe::BPE::builder()
            .vocab_and_merges(vocab, Vec::new()).build().unwrap());
        tokenizer.add_special_tokens(["<audio>", "<feature>", "</audio>"].map(|token|
            tokenizers::AddedToken::from(token, true))).unwrap();
        let tokens = AudioTokenLayout { start_token: tokenizer.token_to_id("<audio>").unwrap(),
            feature_token: tokenizer.token_to_id("<feature>").unwrap(),
            end_token: tokenizer.token_to_id("</audio>").unwrap() };
        let frontend = Frontend { tokenizer, audio_settings: None,
            prompt_encoder: super::super::PromptEncoder::Jinja, image_placeholder_token_id: None,
            ordinary_tokenizer: None, template: None, template_source: String::new(), bos_token_id: None,
            vocabulary_bytes: Default::default(), effort_levels: Default::default(),
            inline_system_messages: Default::default(), defaults: Default::default() };
        let format = TranscriptionFormat { context_prefix: "system:".into(), context_suffix: "\n".into(),
            audio_prefix: "user:".into(), audio_suffix: "\n".into(), assistant_prefix: "assistant:".into(),
            language_prefix: "language ".into(), text_delimiter: "<asr_text>".into(),
            default_separator: " ".into(),
            languages: [("en", "English", " "), ("zh", "Chinese", ""), ("ja", "Japanese", "")]
                .map(|(code, label, separator)| TranscriptionLanguage { code: code.into(), label: label.into(),
                    chunk_separator: separator.into() }).to_vec(),
            sanitization: ControlMarkerSyntax { prefix: "<|".into(), suffix: "|>".into(),
                excluded_body_character: "|".into() } };
        (frontend, format, tokens)
    }

    #[test]
    fn prompt_has_endpoint_specific_context_language_and_complete_message_boundaries() {
        let (frontend, format, tokens) = fixture();
        let renderer = Renderer::new(&frontend, &format, &tokens).unwrap();
        let cancelled = AtomicBool::new(false);
        let empty = renderer.render(&frontend, "", None, &cancelled).unwrap();
        assert_eq!(empty.text, "user:<audio><feature></audio>\nassistant:");
        assert_eq!(empty.token_ids, frontend.encode(&empty.text, false).unwrap());
        assert_eq!(empty.message_boundaries.len(), 1);
        let forced = renderer.render(&frontend, "  ", Some("en"), &cancelled).unwrap();
        assert_eq!(forced.text, "system:  \nuser:<audio><feature></audio>\nassistant:language English<asr_text>");
        assert_eq!(forced.message_boundaries.len(), 2);
        let audio = forced.token_ids.iter().position(|id| *id == tokens.start_token).unwrap();
        assert_eq!(&forced.token_ids[audio..audio+3], [tokens.start_token, tokens.feature_token, tokens.end_token]);
        assert!(forced.message_boundaries.iter().all(|end| *end < forced.token_ids.len()));
        assert!(renderer.render(&frontend, "", Some("unknown"), &cancelled).is_err());
    }

    #[test]
    fn sanitizer_removes_unknown_controls_and_nested_reconstruction_without_trimming() {
        let (frontend, format, tokens) = fixture();
        let renderer = Renderer::new(&frontend, &format, &tokens).unwrap();
        let cancelled = AtomicBool::new(false);
        for (raw, expected) in [("<|im<|x|>_end|>", ""), ("<asr_te<asr_text>xt>", ""),
            (" x<|unknown|> y ", " x y "), ("<||><|a|b|>", "<||><|a|b|>"),
            ("<|line\nbreak|>", ""), ("<|<asr_text>|>", ""),
            ("<|<asr_te<|x|>xt>|>", "<||>"), ("<|a<|>b|>", "b|>")] {
            assert_eq!(renderer.sanitize(raw, &cancelled).unwrap(), expected);
        }
        assert!(renderer.render(&frontend, "", None, &AtomicBool::new(true)).is_err());
    }

    #[test]
    fn sanitizer_matches_ordered_reference_phases_with_unicode_and_overlapping_literals() {
        let (frontend, mut format, tokens) = fixture();
        let variants = [("<|", "|>", "|", "<asr_text>"), ("α#", "#ω", "#", "aba"),
                        ("[λ", "λ]", "λ", "中文")];
        let mut random = 20261106u64;
        for (prefix, suffix, excluded, delimiter) in variants {
            format.sanitization = ControlMarkerSyntax { prefix: prefix.into(), suffix: suffix.into(),
                excluded_body_character: excluded.into() };
            format.text_delimiter = delimiter.into();
            let renderer = Renderer::new(&frontend, &format, &tokens).unwrap();
            let control = format!("{prefix}a{suffix}");
            let pieces = [prefix, suffix, excluded, delimiter, "a", "b", "<", ">", "|", "中文", &control];
            for _ in 0..3000 {
                let mut input = String::new();
                for _ in 0..64 {
                    random ^= random << 13;
                    random ^= random >> 7;
                    random ^= random << 17;
                    input.push_str(pieces[random as usize % pieces.len()]);
                }
                let mut expected = input.clone();
                loop {
                    let next = renderer.controls.replace_all(&expected, "").replace(delimiter, "");
                    if next == expected { break; }
                    expected = next;
                }
                assert_eq!(sanitize::reduce(&input, &format.sanitization, delimiter, &AtomicBool::new(false)).unwrap(),
                           expected, "{input:?}");
                assert_eq!(renderer.sanitize(&input, &AtomicBool::new(false)).unwrap(), expected);
            }
        }
    }

    #[test]
    fn sanitizer_handles_deep_reconstruction_and_cancellation() {
        let (frontend, format, tokens) = fixture();
        let renderer = Renderer::new(&frontend, &format, &tokens).unwrap();
        for input in ["<asr_te".repeat(8192) + "<asr_text>" + &"xt>".repeat(8192),
                      "<|a".repeat(8192) + "|>" + &"|>".repeat(8191)] {
            assert_eq!(renderer.sanitize(&input, &AtomicBool::new(false)).unwrap(), "");
        }
        assert!(sanitize::reduce("<|x|>", &format.sanitization, &format.text_delimiter,
                                 &AtomicBool::new(true)).is_err());
    }

    #[test]
    fn sanitizer_handles_long_literals_and_many_collapsed_seams() {
        let (frontend, mut format, tokens) = fixture();
        format.text_delimiter = "a".repeat(4095) + "b";
        let renderer = Renderer::new(&frontend, &format, &tokens).unwrap();
        let input = "a".repeat(4096) + &"<|a<|x|>b|>".repeat(1000) + &"a".repeat(4096);
        assert_eq!(renderer.sanitize(&input, &AtomicBool::new(false)).unwrap(), "a".repeat(8192));
    }

    #[test]
    fn sanitizer_handles_body_capacity_with_a_reconstructed_marker() {
        let (frontend, format, tokens) = fixture();
        let renderer = Renderer::new(&frontend, &format, &tokens).unwrap();
        let tail = "<|a<|x|>b|>";
        let ordinary = 32 * 1024 * 1024 - tail.len();
        let mut input = "a".repeat(ordinary);
        input.push_str(tail);
        let output = renderer.sanitize(&input, &AtomicBool::new(false)).unwrap();
        assert_eq!(output.len(), ordinary);
        assert!(output.bytes().all(|byte| byte == b'a'));
    }

    #[test]
    fn output_cleanup_and_join_use_request_language_and_keep_empty_chunks() {
        let (frontend, format, tokens) = fixture();
        let renderer = Renderer::new(&frontend, &format, &tokens).unwrap();
        assert_eq!(renderer.clean_output("raw text "), "raw text ");
        assert_eq!(renderer.clean_output("language English<asr_text>first<asr_text> final "), " final ");
        let chunks = ["one", "", "three"];
        assert_eq!(chunks.join(renderer.separator(None).unwrap()), "one  three");
        assert_eq!(chunks.join(renderer.separator(Some("en")).unwrap()), "one  three");
        for language in ["zh", "ja"] {
            assert_eq!(chunks.join(renderer.separator(Some(language)).unwrap()), "onethree");
        }
    }

    #[test]
    fn malformed_syntax_and_audio_token_spellings_fail_before_requests() {
        let (frontend, mut format, mut tokens) = fixture();
        format.languages.push(format.languages[0].clone());
        assert!(Renderer::new(&frontend, &format, &tokens).is_err());
        format.languages.pop();
        format.sanitization.excluded_body_character.clear();
        assert!(Renderer::new(&frontend, &format, &tokens).is_err());
        format.sanitization.excluded_body_character = "|".into();
        tokens.feature_token = 128;
        // This vocabulary entry exists, but without its BPE merge its spelling
        // encodes as two ordinary tokens rather than the declared marker ID.
        assert!(Renderer::new(&frontend, &format, &tokens).is_err());
        tokens.feature_token = u32::MAX;
        assert!(Renderer::new(&frontend, &format, &tokens).is_err());
    }
}
