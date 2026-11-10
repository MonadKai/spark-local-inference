/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-FileCopyrightText: Copyright (c) 2022 Samuel Colvin
 * SPDX-License-Identifier: Apache-2.0 AND MIT */

//! Normalize declared request seed syntax once, before protocol deserialization.
//!
//! Text cleanup follows pydantic-core 2.46.5's input/shared.rs, distributed
//! under MIT; see helios/licenses/Pydantic.txt. No Python runs in this path.

use crate::{Result, model::contract::RequestSeedSyntax};
use serde_json::{Map, Value};

fn signed_text(text: &str) -> Option<i64> {
    if let Ok(value) = text.parse() { return Some(value); }
    let mut text = text.trim();
    if let Some(rest) = text.strip_prefix('+') {
        if rest.starts_with('-') { return None; }
        text = rest;
    }
    let negative = text.starts_with('-');
    if negative {
        text = &text[1..];
        if text.starts_with(['-', '+']) { return None; }
    }
    if text.starts_with('0') {
        let mut begin = text.len().checked_sub(1)?;
        for (index, byte) in text.bytes().enumerate().skip(1) {
            match byte {
                b'0' | b'_' => {}
                b'1'..=b'9' | b'-' => { begin = index; break; }
                b'.' => { begin = index - 1; break; }
                _ => return None,
            }
        }
        text = &text[begin..];
    } else if !text.starts_with(|ch: char| ch.is_ascii_digit() || ch == '-') {
        return None;
    }
    if let Some((integer, fraction)) = text.split_once('.') {
        if !fraction.is_empty() && fraction.bytes().all(|byte| byte == b'0') { text = integer; }
    }
    if text.starts_with('_') || text.ends_with('_') || text.contains("__") { return None; }
    let mut normalized = String::new();
    if negative { normalized.push('-'); }
    normalized.extend(text.chars().filter(|&ch| ch != '_'));
    normalized.parse().ok()
}

fn signed(value: &Value) -> Option<i64> {
    match value {
        Value::Bool(value) => Some(i64::from(*value)),
        Value::String(text) => signed_text(text),
        Value::Number(number) if number.is_i64() => number.as_i64(),
        Value::Number(number) if number.is_f64() => {
            let value = number.as_f64()?;
            // The actual pinned validator uses strict float bounds. Integer
            // i64::MIN remains valid; the rounded f64 boundary does not.
            (value.is_finite() && value.fract() == 0. && value > i64::MIN as f64
                && value < i64::MAX as f64).then_some(value as i64)
        }
        _ => None,
    }
}

fn request_key(value: Option<&Value>) -> Result<Option<u64>> {
    let Some(value) = value.filter(|value| !value.is_null()) else { return Ok(None); };
    let value = signed(value).ok_or("seed must be a signed 64-bit integer or null")?;
    Ok((value != -1).then_some(value as u64))
}

pub(super) fn normalize(fields: &mut Map<String, Value>, syntax: Option<RequestSeedSyntax>) -> Result<()> {
    let Some(RequestSeedSyntax::Signed64Optional) = syntax else { return Ok(()); };
    let seed = request_key(fields.get("seed"))?.unwrap_or_else(|| {
        use std::hash::BuildHasher;
        std::collections::hash_map::RandomState::new().hash_one(0u8)
    });
    // The prepared/native ABI remains uint64. Each parent normalizes once;
    // child requests receive this same key, rather than generating another.
    fields.insert("seed".into(), Value::from(seed));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn signed_seeds_preserve_bits_and_unseeded_inputs_remain_distinct_from_zero() {
        assert_eq!(request_key(None).unwrap(), None);
        assert_eq!(request_key(Some(&Value::Null)).unwrap(), None);
        assert_eq!(request_key(Some(&json!(-1))).unwrap(), None);
        assert_eq!(request_key(Some(&json!(0))).unwrap(), Some(0));
        assert_eq!(request_key(Some(&json!(-2))).unwrap(), Some(u64::MAX - 1));
        assert_eq!(request_key(Some(&json!(i64::MIN))).unwrap(), Some(1 << 63));
        assert_eq!(request_key(Some(&json!(i64::MAX))).unwrap(), Some(i64::MAX as u64));
        for value in [json!(u64::MAX), json!(1.5), json!(-9223372036854775808_f64), json!("null")] {
            assert!(request_key(Some(&value)).is_err(), "{value}");
        }
    }

    #[test]
    fn absent_seed_syntax_preserves_legacy_unsigned_defaults_and_validation() {
        for value in [json!({}), json!({"seed":null}), json!({"seed":-1}), json!({"seed":u64::MAX})] {
            let mut fields = value.as_object().unwrap().clone();
            normalize(&mut fields, None).unwrap();
            assert_eq!(Value::Object(fields), value);
        }
    }

    #[cfg(feature = "runtime-package")]
    #[test]
    fn seed_syntax_is_additive_and_unknown_values_fail_before_requests() {
        use prost::Message;
        use crate::{runtime_package::wire, model::contract::FrontendContract};
        let old = wire::FrontendContract::decode(&[][..]).unwrap();
        assert_eq!(old.request_seed_syntax, None);
        let mut old_json = serde_json::to_value(old).unwrap();
        old_json.as_object_mut().unwrap().remove("request_seed_syntax");
        let legacy: FrontendContract = serde_json::from_value(old_json).unwrap();
        assert_eq!(legacy.request_seed_syntax, None);
        let encoded = wire::FrontendContract::decode(&[0x70, 0x01][..]).unwrap();
        let syntax: FrontendContract = serde_json::from_value(serde_json::to_value(encoded).unwrap()).unwrap();
        assert_eq!(syntax.request_seed_syntax, Some(RequestSeedSyntax::Signed64Optional));
        for value in [0, 2, 127] {
            let unknown = wire::FrontendContract::decode(&[0x70, value][..]).unwrap();
            assert!(serde_json::from_value::<FrontendContract>(serde_json::to_value(unknown).unwrap()).is_err());
        }
    }

    #[test]
    fn declared_seed_normalization_matches_independent_pinned_request_observations() {
        // This fixture contains request-model observations, not reimplemented
        // Pydantic expectations; its generator runs inside the pinned image.
        let reference: Value = serde_json::from_str(include_str!("../../../tests/fixtures/asr-seed-reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let mut fields = case["options"].as_object().unwrap().clone();
            let outcome = normalize(&mut fields, Some(RequestSeedSyntax::Signed64Optional));
            assert_eq!(outcome.is_ok(), case["request_valid"].as_bool().unwrap(), "{case}");
            if outcome.is_ok() && !case["seed"].is_null() {
                assert_eq!(fields["seed"].as_u64().unwrap(), case["seed"].as_i64().unwrap() as u64, "{case}");
            }
        }
    }
}
