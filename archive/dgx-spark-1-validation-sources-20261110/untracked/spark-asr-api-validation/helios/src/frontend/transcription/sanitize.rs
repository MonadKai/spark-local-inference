/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! Ordered, simultaneous deletion phases without repeatedly copying the input.

use super::*;
use std::collections::{BTreeSet, VecDeque};

const REMOVED: u32 = 1 << 31;
const END: u32 = REMOVED - 1;

struct Text {
    characters: Vec<char>,
    previous: Vec<u32>,
    next: Vec<u32>,
    separators: BTreeSet<u32>,
    head: u32,
}

impl Text {
    fn new(text: &str, separator: char, cancelled: &AtomicBool) -> Result<Self> {
        let mut characters = Vec::new();
        let mut separators = BTreeSet::new();
        let (mut previous, mut next) = (Vec::new(), Vec::new());
        for (index, ch) in text.chars().enumerate() {
            check_cancelled(index, cancelled)?;
            if index >= END as usize { return Err(invalid("transcription context is too large")); }
            if ch == separator { separators.insert(index as u32); }
            characters.push(ch);
            previous.push(if index == 0 { END } else { index as u32 - 1 });
            next.push(END);
            if index > 0 { next[index - 1] = index as u32; }
        }
        let count = characters.len() as u32;
        Ok(Self { previous, next, separators,
            head: if count == 0 { END } else { 0 }, characters })
    }

    fn live(&self, index: u32) -> bool { index != END && self.next[index as usize] & REMOVED == 0 }
    fn after(&self, index: u32) -> u32 { self.next[index as usize] & !REMOVED }

    fn pattern(&self, start: u32, pattern: &[char]) -> Option<(u32, u32)> {
        let (mut at, mut last) = (start, start);
        for ch in pattern {
            if !self.live(at) || self.characters[at as usize] != *ch { return None; }
            last = at;
            at = self.after(at);
        }
        Some((start, last))
    }

    fn control(&self, separator: u32, prefix: &[char], suffix: &[char]) -> Option<(u32, u32)> {
        if !self.live(separator) { return None; }
        let mut start = separator;
        for (offset, ch) in prefix.iter().rev().enumerate() {
            if !self.live(start) || self.characters[start as usize] != *ch { return None; }
            if offset + 1 < prefix.len() { start = self.previous[start as usize]; }
        }
        let right = *self.separators.range((std::ops::Bound::Excluded(separator), std::ops::Bound::Unbounded)).next()?;
        if self.after(separator) == right { return None; } // The body must be nonempty.
        let (_, last) = self.pattern(right, suffix)?;
        Some((start, last))
    }

    fn resolve(&mut self, index: u32, forward: bool, cancelled: &AtomicBool) -> Result<u32> {
        let (mut at, mut count) = (index, 0);
        while at != END && !self.live(at) {
            check_cancelled(count, cancelled)?;
            at = if forward { self.after(at) } else { self.previous[at as usize] };
            count += 1;
        }
        let mut cursor = index;
        while cursor != END && !self.live(cursor) {
            check_cancelled(count, cancelled)?;
            let next = if forward { self.after(cursor) } else { self.previous[cursor as usize] };
            if forward { self.next[cursor as usize] = at | REMOVED; }
            else { self.previous[cursor as usize] = at; }
            cursor = next;
            count += 1;
        }
        Ok(at)
    }

    fn delete(&mut self, first: u32, last: u32, cancelled: &AtomicBool) -> Result<(u32, u32)> {
        let left = self.previous[first as usize];
        let right = self.after(last);
        let (mut at, mut deleted) = (first, 0usize);
        while at != right {
            if deleted % 1024 == 0 && cancelled.load(Ordering::Acquire) {
                return Err(invalid("transcription preparation cancelled"));
            }
            let next = self.after(at);
            self.next[at as usize] |= REMOVED;
            self.separators.remove(&at);
            at = next;
            deleted += 1;
        }
        if left == END { self.head = right; } else { self.next[left as usize] = right; }
        if right != END { self.previous[right as usize] = left; }
        Ok((left, right))
    }

    fn literal_matches(&self, indices: impl Iterator<Item = u32>, literal: &[char], failure: &[usize],
        matches: &mut BTreeSet<u32>, cancelled: &AtomicBool) -> Result<()> {
        let mut recent = VecDeque::with_capacity(literal.len());
        let mut matched = 0;
        for (count, index) in indices.enumerate() {
            check_cancelled(count, cancelled)?;
            recent.push_back(index);
            if recent.len() > literal.len() { recent.pop_front(); }
            let ch = self.characters[index as usize];
            while matched > 0 && literal[matched] != ch { matched = failure[matched - 1]; }
            if literal[matched] == ch { matched += 1; }
            if matched == literal.len() {
                matches.insert(*recent.front().unwrap());
                matched = failure[matched - 1];
            }
        }
        Ok(())
    }

    fn seams(&mut self, seams: Vec<(u32, u32)>, prefix: &[char], suffix: &[char], literal: &[char],
        failure: &[usize], controls: &mut BTreeSet<u32>, literals: &mut BTreeSet<u32>,
        cancelled: &AtomicBool) -> Result<()> {
        let mut unique = BTreeSet::new();
        for (count, (left, right)) in seams.into_iter().enumerate() {
            check_cancelled(count, cancelled)?;
            let left = self.resolve(left, false, cancelled)?;
            let right = self.resolve(right, true, cancelled)?;
            unique.insert((left, right));
        }
        for (count, (left, right)) in unique.into_iter().enumerate() {
            check_cancelled(count, cancelled)?;
            let mut window = Vec::new();
            let mut at = left;
            // A new literal must cross a deletion seam. Its start is at most
            // literal.len()-1 surviving characters before that seam.
            for _ in 1..literal.len() {
                if at == END { break; }
                window.push(at);
                at = self.previous[at as usize];
            }
            window.reverse();
            at = right;
            for _ in 1..literal.len() {
                if at == END { break; }
                window.push(at);
                at = self.after(at);
            }
            self.literal_matches(window.into_iter(), literal, failure, literals, cancelled)?;
            // Prefix and suffix contain the separator only at their facing
            // ends. Only the adjacent separator pair and its two neighbours
            // can acquire a new control match after a splice.
            let before = if left == END { None } else { self.separators.range(..=left).next_back().copied() };
            let after = if right == END { None } else { self.separators.range(right..).next().copied() };
            let previous = before.and_then(|index| self.separators.range(..index).next_back().copied());
            for separator in [previous, before, after].into_iter().flatten() {
                if let Some((start, _)) = self.control(separator, prefix, suffix) { controls.insert(start); }
            }
        }
        Ok(())
    }
}

fn check_cancelled(count: usize, cancelled: &AtomicBool) -> Result<()> {
    if count % 1024 == 0 && cancelled.load(Ordering::Acquire) {
        return Err(invalid("transcription preparation cancelled"));
    }
    Ok(())
}

pub(super) fn reduce(text: &str, syntax: &crate::model::contract::ControlMarkerSyntax,
    delimiter: &str, cancelled: &AtomicBool) -> Result<String> {
    let prefix: Vec<_> = syntax.prefix.chars().collect();
    let suffix: Vec<_> = syntax.suffix.chars().collect();
    let literal: Vec<_> = delimiter.chars().collect();
    let separator = syntax.excluded_body_character.chars().next().unwrap();
    if cancelled.load(Ordering::Acquire) { return Err(invalid("transcription preparation cancelled")); }
    let mut failure = vec![0; literal.len()];
    for index in 1..literal.len() {
        let mut matched = failure[index - 1];
        while matched > 0 && literal[matched] != literal[index] { matched = failure[matched - 1]; }
        if literal[matched] == literal[index] { matched += 1; }
        failure[index] = matched;
    }
    let mut text = Text::new(text, separator, cancelled)?;
    let mut controls = BTreeSet::new();
    for (count, &index) in text.separators.iter().enumerate() {
        check_cancelled(count, cancelled)?;
        if let Some((start, _)) = text.control(index, &prefix, &suffix) { controls.insert(start); }
    }
    let mut literals = BTreeSet::new();
    text.literal_matches(0..text.characters.len() as u32, &literal, &failure, &mut literals, cancelled)?;
    while !controls.is_empty() || !literals.is_empty() {
        if cancelled.load(Ordering::Acquire) { return Err(invalid("transcription preparation cancelled")); }
        let (mut next_controls, mut next_literals) = (BTreeSet::new(), BTreeSet::new());
        for control_phase in [true, false] {
            let candidates = if control_phase { std::mem::take(&mut controls) } else { std::mem::take(&mut literals) };
            let mut spans = Vec::new();
            let mut after_previous = None;
            // Freeze complete match spans before deleting anything in this
            // phase. Newly formed same-phase matches belong to the next round.
            for (index, first) in candidates.into_iter().enumerate() {
                if index % 1024 == 0 && cancelled.load(Ordering::Acquire) {
                    return Err(invalid("transcription preparation cancelled"));
                }
                if after_previous.is_some_and(|last| first <= last) { continue; }
                let span = if control_phase {
                    text.pattern(first, &prefix).and_then(|(_, separator)| text.control(separator, &prefix, &suffix))
                } else { text.pattern(first, &literal) };
                if let Some(span) = span { after_previous = Some(span.1); spans.push(span); }
            }
            let seams = spans.into_iter().map(|(first, last)| text.delete(first, last, cancelled))
                .collect::<Result<Vec<_>>>()?;
            if control_phase {
                text.seams(seams, &prefix, &suffix, &literal, &failure, &mut next_controls, &mut literals, cancelled)?;
            } else {
                text.seams(seams, &prefix, &suffix, &literal, &failure, &mut next_controls, &mut next_literals, cancelled)?;
            }
        }
        controls = next_controls;
        literals = next_literals;
    }
    let mut result = String::new();
    let (mut at, mut count) = (text.head, 0);
    while at != END {
        check_cancelled(count, cancelled)?;
        result.push(text.characters[at as usize]);
        at = text.after(at);
        count += 1;
    }
    Ok(result)
}
