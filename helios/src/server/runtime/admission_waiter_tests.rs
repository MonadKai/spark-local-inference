/* SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
 * SPDX-License-Identifier: Apache-2.0 */

//! One async waiter exposes semaphore assignments without inspecting Tokio internals.

use super::*;
use std::{collections::{BTreeSet, VecDeque}, future::Future, pin::Pin, task::{Context, Poll}};

type Action = [u64; 3];
type Waiting = Pin<Box<dyn Future<Output = Result<Generation>> + Send>>;

struct Waiters<'a> {
    runtime: Arc<ModelRuntime>,
    worker: Worker<'a>,
    received: [Option<Incoming>; 2],
    readers: [Option<Generation>; 2],
    waiting: Option<Waiting>,
    phase: u64,
    capacity: usize,
}

impl<'a> Waiters<'a> {
    fn new(fixture: &'a Fixture, capacity: usize) -> Self {
        let (mut runtime, receiver) = group_runtime(fixture, capacity);
        runtime.queue_capacity = Arc::new(tokio::sync::Semaphore::new(1));
        Self { runtime: Arc::new(runtime), worker: fixture.worker(receiver),
            received: [None, None], readers: [None, None], waiting: None, phase: 0, capacity }
    }

    fn prompt(row: usize) -> Prompt {
        Fixture::prompt_for(vec![row as u32], 2, PromptData {
            positions: vec![[0; 3]], vision_token_indices: vec![-1], ..Default::default()
        })
    }

    fn producer(&self, row: usize) -> Option<&Incoming> {
        self.received[row].as_ref().or_else(|| self.worker.pending.iter()
            .find(|item| item.prompt.tokens[0] == row as u32))
    }

    fn row(&self, row: usize) -> [u64; 3] {
        let producer = self.producer(row);
        let location = if self.received[row].is_some() { 1 } else if producer.is_some() { 2 } else { 0 };
        if let Some(item) = producer {
            assert_eq!(item._capacity.as_ref().unwrap().num_permits(), 1);
            assert_eq!(item.queue_capacity.is_some(), location == 1,
                "worker enqueue must release only message credit");
            if let Some(reader) = &self.readers[row] {
                assert!(Arc::ptr_eq(item._capacity.as_ref().unwrap(), reader._capacity.as_ref().unwrap()),
                    "producer and reader must share host credit");
            }
        }
        let cancellation = producer.map(|item| &item.cancelled)
            .or_else(|| self.readers[row].as_ref().map(|item| &item.cancelled));
        [location, u64::from(self.readers[row].is_some()),
            u64::from(cancellation.is_some_and(|item| item.load(Ordering::Acquire)))]
    }

    fn occupied(&self) -> usize {
        (0..2).filter(|&row| self.producer(row).is_some() || self.readers[row].is_some()).count()
    }

    fn projection(&self) -> Vec<u64> {
        let rows = [self.row(0), self.row(1)];
        let host = self.runtime.capacity.available_permits();
        let message = self.runtime.queue_capacity.available_permits();
        assert_eq!(self.runtime.capacity.is_closed(), self.runtime.queue_capacity.is_closed(),
            "closing admission must close both credit pools");
        assert_eq!(self.waiting.is_some(), self.phase != 0);
        let assigned_host = self.capacity.checked_sub(host + self.occupied() + usize::from(self.phase == 2))
            .expect("waiter host credit accounting");
        let assigned_message = 1usize.checked_sub(message + self.received.iter().filter(|item| item.is_some()).count())
            .expect("waiter message credit accounting");
        // Residual credits can only belong to the single, independently
        // observed pending acquire. They are not unexplained spare capacity.
        assert!(assigned_host <= usize::from(self.phase == 1), "unowned host credit");
        assert!(assigned_message <= usize::from(self.phase == 2), "unowned message credit");
        let mut result = vec![self.capacity as u64, host as u64, message as u64,
            u64::from(self.runtime.capacity.is_closed()), self.phase, assigned_host as u64, assigned_message as u64];
        for row in rows { result.extend(row); }
        result
    }

    fn receive(&mut self, row: usize, reader: Generation) {
        let mut message = self.worker.receiver.try_recv().unwrap();
        assert_eq!(message.len(), 1);
        let incoming = message.pop().unwrap();
        assert_eq!(incoming.prompt.tokens[0], row as u32);
        assert!(self.received[row].is_none() && self.readers[row].is_none());
        self.received[row] = Some(incoming);
        self.readers[row] = Some(reader);
    }

    fn poll(&mut self) {
        let result = self.waiting.as_mut().unwrap().as_mut()
            .poll(&mut Context::from_waker(futures_util::task::noop_waker_ref()));
        match result {
            Poll::Pending => {
                self.phase = if self.capacity - self.occupied() > self.runtime.capacity.available_permits() { 2 } else { 1 };
            }
            Poll::Ready(result) => {
                self.waiting.take();
                self.phase = 0;
                match result {
                    Ok(reader) => self.receive(1, reader),
                    Err(error) => {
                        assert!(self.runtime.capacity.is_closed(), "open admission lost a valid waiter: {error}");
                        assert!(matches!(error, Error::Admission(AdmissionError::Unavailable(_))));
                    }
                }
            }
        }
    }

    fn apply(&mut self, [kind, row, value]: Action) {
        let row = row as usize;
        match kind {
            1 => self.receive(0, self.runtime.generate(Self::prompt(0)).unwrap()),
            2 => {
                let runtime = self.runtime.clone();
                self.waiting = Some(Box::pin(async move { runtime.generate_when_available(Self::prompt(1)).await }));
                self.poll();
            }
            3 => self.poll(),
            4 => { self.waiting.take(); self.phase = 0; }
            5 => { self.readers[row].take(); }
            6 => {
                if self.received[row].take().is_none() {
                    let index = self.worker.pending.iter().position(|item| item.prompt.tokens[0] == row as u32).unwrap();
                    self.worker.pending.remove(index);
                }
            }
            7 => self.worker.enqueue(vec![self.received[row].take().unwrap()]),
            8 => {
                if value == 0 { self.runtime.request_stop(); }
                else { drop(CloseAdmission { requests: self.runtime.capacity.clone(), messages: self.runtime.queue_capacity.clone() }); }
            }
            9 => assert!(matches!(self.runtime.generate(Self::prompt(0)),
                Err(Error::Admission(AdmissionError::Capacity(_) | AdmissionError::Unavailable(_))))),
            _ => panic!("unknown admission-waiter action"),
        }
    }

    fn actions(&self) -> Vec<Action> {
        let mut result = Vec::new();
        let closed = self.runtime.capacity.is_closed();
        let host = self.runtime.capacity.available_permits();
        let message = self.runtime.queue_capacity.available_permits();
        if !closed && host > 0 && message > 0 && self.row(0)[..2] == [0, 0] { result.push([1, 0, 0]); }
        if !closed && self.waiting.is_none() && self.row(1)[..2] == [0, 0] { result.push([2, 0, 0]); }
        if self.waiting.is_some() { result.extend([[3, 0, 0], [4, 0, 0]]); }
        for row in 0..2 {
            if self.readers[row].is_some() { result.push([5, row as u64, 0]); }
            if self.producer(row).is_some() { result.push([6, row as u64, 0]); }
            if self.received[row].is_some() { result.push([7, row as u64, 0]); }
        }
        if !closed { result.extend([[8, 0, 0], [8, 0, 1]]); }
        if closed || host == 0 || message == 0 { result.push([9, 0, 0]); }
        result
    }
}

#[test]
fn admission_waiters_match_credit_model() {
    let fixture = Fixture::new();
    let mut edges = BTreeSet::new();
    let (mut host_wait, mut queue_wait, mut assigned_host, mut assigned_message) = (false, false, false, false);
    let mut closed_waiter = [false; 2];
    let mut cancelled_waiter = [false; 2];
    let mut total = 0;
    for capacity in [1, 2] {
        let mut paths = VecDeque::from([Vec::<Action>::new()]);
        let mut visited = BTreeSet::new();
        while let Some(path) = paths.pop_front() {
            let mut current = Waiters::new(&fixture, capacity);
            for action in &path { current.apply(*action); }
            let before = current.projection();
            if !visited.insert(before.clone()) { continue; }
            host_wait |= before[4] == 1;
            queue_wait |= before[4] == 2;
            assigned_host |= before[5] == 1;
            assigned_message |= before[6] == 1;
            assert!(path.len() < 24, "increase the explicit admission-waiter exploration bound");
            for action in current.actions() {
                let mut candidate = Waiters::new(&fixture, capacity);
                for previous in &path { candidate.apply(*previous); }
                candidate.apply(action);
                let after = candidate.projection();
                if before[4] > 0 && action[0] == 4 { cancelled_waiter[before[4] as usize - 1] = true; }
                if before[3] == 1 && before[4] > 0 && action[0] == 3 { closed_waiter[before[4] as usize - 1] = true; }
                edges.insert((before.clone(), action, after));
                let mut next = path.clone(); next.push(action); paths.push_back(next);
            }
        }
        total += visited.len();
        // Repeated real reuse has no admission-count cutoff or new runtime.
        let mut reused = Waiters::new(&fixture, capacity);
        for _ in 0..100 {
            for action in [[1, 0, 0], [2, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0]] { reused.apply(action); }
            assert_eq!(reused.projection(), Waiters::new(&fixture, capacity).projection());
        }
    }
    assert!(host_wait && queue_wait && assigned_host && assigned_message, "missing pending/assigned credit state");
    assert!(closed_waiter.into_iter().all(|value| value) && cancelled_waiter.into_iter().all(|value| value),
        "missing waiter cancellation or shutdown path");
    for edge in &edges { println!("ADMISSION_WAITER_EDGE [{:?}, {:?}, {:?}]", edge.0, edge.1, edge.2); }
    println!("ADMISSION_WAITER_COVERAGE states={total} edges={} reuse=100 per capacity", edges.len());
}
