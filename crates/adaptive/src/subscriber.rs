// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Event subscriber factory and event-to-record mapping helpers.

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use crate::context_helpers::extract_scope_path;
use crate::types::records::{CallKind, CallRecord, read_call_adaptive_hints};
use nemo_flow::api::event::{Event, ScopeCategory};
use nemo_flow::api::runtime::EventSubscriberFn;
use nemo_flow::api::scope::ScopeType;

#[cfg(test)]
pub(crate) fn create_subscriber(
    tx: tokio::sync::mpsc::UnboundedSender<(Event, Vec<String>)>,
) -> EventSubscriberFn {
    create_subscriber_with_counter(tx, Arc::new(AtomicUsize::new(0)))
}

pub(crate) fn create_subscriber_with_counter(
    tx: tokio::sync::mpsc::UnboundedSender<(Event, Vec<String>)>,
    pending_events: Arc<AtomicUsize>,
) -> EventSubscriberFn {
    std::sync::Arc::new(move |event: &Event| {
        pending_events.fetch_add(1, Ordering::SeqCst);
        let scope_path = extract_scope_path();
        if tx.send((event.clone(), scope_path)).is_err() {
            pending_events.fetch_sub(1, Ordering::SeqCst);
        }
    })
}

pub(crate) fn event_to_call_record(event: &Event, scope_path: &[String]) -> Option<CallRecord> {
    if event.scope_category() != Some(ScopeCategory::Start) {
        return None;
    }
    let (kind, annotated_request) = match event.category().map(|category| category.as_str()) {
        Some("llm") => (CallKind::Llm, event.annotated_request().cloned()),
        Some("tool") => (CallKind::Tool, None),
        _ => return None,
    };
    let adaptive_hints = annotated_request
        .as_deref()
        .and_then(read_call_adaptive_hints);
    Some(CallRecord {
        kind,
        name: event.name().to_string(),
        started_at: *event.timestamp(),
        scope_uuid: event.uuid(),
        tool_call_id: event.tool_call_id().map(str::to_string),
        annotated_request,
        adaptive_hints,
        function_path: scope_path.to_vec(),
        ..CallRecord::default()
    })
}

pub(crate) fn is_run_boundary(event: &Event) -> bool {
    event.scope_type() == Some(ScopeType::Agent)
        && matches!(
            event.scope_category(),
            Some(ScopeCategory::Start | ScopeCategory::End)
        )
}

#[cfg(test)]
#[path = "../tests/coverage/subscriber_tests.rs"]
mod tests;
