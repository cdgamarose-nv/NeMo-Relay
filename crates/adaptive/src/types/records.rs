// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Run and call record types collected by the adaptive telemetry pipeline.

use std::sync::Arc;

use chrono::{DateTime, Utc};
use nemo_flow::codec::request::AnnotatedLlmRequest;
use nemo_flow::codec::response::FinishReason;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

use crate::types::metadata::MetadataEnvelope;

const NEMO_FLOW_INTERNAL_EXTRA_KEY: &str = "_nemo_flow_internal";
const ADAPTIVE_HINTS_EXTRA_KEY: &str = "adaptive_hints";

/// Compact backend timing extracted from an annotated LLM response.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BackendTiming {
    /// Backend admission/prefill queue wait in milliseconds, when provided.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prefill_wait_time_ms: Option<f64>,
    /// Backend prefill execution time in milliseconds, when provided.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prefill_time_ms: Option<f64>,
    /// Backend time-to-first-token in milliseconds, when provided.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub ttft_ms: Option<f64>,
    /// Backend total request time in milliseconds, when provided.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub total_time_ms: Option<f64>,
    /// Router queue depth reported by the backend, when provided.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub router_queue_depth: Option<f64>,
}

impl BackendTiming {
    /// Whether no timing fields are present.
    pub fn is_empty(&self) -> bool {
        self.prefill_wait_time_ms.is_none()
            && self.prefill_time_ms.is_none()
            && self.ttft_ms.is_none()
            && self.total_time_ms.is_none()
            && self.router_queue_depth.is_none()
    }
}

/// Adaptive hint feedback captured at request time for learner updates.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CallAdaptiveHints {
    /// Priority residual arm selected on the request path, when adaptive priority is enabled.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub selected_priority_residual_arm: Option<u8>,
    /// Context key used for the selected priority residual decision, when present.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub selected_priority_residual_key: Option<String>,
    /// Final priority emitted to the backend.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub emitted_priority: Option<u32>,
    /// Maximum priority allowed by the structural policy for this call.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub priority_cap: Option<u32>,
    /// Empirical OSL source selected on the request path, when empirical OSL is enabled.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub selected_osl_source: Option<String>,
    /// Final OSL emitted to the backend after request caps, when present.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub emitted_osl: Option<u32>,
    /// Whether the selected empirical OSL context passed confidence.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub osl_confidence_passed: Option<bool>,
    /// Number of samples in the selected empirical OSL context.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub osl_sample_count: Option<u32>,
}

impl CallAdaptiveHints {
    /// Whether this feedback record contains no learner-update facts.
    pub fn is_empty(&self) -> bool {
        self.selected_priority_residual_arm.is_none()
            && self.selected_priority_residual_key.is_none()
            && self.emitted_priority.is_none()
            && self.priority_cap.is_none()
            && self.selected_osl_source.is_none()
            && self.emitted_osl.is_none()
            && self.osl_confidence_passed.is_none()
            && self.osl_sample_count.is_none()
    }
}

/// Attach local-only adaptive feedback to an annotated request.
pub(crate) fn write_call_adaptive_hints(
    annotated: &mut AnnotatedLlmRequest,
    hints: &CallAdaptiveHints,
) {
    if hints.is_empty() {
        return;
    }
    let Ok(value) = serde_json::to_value(hints) else {
        return;
    };
    let internal = annotated
        .extra
        .entry(NEMO_FLOW_INTERNAL_EXTRA_KEY.to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    let Some(internal) = internal.as_object_mut() else {
        return;
    };
    internal.insert(ADAPTIVE_HINTS_EXTRA_KEY.to_string(), value);
}

/// Read local-only adaptive feedback from an annotated request.
pub(crate) fn read_call_adaptive_hints(
    annotated: &AnnotatedLlmRequest,
) -> Option<CallAdaptiveHints> {
    let value = annotated
        .extra
        .get(NEMO_FLOW_INTERNAL_EXTRA_KEY)?
        .as_object()?
        .get(ADAPTIVE_HINTS_EXTRA_KEY)?;
    let hints = serde_json::from_value::<CallAdaptiveHints>(value.clone()).ok()?;
    (!hints.is_empty()).then_some(hints)
}

/// Kind of runtime call captured in adaptive telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CallKind {
    /// LLM or model-provider invocation.
    Llm,
    /// Tool invocation.
    Tool,
}

/// Telemetry record for a single tool or LLM call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRecord {
    /// Category of call represented by this record.
    pub kind: CallKind,
    /// Logical tool or provider name.
    pub name: String,
    /// Timestamp when the call began.
    pub started_at: DateTime<Utc>,
    /// Timestamp when the call finished, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
    /// Runtime scope UUID for this call.
    #[serde(default)]
    pub scope_uuid: Uuid,
    /// Adaptive metadata snapshot associated with the call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata_snapshot: Option<MetadataEnvelope>,
    /// Output token count reported by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub output_tokens: Option<u32>,
    /// Prompt token count reported by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub prompt_tokens: Option<u32>,
    /// Total token count reported by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub total_tokens: Option<u32>,
    /// Normalized model name associated with the call, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub model_name: Option<String>,
    /// Number of tool calls issued by the provider, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_count: Option<u32>,
    /// Provider tool-call correlation identifier for tool calls, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub tool_call_id: Option<String>,
    /// Annotated request captured for Adaptive Cache Governor (ACG) analysis,
    /// when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_request: Option<Arc<nemo_flow::codec::request::AnnotatedLlmRequest>>,
    /// Annotated response captured for Adaptive Cache Governor (ACG) analysis,
    /// when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub annotated_response: Option<Arc<nemo_flow::codec::response::AnnotatedLlmResponse>>,
    /// Agent/function scope path captured at call-start time.
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub function_path: Vec<String>,
    /// Runtime parent scope UUID for this call, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent_uuid: Option<Uuid>,
    /// Compact graph node identity for calls issued inside a graph node.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub graph: Option<GraphCallContext>,
    /// Run-local ordinal for this call, starting at one.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub run_call_index: Option<u32>,
    /// Normalized finish reason, when reported by the provider.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub finish_reason: Option<FinishReason>,
    /// Backend timing extracted from response metadata, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub backend_timing: Option<BackendTiming>,
    /// Adaptive hints and learner feedback captured when the request was issued.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub adaptive_hints: Option<CallAdaptiveHints>,
}

impl Default for CallRecord {
    fn default() -> Self {
        Self {
            kind: CallKind::Llm,
            name: String::new(),
            started_at: chrono::DateTime::UNIX_EPOCH.into(),
            ended_at: None,
            scope_uuid: Uuid::nil(),
            metadata_snapshot: None,
            output_tokens: None,
            prompt_tokens: None,
            total_tokens: None,
            model_name: None,
            tool_call_count: None,
            tool_call_id: None,
            annotated_request: None,
            annotated_response: None,
            function_path: vec![],
            parent_uuid: None,
            graph: None,
            run_call_index: None,
            finish_reason: None,
            backend_timing: None,
            adaptive_hints: None,
        }
    }
}

/// Compact graph context for a call.
///
/// These facts are intentionally small: enough to reconstruct graph/node
/// dependency structure without storing full framework metadata in every call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphCallContext {
    /// Graph scope name, when available.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub graph_name: Option<String>,
    /// Graph node name that issued this call.
    pub node_name: String,
    /// Graph task identifier for this node execution.
    pub task_id: String,
}

/// Telemetry record for one observed agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Unique run identifier.
    pub id: Uuid,
    /// Agent identifier that produced the run.
    pub agent_id: String,
    /// Calls observed during the run.
    pub calls: Vec<CallRecord>,
    /// Timestamp when the run began.
    pub started_at: DateTime<Utc>,
    /// Timestamp when the run finished, when known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<DateTime<Utc>>,
}
