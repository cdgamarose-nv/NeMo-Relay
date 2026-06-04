// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! AdaptiveHintsIntercept: opt-in LLM request intercept that injects AgentHints
//! from HotCache trie.
//!
//! This module provides [`AdaptiveHintsIntercept`], which builds [`AgentHints`] from
//! the prediction trie in [`HotCache`] and injects them into LLM request
//! headers as a request intercept. AdaptiveHintsIntercept is opt-in and synchronously
//! transforms the [`LlmRequest`] before it reaches the callable.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};

use nemo_flow::api::llm::LlmRequest;
use nemo_flow::api::runtime::LlmRequestInterceptFn;
use nemo_flow::codec::request::{AnnotatedLlmRequest, Message};
use serde_json::{Map, Value};

use crate::context_helpers::{
    WorkflowClass, extract_graph_call_context, extract_scope_path, read_manual_latency_sensitivity,
    read_workflow_class, resolve_agent_id, resolve_run_boundary_scope_uuid,
};
use crate::dag::{
    DEFAULT_DAG_PRIORITY_CAP, DagCpmState, MAX_DAG_PRIORITY_CAP, llm_structural_key_with_graph,
    project_priority_prior,
};
use crate::intercepts::AGENT_HINTS_HEADER_KEY;
use crate::osl_empirical::{OslEmpiricalState, OslRequestSignature, cap_osl_to_request_limit};
use crate::priority_residual::PriorityResidualState;
use crate::trie::builder::SensitivityConfig;
use crate::trie::lookup::PredictionTrieLookup;
use crate::types::cache::HotCache;
use crate::types::metadata::AgentHints;
use crate::types::records::{CallAdaptiveHints, GraphCallContext, write_call_adaptive_hints};
use uuid::Uuid;

const DEFAULT_PRIORITY_ADJUSTMENT: i32 = 0;
const DYNAMO_TIMING_EXTRA_FIELD: &str = "timing";

#[derive(Debug, Default)]
struct HintSelection {
    hints: Option<AgentHints>,
    feedback: Option<CallAdaptiveHints>,
}

/// Builds [`AgentHints`] from a trie prediction and optional default hints.
///
/// Falls back to `default_hints` if no prediction is available.
/// Sets `prefix_id` to `"{agent_id}-d{scope_depth}"` per architecture doc.
pub(crate) fn build_agent_hints(
    prediction: Option<&crate::trie::data_models::LlmCallPrediction>,
    default_hints: &Option<AgentHints>,
    agent_id: &str,
    call_index: u32,
    scope_depth: usize,
) -> Option<AgentHints> {
    if let Some(pred) = prediction {
        let scale = SensitivityConfig::default().sensitivity_scale;
        let ls = pred.latency_sensitivity.unwrap_or(1);
        Some(AgentHints {
            osl: Some(pred.output_tokens.p90.round() as u32),
            iat: pred.interarrival_ms.mean.round() as u32,
            priority: (scale as i32 - ls as i32).max(0),
            latency_sensitivity: ls as f64,
            prefix_id: format!("{agent_id}-d{scope_depth}"),
            total_requests: pred.remaining_calls.mean.round() as u32 + call_index,
        })
    } else {
        default_hints.clone()
    }
}

fn apply_manual_latency_override(
    hints: Option<AgentHints>,
    manual_ls: Option<u32>,
    effective_agent_id: &str,
    scope_depth: usize,
) -> Option<AgentHints> {
    match (hints, manual_ls) {
        (Some(mut hints), Some(manual)) => {
            let manual_f = manual as f64;
            if manual_f > hints.latency_sensitivity {
                let scale = SensitivityConfig::default().sensitivity_scale;
                hints.latency_sensitivity = manual_f;
                hints.priority = (scale as i32 - manual_f.round() as i32).max(0);
            }
            Some(hints)
        }
        (Some(hints), None) => Some(hints),
        (None, Some(manual)) => Some(manual_agent_hints(manual, effective_agent_id, scope_depth)),
        (None, None) => None,
    }
}

fn apply_dag_priority_prior(
    hints: Option<AgentHints>,
    dag_cpm: Option<&DagCpmState>,
    priority_residual: Option<&PriorityResidualState>,
    structural_key: &str,
    model_name: &str,
    effective_agent_id: &str,
    scope_depth: usize,
    priority_cap: u32,
    fallback_priority: Option<u32>,
) -> HintSelection {
    let learned = dag_cpm
        .and_then(|state| state.nodes.get(structural_key))
        .and_then(|node| {
            let priority = project_priority_prior(node, DEFAULT_PRIORITY_ADJUSTMENT, priority_cap)?;
            Some((priority, node.criticality_ewma))
        });

    match (hints, learned, fallback_priority) {
        (Some(mut hints), Some((priority, criticality)), _) => {
            let (priority, feedback) = apply_priority_residual(
                priority,
                priority_residual,
                criticality,
                model_name,
                priority_cap,
            );
            hints.priority = priority as i32;
            HintSelection {
                hints: Some(hints),
                feedback,
            }
        }
        (None, Some((priority, criticality)), _) => {
            let (priority, feedback) = apply_priority_residual(
                priority,
                priority_residual,
                criticality,
                model_name,
                priority_cap,
            );
            HintSelection {
                hints: Some(priority_only_agent_hints(
                    priority,
                    effective_agent_id,
                    scope_depth,
                )),
                feedback,
            }
        }
        (Some(mut hints), None, Some(priority)) => {
            hints.priority = hints.priority.max(priority as i32);
            HintSelection {
                hints: Some(hints),
                feedback: None,
            }
        }
        (None, None, Some(priority)) => HintSelection {
            hints: Some(priority_only_agent_hints(
                priority,
                effective_agent_id,
                scope_depth,
            )),
            feedback: None,
        },
        (hints, None, None) => HintSelection {
            hints,
            feedback: None,
        },
    }
}

fn apply_priority_residual(
    priority_prior: u32,
    priority_residual: Option<&PriorityResidualState>,
    criticality: f64,
    model_name: &str,
    priority_cap: u32,
) -> (u32, Option<CallAdaptiveHints>) {
    let Some(decision) =
        priority_residual.and_then(|state| state.decision(criticality, model_name))
    else {
        return (priority_prior, None);
    };
    let cap = priority_cap.min(MAX_DAG_PRIORITY_CAP);
    let priority = (priority_prior as i32 + decision.delta).clamp(0, cap as i32) as u32;
    (
        priority,
        Some(CallAdaptiveHints {
            selected_priority_residual_arm: Some(decision.action.arm()),
            selected_priority_residual_key: Some(decision.key),
            emitted_priority: Some(priority),
            priority_cap: Some(cap),
            ..CallAdaptiveHints::default()
        }),
    )
}

fn merge_feedback(selection: &mut HintSelection, feedback: CallAdaptiveHints) {
    if feedback.is_empty() {
        return;
    }
    let target = selection
        .feedback
        .get_or_insert_with(CallAdaptiveHints::default);
    if target.selected_priority_residual_arm.is_none() {
        target.selected_priority_residual_arm = feedback.selected_priority_residual_arm;
    }
    if target.selected_priority_residual_key.is_none() {
        target.selected_priority_residual_key = feedback.selected_priority_residual_key;
    }
    if target.emitted_priority.is_none() {
        target.emitted_priority = feedback.emitted_priority;
    }
    if target.priority_cap.is_none() {
        target.priority_cap = feedback.priority_cap;
    }
    target.selected_osl_source = feedback.selected_osl_source;
    target.emitted_osl = feedback.emitted_osl;
    target.osl_confidence_passed = feedback.osl_confidence_passed;
    target.osl_sample_count = feedback.osl_sample_count;
}

fn priority_only_agent_hints(
    priority: u32,
    effective_agent_id: &str,
    scope_depth: usize,
) -> AgentHints {
    AgentHints {
        osl: None,
        iat: 0,
        priority: priority as i32,
        latency_sensitivity: 0.0,
        prefix_id: format!("{effective_agent_id}-d{scope_depth}"),
        total_requests: 0,
    }
}

fn osl_only_agent_hints(osl: u32, effective_agent_id: &str, scope_depth: usize) -> AgentHints {
    AgentHints {
        osl: Some(osl),
        iat: 0,
        priority: 0,
        latency_sensitivity: 0.0,
        prefix_id: format!("{effective_agent_id}-d{scope_depth}"),
        total_requests: 0,
    }
}

fn apply_empirical_osl_overlay(
    mut selection: HintSelection,
    empirical_state: Option<&OslEmpiricalState>,
    annotated: Option<&AnnotatedLlmRequest>,
    root_uuid: Option<Uuid>,
    model_name: &str,
    state_agent_id: &str,
    hint_prefix_agent_id: &str,
    scope_depth: usize,
) -> HintSelection {
    let Some(empirical_state) = empirical_state else {
        return selection;
    };

    let predicted_osl = annotated.and_then(|request| {
        let signature = OslRequestSignature::from_request(request);
        let mut outcome = empirical_state.predict(root_uuid, state_agent_id, model_name, signature);
        if let Some(prediction) = outcome.emitted_osl {
            outcome.emitted_osl = Some(cap_osl_to_request_limit(prediction, request));
        }
        merge_feedback(
            &mut selection,
            CallAdaptiveHints {
                selected_osl_source: Some(outcome.source.to_string()),
                emitted_osl: outcome.emitted_osl,
                osl_confidence_passed: Some(outcome.confidence_passed),
                osl_sample_count: outcome.sample_count,
                ..CallAdaptiveHints::default()
            },
        );
        outcome.emitted_osl
    });

    match (&mut selection.hints, predicted_osl) {
        (Some(hints), Some(osl)) => hints.osl = Some(osl),
        (Some(hints), None) => hints.osl = None,
        (None, Some(osl)) => {
            selection.hints = Some(osl_only_agent_hints(osl, hint_prefix_agent_id, scope_depth));
        }
        (None, None) => {}
    }

    selection
}

fn priority_cap_for_workflow_class(workflow_class: Option<WorkflowClass>) -> u32 {
    match workflow_class {
        Some(WorkflowClass::Background) => 1,
        Some(WorkflowClass::Interactive) => MAX_DAG_PRIORITY_CAP,
        Some(WorkflowClass::Standard) | None => DEFAULT_DAG_PRIORITY_CAP,
    }
}

fn fallback_priority_prior(
    annotated: Option<&AnnotatedLlmRequest>,
    call_index: u32,
    scope_depth: usize,
    priority_cap: u32,
) -> Option<u32> {
    if priority_cap == 0 {
        return None;
    }

    if request_ends_with_tool_result(annotated) {
        return Some(priority_cap.min(2));
    }

    if call_index == 1 || scope_depth == 0 {
        return Some(priority_cap.min(1));
    }

    None
}

fn request_ends_with_tool_result(annotated: Option<&AnnotatedLlmRequest>) -> bool {
    annotated
        .and_then(|request| request.messages.last())
        .is_some_and(|message| matches!(message, Message::Tool { .. }))
}

fn manual_agent_hints(manual: u32, effective_agent_id: &str, scope_depth: usize) -> AgentHints {
    let scale = SensitivityConfig::default().sensitivity_scale;
    AgentHints {
        osl: None,
        iat: 0,
        priority: (scale as i32 - manual as i32).max(0),
        latency_sensitivity: manual as f64,
        prefix_id: format!("{effective_agent_id}-d{scope_depth}"),
        total_requests: 0,
    }
}

fn ensure_nvext_object(body: &mut Map<String, Value>) -> Option<&mut Map<String, Value>> {
    if !body.contains_key("nvext") {
        body.insert("nvext".to_string(), Value::Object(Map::new()));
    }
    body.get_mut("nvext").and_then(Value::as_object_mut)
}

fn ensure_dynamo_timing_field(nvext: &mut Map<String, Value>) {
    match nvext.get_mut("extra_fields") {
        Some(Value::Array(fields)) => {
            let has_timing = fields
                .iter()
                .any(|field| field.as_str() == Some(DYNAMO_TIMING_EXTRA_FIELD));
            if !has_timing {
                fields.push(Value::String(DYNAMO_TIMING_EXTRA_FIELD.to_string()));
            }
        }
        Some(value) => {
            *value = Value::Array(vec![Value::String(DYNAMO_TIMING_EXTRA_FIELD.to_string())]);
        }
        None => {
            nvext.insert(
                "extra_fields".to_string(),
                Value::Array(vec![Value::String(DYNAMO_TIMING_EXTRA_FIELD.to_string())]),
            );
        }
    }
}

fn request_dynamo_timing_in_body(body: &mut Map<String, Value>) {
    if let Some(nvext) = ensure_nvext_object(body) {
        ensure_dynamo_timing_field(nvext);
    }
}

fn request_dynamo_timing(request: &mut LlmRequest) {
    if let Some(body) = request.content.as_object_mut() {
        request_dynamo_timing_in_body(body);
    }
}

fn request_dynamo_timing_annotated(annotated: &mut AnnotatedLlmRequest) {
    request_dynamo_timing_in_body(&mut annotated.extra);
}

fn write_agent_hints_to_nvext(body: &mut Map<String, Value>, serialized_hints: Value) {
    if let Some(nvext) = ensure_nvext_object(body) {
        nvext.insert("agent_hints".to_string(), serialized_hints);
    }
}

fn inject_agent_hints(request: &mut LlmRequest, hints: &AgentHints) {
    let Ok(serialized_hints) = serde_json::to_value(hints) else {
        return;
    };

    if let Some(body) = request.content.as_object_mut() {
        write_agent_hints_to_nvext(body, serialized_hints.clone());
    }

    request.headers.insert(
        AGENT_HINTS_HEADER_KEY.to_string(),
        Value::String(serialized_hints.to_string()),
    );
}

fn inject_agent_hints_annotated(annotated: &mut AnnotatedLlmRequest, hints: &AgentHints) {
    let Ok(serialized_hints) = serde_json::to_value(hints) else {
        return;
    };

    write_agent_hints_to_nvext(&mut annotated.extra, serialized_hints);
}

fn clear_priority_residual_feedback(feedback: &mut CallAdaptiveHints) {
    feedback.selected_priority_residual_arm = None;
    feedback.selected_priority_residual_key = None;
}

fn record_emitted_hint_facts(
    feedback: &mut CallAdaptiveHints,
    hints: &AgentHints,
    priority_cap: u32,
) {
    if hints.priority >= 0 {
        feedback.emitted_priority = Some(hints.priority as u32);
        feedback.priority_cap = Some(priority_cap.min(MAX_DAG_PRIORITY_CAP));
    }
    feedback.emitted_osl = hints.osl;
}

/// Opt-in LLM request intercept that injects [`AgentHints`] into request
/// headers from the adaptive state in [`HotCache`].
///
/// Constructed via [`AdaptiveHintsIntercept::new`] and converted to an
/// [`LlmRequestInterceptFn`] via [`AdaptiveHintsIntercept::into_request_fn`] for
/// registration with the NeMo Flow runtime.
pub struct AdaptiveHintsIntercept {
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    call_counter: AtomicU32,
}

impl AdaptiveHintsIntercept {
    /// Creates a new `AdaptiveHintsIntercept`.
    pub fn new(hot_cache: Arc<RwLock<HotCache>>, agent_id: String) -> Self {
        Self {
            hot_cache,
            agent_id,
            call_counter: AtomicU32::new(1),
        }
    }

    fn effective_agent_id(&self) -> String {
        resolve_agent_id().unwrap_or_else(|| self.agent_id.clone())
    }

    fn load_hint_selection(
        &self,
        llm_name: &str,
        scope_path: &[String],
        effective_agent_id: &str,
        call_index: u32,
        scope_depth: usize,
        priority_cap: u32,
        annotated: Option<&AnnotatedLlmRequest>,
        root_uuid: Option<Uuid>,
        graph_context: Option<&GraphCallContext>,
    ) -> HintSelection {
        let Ok(cache_guard) = self.hot_cache.read() else {
            return HintSelection::default();
        };

        let trie_hints = if let Some(ref trie) = cache_guard.trie {
            let lookup = PredictionTrieLookup::new(trie);
            let prediction = lookup.find(scope_path, call_index);
            build_agent_hints(
                prediction,
                &cache_guard.agent_hints_default,
                effective_agent_id,
                call_index,
                scope_depth,
            )
        } else {
            cache_guard.agent_hints_default.clone()
        };

        let structural_key = llm_structural_key_with_graph(scope_path, llm_name, graph_context);
        let fallback_priority = cache_guard.dag_cpm.as_ref().and_then(|_| {
            fallback_priority_prior(annotated, call_index, scope_depth, priority_cap)
        });
        let model_name = annotated
            .and_then(|request| request.model.as_deref())
            .unwrap_or(llm_name);
        let selection = apply_dag_priority_prior(
            trie_hints,
            cache_guard.dag_cpm.as_ref(),
            cache_guard.priority_residual.as_ref(),
            &structural_key,
            model_name,
            effective_agent_id,
            scope_depth,
            priority_cap,
            fallback_priority,
        );
        apply_empirical_osl_overlay(
            selection,
            cache_guard.osl_empirical.as_ref(),
            annotated,
            root_uuid,
            model_name,
            &self.agent_id,
            effective_agent_id,
            scope_depth,
        )
    }

    /// Converts this intercept into an [`LlmRequestInterceptFn`] suitable for
    /// registration with [`register_llm_request_intercept`].
    ///
    /// The returned closure reads HotCache hints, projects DAG CPM priority when
    /// available, injects AgentHints into the request headers and body, and
    /// returns the transformed request.
    pub fn into_request_fn(self) -> LlmRequestInterceptFn {
        let this = Arc::new(self);
        Box::new(
            move |name: &str,
                  mut request: LlmRequest,
                  mut annotated: Option<AnnotatedLlmRequest>| {
                let scope_path = extract_scope_path();
                let manual_ls = read_manual_latency_sensitivity();
                let priority_cap = priority_cap_for_workflow_class(read_workflow_class());
                let root_uuid = resolve_run_boundary_scope_uuid();
                let graph_context = extract_graph_call_context();
                let scope_depth = scope_path.len();
                let call_index = this.call_counter.fetch_add(1, Ordering::Relaxed);

                let effective_agent_id = this.effective_agent_id();
                let mut selection = this.load_hint_selection(
                    name,
                    &scope_path,
                    &effective_agent_id,
                    call_index,
                    scope_depth,
                    priority_cap,
                    annotated.as_ref(),
                    root_uuid,
                    graph_context.as_ref(),
                );
                let final_hints = apply_manual_latency_override(
                    selection.hints.take(),
                    manual_ls,
                    &effective_agent_id,
                    scope_depth,
                );

                request_dynamo_timing(&mut request);
                if let Some(annotated) = annotated.as_mut() {
                    request_dynamo_timing_annotated(annotated);
                }

                if let Some(hints) = final_hints.as_ref() {
                    inject_agent_hints(&mut request, &hints);
                    if let Some(annotated) = annotated.as_mut() {
                        inject_agent_hints_annotated(annotated, &hints);
                    }
                }

                if let Some(annotated) = annotated.as_mut() {
                    let mut feedback = selection.feedback.take().unwrap_or_default();
                    if manual_ls.is_some() {
                        clear_priority_residual_feedback(&mut feedback);
                    }
                    if let Some(hints) = final_hints.as_ref() {
                        record_emitted_hint_facts(&mut feedback, hints, priority_cap);
                    }
                    write_call_adaptive_hints(annotated, &feedback);
                }

                Ok((request, annotated))
            },
        )
    }
}

#[cfg(test)]
#[path = "../tests/unit/adaptive_hints_intercept_tests.rs"]
mod tests;
