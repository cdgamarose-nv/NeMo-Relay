// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for adaptive hints intercept in the NeMo Flow adaptive crate.

use super::*;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::dag::{DagCpmNodeState, DagCpmState};
use crate::priority_residual::{
    PressureState, PriorityResidualAction, PriorityResidualArmState, PriorityResidualContextState,
    PriorityResidualState,
};
use crate::trie::data_models::{LlmCallPrediction, PredictionMetrics};
use chrono::Utc;
use nemo_flow::api::runtime::current_scope_stack;
use nemo_flow::api::scope::ScopeType;
use nemo_flow::api::scope::{pop_scope, push_scope};
use nemo_flow::codec::request::{AnnotatedLlmRequest, Message, MessageContent};

static TEST_MUTEX: OnceLock<Mutex<()>> = OnceLock::new();

fn test_mutex() -> &'static Mutex<()> {
    TEST_MUTEX.get_or_init(|| Mutex::new(()))
}

fn reset_root_metadata() {
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

fn make_prediction(latency_sensitivity: Option<u32>) -> LlmCallPrediction {
    LlmCallPrediction {
        remaining_calls: PredictionMetrics {
            sample_count: 10,
            mean: 5.0,
            p50: 5.0,
            p90: 8.0,
            p95: 9.0,
        },
        interarrival_ms: PredictionMetrics {
            sample_count: 10,
            mean: 200.0,
            p50: 180.0,
            p90: 300.0,
            p95: 350.0,
        },
        output_tokens: PredictionMetrics {
            sample_count: 10,
            mean: 100.0,
            p50: 90.0,
            p90: 150.0,
            p95: 180.0,
        },
        latency_sensitivity,
    }
}

fn annotated_with_messages(messages: Vec<Message>) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages,
        model: Some("model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    }
}

#[test]
fn test_build_agent_hints_from_prediction() {
    let pred = make_prediction(Some(4));

    let hints = build_agent_hints(Some(&pred), &None, "test-agent", 2, 3).unwrap();
    assert_eq!(hints.osl, 150, "osl = output_tokens.p90");
    assert_eq!(hints.iat, 200, "iat = interarrival_ms.mean");
    assert_eq!(hints.priority, 1, "priority = 5 - 4 = 1");
    assert!((hints.latency_sensitivity - 4.0).abs() < f64::EPSILON);
    assert_eq!(hints.prefix_id, "test-agent-d3");
    assert_eq!(hints.total_requests, 7, "total_requests = 5 + 2 = 7");
}

#[test]
fn test_build_agent_hints_falls_back_to_defaults() {
    let defaults = AgentHints {
        osl: 42,
        iat: 99,
        priority: 1,
        latency_sensitivity: 4.0,
        prefix_id: "fallback".into(),
        total_requests: 10,
    };
    let hints = build_agent_hints(None, &Some(defaults.clone()), "agent", 1, 0).unwrap();
    assert_eq!(hints.osl, 42);
    assert_eq!(hints.prefix_id, "fallback");
}

#[test]
fn test_build_agent_hints_none_when_no_prediction_and_no_defaults() {
    let hints = build_agent_hints(None, &None, "agent", 1, 0);
    assert!(hints.is_none());
}

#[test]
fn test_build_agent_hints_defaults_missing_latency_sensitivity_to_one() {
    let pred = make_prediction(None);
    let hints = build_agent_hints(Some(&pred), &None, "fallback-agent", 4, 1).unwrap();
    assert_eq!(hints.priority, 4);
    assert_eq!(hints.latency_sensitivity, 1.0);
    assert_eq!(hints.prefix_id, "fallback-agent-d1");
    assert_eq!(hints.total_requests, 9);
}

#[test]
fn test_adaptive_hints_intercept_new() {
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: None,
        priority_residual: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let intercept = AdaptiveHintsIntercept::new(hot_cache, "test".to_string());
    assert_eq!(intercept.call_counter.load(Ordering::Relaxed), 1);
    assert_eq!(intercept.agent_id, "test");
}

#[test]
fn test_adaptive_hints_intercept_into_request_fn_compiles() {
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: None,
        priority_residual: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let intercept = AdaptiveHintsIntercept::new(hot_cache, "test".to_string());
    let _req_fn: LlmRequestInterceptFn = intercept.into_request_fn();
    // If this compiles and runs, the type is correct.
}

#[test]
fn test_adaptive_hints_intercept_injects_prediction_hints_and_manual_override() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let mut root = crate::trie::data_models::PredictionTrieNode::new("root");
    let mut agent_node = crate::trie::data_models::PredictionTrieNode::new("scope-agent");
    let mut function_node = crate::trie::data_models::PredictionTrieNode::new("step");
    function_node
        .predictions_by_call_index
        .insert(1, make_prediction(Some(2)));
    agent_node.children.insert("step".into(), function_node);
    root.children.insert("scope-agent".into(), agent_node);

    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: Some(root),
        agent_hints_default: None,
        dag_cpm: None,
        priority_residual: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let intercept = AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string());
    let req_fn = intercept.into_request_fn();

    let agent_scope = push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("scope-agent")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let function_scope = push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("step")
            .scope_type(ScopeType::Function)
            .parent(&agent_scope)
            .build(),
    )
    .unwrap();
    crate::context_helpers::set_latency_sensitivity(5).unwrap();

    let annotated = AnnotatedLlmRequest {
        messages: vec![Message::User {
            content: MessageContent::Text("hello".into()),
            name: None,
        }],
        model: Some("model".into()),
        params: None,
        tools: None,
        tool_choice: None,
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };
    let (request, returned_annotated) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        Some(annotated.clone()),
    )
    .unwrap();

    let body_hints = &request.content["nvext"]["agent_hints"];
    assert_eq!(body_hints["osl"], serde_json::json!(150));
    assert_eq!(body_hints["iat"], serde_json::json!(200));
    assert_eq!(body_hints["latency_sensitivity"], serde_json::json!(5.0));
    assert_eq!(body_hints["priority"], serde_json::json!(0));
    assert_eq!(body_hints["prefix_id"], serde_json::json!("scope-agent-d2"));
    assert_eq!(body_hints["total_requests"], serde_json::json!(6));
    assert_eq!(
        request.headers.get(AGENT_HINTS_HEADER_KEY).unwrap(),
        body_hints
    );
    assert_eq!(returned_annotated, Some(annotated));

    pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&function_scope.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&agent_scope.uuid)
            .build(),
    )
    .unwrap();
    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_uses_defaults_and_ignores_poisoned_cache() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let defaults = AgentHints {
        osl: 9,
        iat: 12,
        priority: 3,
        latency_sensitivity: 2.0,
        prefix_id: "defaults".into(),
        total_requests: 11,
    };
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: Some(defaults.clone()),
        dag_cpm: None,
        priority_residual: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();
    let (request, annotated) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        None,
    )
    .unwrap();
    assert_eq!(
        request.headers.get(AGENT_HINTS_HEADER_KEY),
        Some(&serde_json::to_value(&defaults).unwrap())
    );
    assert_eq!(
        request.content["nvext"]["agent_hints"]["prefix_id"],
        "defaults"
    );
    assert!(annotated.is_none());

    let poisoned_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: Some(defaults),
        dag_cpm: None,
        priority_residual: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let _ = std::panic::catch_unwind({
        let poisoned_cache = poisoned_cache.clone();
        move || {
            let _guard = poisoned_cache.write().unwrap();
            panic!("poison adaptive cache lock");
        }
    });
    let poisoned_req_fn =
        AdaptiveHintsIntercept::new(poisoned_cache, "fallback-agent".to_string()).into_request_fn();
    let (poisoned_request, _) = poisoned_req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({"existing": true}),
        },
        None,
    )
    .unwrap();
    assert!(
        poisoned_request
            .headers
            .get(AGENT_HINTS_HEADER_KEY)
            .is_none()
    );
    assert_eq!(
        poisoned_request.content,
        serde_json::json!({"existing": true})
    );

    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_overrides_cached_priority_from_dag_cpm() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let mut dag_cpm = DagCpmState::new("fallback-agent");
    dag_cpm.nodes.insert(
        "root/llm:model".to_string(),
        DagCpmNodeState {
            observation_count: 3,
            duration_ms_ewma: 1000.0,
            slack_ms_ewma: 0.0,
            criticality_ewma: 1.0,
            queue_horizon_ms_ewma: 3000.0,
            last_updated_at: None,
        },
    );
    let defaults = AgentHints {
        osl: 9,
        iat: 12,
        priority: 0,
        latency_sensitivity: 2.0,
        prefix_id: "defaults".into(),
        total_requests: 11,
    };
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: Some(defaults),
        dag_cpm: Some(dag_cpm),
        priority_residual: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();

    let (request, _) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        None,
    )
    .unwrap();

    let body_hints = &request.content["nvext"]["agent_hints"];
    assert_eq!(body_hints["priority"], serde_json::json!(3));
    assert_eq!(body_hints["osl"], serde_json::json!(9));
    assert_eq!(body_hints["prefix_id"], serde_json::json!("defaults"));

    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_emits_priority_only_from_dag_cpm() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let mut dag_cpm = DagCpmState::new("fallback-agent");
    dag_cpm.nodes.insert(
        "root/llm:model".to_string(),
        DagCpmNodeState {
            observation_count: 1,
            duration_ms_ewma: 1000.0,
            slack_ms_ewma: 1500.0,
            criticality_ewma: 0.5,
            queue_horizon_ms_ewma: 3000.0,
            last_updated_at: None,
        },
    );
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: Some(dag_cpm),
        priority_residual: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();

    let (request, _) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        None,
    )
    .unwrap();

    let body_hints = &request.content["nvext"]["agent_hints"];
    assert_eq!(body_hints["priority"], serde_json::json!(2));
    assert_eq!(body_hints["osl"], serde_json::json!(0));
    assert_eq!(body_hints["iat"], serde_json::json!(0));
    assert_eq!(
        body_hints["prefix_id"],
        serde_json::json!("fallback-agent-d0")
    );

    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_applies_priority_residual_and_records_feedback() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let mut dag_cpm = DagCpmState::new("fallback-agent");
    dag_cpm.nodes.insert(
        "root/llm:model".to_string(),
        DagCpmNodeState {
            observation_count: 1,
            duration_ms_ewma: 1000.0,
            slack_ms_ewma: 1200.0,
            criticality_ewma: 0.6,
            queue_horizon_ms_ewma: 3000.0,
            last_updated_at: None,
        },
    );

    let now = Utc::now();
    let residual_key = "crit:medium|pressure:high|model:model".to_string();
    let mut residual_context = PriorityResidualContextState::default();
    residual_context.arms[PriorityResidualAction::Noop.arm() as usize] = PriorityResidualArmState {
        pulls: 1,
        effective_pulls: 1.0,
        mean_loss: 1.0,
    };
    residual_context.total_pulls = 1;
    residual_context.last_updated_at = Some(now);
    let mut priority_residual = PriorityResidualState::new("fallback-agent");
    priority_residual
        .contexts
        .insert(residual_key.clone(), residual_context);
    priority_residual.pressure_by_model.insert(
        "model".to_string(),
        PressureState {
            prefill_wait_time_ms: 1200.0,
            last_updated_at: Some(now),
        },
    );

    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: Some(dag_cpm),
        priority_residual: Some(priority_residual),
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();
    let annotated = annotated_with_messages(vec![Message::User {
        content: MessageContent::Text("hello".into()),
        name: None,
    }]);

    let (request, returned_annotated) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        Some(annotated),
    )
    .unwrap();

    assert_eq!(
        request.content["nvext"]["agent_hints"]["priority"],
        serde_json::json!(3)
    );
    assert!(request.content.get("_nemo_flow_internal").is_none());
    let internal = returned_annotated
        .unwrap()
        .extra
        .get("_nemo_flow_internal")
        .unwrap()
        .clone();
    assert_eq!(
        internal["adaptive_hints"]["selected_priority_residual_arm"],
        serde_json::json!(2)
    );
    assert_eq!(
        internal["adaptive_hints"]["selected_priority_residual_key"],
        serde_json::json!(residual_key)
    );
    assert_eq!(
        internal["adaptive_hints"]["emitted_priority"],
        serde_json::json!(3)
    );

    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_uses_workflow_class_priority_cap() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = Some(serde_json::json!({
        "nemo_flow_adaptive": {
            "scheduling_class": "interactive",
        }
    }));
    drop(stack);

    let mut dag_cpm = DagCpmState::new("fallback-agent");
    dag_cpm.nodes.insert(
        "root/llm:model".to_string(),
        DagCpmNodeState {
            observation_count: 3,
            duration_ms_ewma: 1000.0,
            slack_ms_ewma: 0.0,
            criticality_ewma: 1.0,
            queue_horizon_ms_ewma: 3000.0,
            last_updated_at: None,
        },
    );
    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: Some(dag_cpm),
        priority_residual: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();

    let (request, _) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        None,
    )
    .unwrap();

    assert_eq!(
        request.content["nvext"]["agent_hints"]["priority"],
        serde_json::json!(5)
    );

    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_uses_join_fallback_without_matching_dag_cpm_node() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: Some(DagCpmState::new("fallback-agent")),
        priority_residual: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();
    let annotated = annotated_with_messages(vec![Message::Tool {
        content: MessageContent::Text("tool result".into()),
        tool_call_id: "call-1".into(),
    }]);

    let (request, _) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        Some(annotated),
    )
    .unwrap();

    let body_hints = &request.content["nvext"]["agent_hints"];
    assert_eq!(body_hints["priority"], serde_json::json!(2));
    assert_eq!(body_hints["osl"], serde_json::json!(0));
    assert_eq!(
        body_hints["prefix_id"],
        serde_json::json!("fallback-agent-d0")
    );

    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_caps_join_fallback_for_background_workflow() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = Some(serde_json::json!({
        "nemo_flow_adaptive": {
            "workflow_class": "background",
        }
    }));
    drop(stack);

    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: Some(DagCpmState::new("fallback-agent")),
        priority_residual: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();
    let annotated = annotated_with_messages(vec![Message::Tool {
        content: MessageContent::Text("tool result".into()),
        tool_call_id: "call-1".into(),
    }]);

    let (request, _) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        Some(annotated),
    )
    .unwrap();

    assert_eq!(
        request.content["nvext"]["agent_hints"]["priority"],
        serde_json::json!(1)
    );

    reset_root_metadata();
}

#[test]
fn test_adaptive_hints_intercept_uses_small_root_fallback_without_matching_dag_cpm_node() {
    let _guard = test_mutex().lock().unwrap();
    reset_root_metadata();

    let hot_cache = Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: Some(DagCpmState::new("fallback-agent")),
        priority_residual: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }));
    let req_fn =
        AdaptiveHintsIntercept::new(hot_cache, "fallback-agent".to_string()).into_request_fn();

    let (request, _) = req_fn(
        "model",
        LlmRequest {
            headers: serde_json::Map::new(),
            content: serde_json::json!({}),
        },
        None,
    )
    .unwrap();

    assert_eq!(
        request.content["nvext"]["agent_hints"]["priority"],
        serde_json::json!(1)
    );

    reset_root_metadata();
}

#[test]
fn test_apply_manual_latency_override_and_inject_agent_hints_cover_manual_paths() {
    let base_hints = AgentHints {
        osl: 10,
        iat: 20,
        priority: 3,
        latency_sensitivity: 3.0,
        prefix_id: "base-d1".into(),
        total_requests: 4,
    };

    let unchanged =
        apply_manual_latency_override(Some(base_hints.clone()), Some(2), "agent", 1).unwrap();
    assert_eq!(unchanged.latency_sensitivity, 3.0);

    let manual_only = apply_manual_latency_override(None, Some(5), "agent", 2).unwrap();
    assert_eq!(manual_only.osl, 0);
    assert_eq!(manual_only.iat, 0);
    assert_eq!(manual_only.latency_sensitivity, 5.0);
    assert_eq!(manual_only.prefix_id, "agent-d2");
    assert_eq!(manual_only.total_requests, 0);

    assert!(apply_manual_latency_override(None, None, "agent", 0).is_none());

    let mut non_object_request = LlmRequest {
        headers: serde_json::Map::new(),
        content: serde_json::json!("scalar"),
    };
    inject_agent_hints(&mut non_object_request, &manual_only);
    assert_eq!(
        non_object_request.headers.get(AGENT_HINTS_HEADER_KEY),
        Some(&serde_json::to_value(&manual_only).unwrap())
    );
}
