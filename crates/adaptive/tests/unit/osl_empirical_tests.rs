// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::sync::{Arc, RwLock};

use chrono::{TimeZone, Utc};
use nemo_flow::codec::request::{
    AnnotatedLlmRequest, FunctionDefinition, Message, MessageContent, ToolDefinition,
};
use nemo_flow::codec::response::FinishReason;
use uuid::Uuid;

use crate::learner::traits::Learner;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::records::{CallKind, CallRecord, RunRecord};

use super::{
    MAX_PERSISTENT_CONTEXTS, MAX_RUN_CONTEXTS, OSL_MAX_SPREAD_MULTIPLIER, OslContextKey,
    OslContextScope, OslContextStats, OslEmpiricalLearner, OslEmpiricalState, OslMessageRole,
    OslRequestSignature,
};

fn request(messages: Vec<Message>, tools: Option<Vec<ToolDefinition>>) -> AnnotatedLlmRequest {
    AnnotatedLlmRequest {
        messages,
        model: Some("model-a".to_string()),
        params: None,
        tools,
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

fn tool_definition() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "search".to_string(),
            description: None,
            parameters: None,
        },
    }
}

fn empty_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: None,
        priority_residual: None,
        osl_empirical: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

fn llm_call(
    output_tokens: Option<u32>,
    annotated_request: Option<AnnotatedLlmRequest>,
) -> CallRecord {
    let now = Utc.timestamp_millis_opt(1000).single().unwrap();
    CallRecord {
        kind: CallKind::Llm,
        name: "llm".to_string(),
        started_at: now,
        ended_at: Some(now),
        output_tokens,
        model_name: Some("model-a".to_string()),
        annotated_request: annotated_request.map(Arc::new),
        ..Default::default()
    }
}

fn sample_run(calls: Vec<CallRecord>) -> RunRecord {
    let now = Utc.timestamp_millis_opt(2000).single().unwrap();
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: "agent-a".to_string(),
        calls,
        started_at: now,
        ended_at: Some(now),
    }
}

#[test]
fn request_signature_handles_empty_messages() {
    let signature = OslRequestSignature::from_request(&request(vec![], None));

    assert_eq!(signature.last_message_role, OslMessageRole::Unknown);
    assert!(!signature.has_prior_assistant_message);
    assert!(!signature.has_prior_tool_message);
    assert!(!signature.tools_defined);
}

#[test]
fn request_signature_reads_initial_user_request() {
    let signature = OslRequestSignature::from_request(&request(
        vec![
            Message::System {
                content: MessageContent::Text("system".to_string()),
                name: None,
            },
            Message::User {
                content: MessageContent::Text("question".to_string()),
                name: None,
            },
        ],
        Some(vec![tool_definition()]),
    ));

    assert_eq!(signature.last_message_role, OslMessageRole::User);
    assert!(!signature.has_prior_assistant_message);
    assert!(!signature.has_prior_tool_message);
    assert!(signature.tools_defined);
}

#[test]
fn request_signature_reads_tool_loop_request() {
    let signature = OslRequestSignature::from_request(&request(
        vec![
            Message::User {
                content: MessageContent::Text("question".to_string()),
                name: None,
            },
            Message::Assistant {
                content: None,
                tool_calls: None,
                name: None,
            },
            Message::Tool {
                content: MessageContent::Text("result".to_string()),
                tool_call_id: "call-1".to_string(),
            },
        ],
        Some(vec![tool_definition()]),
    ));

    assert_eq!(signature.last_message_role, OslMessageRole::Tool);
    assert!(signature.has_prior_assistant_message);
    assert!(signature.has_prior_tool_message);
    assert!(signature.tools_defined);
}

#[test]
fn context_storage_key_is_stable_and_length_prefixed() {
    let key = OslContextKey {
        scope: OslContextScope::Workflow {
            agent_id: "agent|with:separators".to_string(),
        },
        model: "model|with:separators".to_string(),
        signature: OslRequestSignature {
            last_message_role: OslMessageRole::Tool,
            has_prior_assistant_message: true,
            has_prior_tool_message: true,
            tools_defined: false,
        },
    }
    .storage_key();

    assert_eq!(
        key,
        "scope=workflow:21:agent|with:separators|model=21:model|with:separators|last=tool|prior_asst=1|prior_tool=1|tools=0"
    );
}

#[test]
fn context_storage_key_distinguishes_scopes() {
    let root_uuid = Uuid::from_u128(42);
    let signature = OslRequestSignature {
        last_message_role: OslMessageRole::User,
        has_prior_assistant_message: false,
        has_prior_tool_message: false,
        tools_defined: true,
    };

    let run_key = OslContextKey {
        scope: OslContextScope::Run { root_uuid },
        model: "model-a".to_string(),
        signature,
    }
    .storage_key();
    let global_key = OslContextKey {
        scope: OslContextScope::Global,
        model: "model-a".to_string(),
        signature,
    }
    .storage_key();

    assert_ne!(run_key, global_key);
    assert!(run_key.starts_with("scope=run:"));
    assert!(global_key.starts_with("scope=global"));
}

#[test]
fn empirical_state_does_not_persist_run_contexts() {
    let mut state = OslEmpiricalState::new("agent-a");
    state
        .contexts
        .insert("persistent".to_string(), Default::default());
    state
        .run_contexts
        .insert("run".to_string(), Default::default());

    let encoded = serde_json::to_value(&state).unwrap();
    assert!(encoded.get("contexts").is_some());
    assert!(encoded.get("run_contexts").is_none());

    let decoded: OslEmpiricalState = serde_json::from_value(encoded).unwrap();
    assert!(decoded.contexts.contains_key("persistent"));
    assert!(decoded.run_contexts.is_empty());
}

#[test]
fn nearest_rank_quantile_uses_exact_integer_rank() {
    let stats = OslContextStats {
        samples: VecDeque::from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        last_updated_at: None,
    };

    assert_eq!(stats.nearest_rank_quantile(50), Some(50));
    assert_eq!(stats.nearest_rank_quantile(85), Some(90));
    assert_eq!(stats.nearest_rank_quantile(90), Some(90));
    assert_eq!(stats.nearest_rank_quantile(100), Some(100));
    assert_eq!(stats.nearest_rank_quantile(0), None);
    assert_eq!(stats.nearest_rank_quantile(101), None);
}

#[test]
fn confidence_requires_min_samples() {
    let stats = OslContextStats {
        samples: VecDeque::from([100, 100, 100]),
        last_updated_at: None,
    };

    assert!(!stats.is_confident(4));
    assert!(stats.is_confident(3));
}

#[test]
fn confidence_uses_exact_p90_p50_spread_boundary() {
    let passing = OslContextStats {
        samples: VecDeque::from([10, 10, 10, 10, 10, 10, 10, 10, 40, 40]),
        last_updated_at: None,
    };
    let failing = OslContextStats {
        samples: VecDeque::from([10, 10, 10, 10, 10, 10, 10, 10, 41, 41]),
        last_updated_at: None,
    };

    assert_eq!(OSL_MAX_SPREAD_MULTIPLIER, 4);
    assert!(passing.is_confident(10));
    assert!(!failing.is_confident(10));
}

#[test]
fn predict_p85_requires_confidence() {
    let confident = OslContextStats {
        samples: VecDeque::from([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        last_updated_at: None,
    };
    let unstable = OslContextStats {
        samples: VecDeque::from([1, 1, 1, 1, 1, 1, 1, 1, 100, 100]),
        last_updated_at: None,
    };

    assert_eq!(confident.predict_p85(10), Some(90));
    assert_eq!(unstable.predict_p85(10), None);
}

#[test]
fn empirical_state_prefers_run_local_prediction_then_persistent_prediction() {
    let root_uuid = Uuid::from_u128(7);
    let req = request(
        vec![Message::User {
            content: MessageContent::Text("question".to_string()),
            name: None,
        }],
        Some(vec![tool_definition()]),
    );
    let signature = OslRequestSignature::from_request(&req);
    let mut state = OslEmpiricalState::new("agent-a");
    let workflow_key = OslContextKey {
        scope: OslContextScope::Workflow {
            agent_id: "agent-a".to_string(),
        },
        model: "model-a".to_string(),
        signature,
    }
    .storage_key();
    state.contexts.insert(
        workflow_key,
        OslContextStats {
            samples: VecDeque::from([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]),
            last_updated_at: None,
        },
    );

    for output_tokens in [10, 20, 30] {
        let call = llm_call(Some(output_tokens), Some(req.clone()));
        assert!(state.observe_run_call(root_uuid, &call));
    }

    let run_outcome = state.predict(Some(root_uuid), "agent-a", "model-a", signature);
    assert_eq!(run_outcome.source, "run_local");
    assert_eq!(run_outcome.emitted_osl, Some(30));
    assert!(run_outcome.confidence_passed);
    assert_eq!(run_outcome.sample_count, Some(3));

    let persistent_outcome = state.predict(None, "agent-a", "model-a", signature);
    assert_eq!(persistent_outcome.source, "workflow");
    assert_eq!(persistent_outcome.emitted_osl, Some(900));
    assert!(persistent_outcome.confidence_passed);

    state.clear_run_contexts(root_uuid);
    assert!(state.run_contexts.is_empty());
    assert_eq!(
        state
            .predict(Some(root_uuid), "agent-a", "model-a", signature)
            .emitted_osl,
        Some(900)
    );
}

#[test]
fn empirical_state_reports_unconfident_and_missing_contexts() {
    let req = request(
        vec![Message::User {
            content: MessageContent::Text("question".to_string()),
            name: None,
        }],
        None,
    );
    let signature = OslRequestSignature::from_request(&req);
    let mut state = OslEmpiricalState::new("agent-a");

    let missing = state.predict(None, "agent-a", "model-a", signature);
    assert_eq!(missing.source, "none");
    assert_eq!(missing.emitted_osl, None);
    assert!(!missing.confidence_passed);
    assert_eq!(missing.sample_count, None);

    state.observe_run_call(Uuid::from_u128(12), &llm_call(Some(100), Some(req)));
    let unconfident = state.predict(Some(Uuid::from_u128(12)), "agent-a", "model-a", signature);
    assert_eq!(unconfident.source, "run_local");
    assert_eq!(unconfident.emitted_osl, None);
    assert!(!unconfident.confidence_passed);
    assert_eq!(unconfident.sample_count, Some(1));
}

#[test]
fn empirical_state_bounds_persistent_and_run_context_counts() {
    let mut state = OslEmpiricalState::new("agent-a");
    let old_time = Utc.timestamp_millis_opt(1000).single().unwrap();
    let new_time = Utc.timestamp_millis_opt(2000).single().unwrap();
    for index in 0..=MAX_PERSISTENT_CONTEXTS {
        state.contexts.insert(
            format!("persistent-{index}"),
            OslContextStats {
                samples: VecDeque::from([index as u32]),
                last_updated_at: Some(old_time),
            },
        );
    }
    state.contexts.insert(
        "newest".to_string(),
        OslContextStats {
            samples: VecDeque::from([999]),
            last_updated_at: Some(new_time),
        },
    );

    let req = request(
        vec![Message::User {
            content: MessageContent::Text("question".to_string()),
            name: None,
        }],
        None,
    );
    let signature = OslRequestSignature::from_request(&req);
    state.observe_context(
        OslContextKey {
            scope: OslContextScope::Workflow {
                agent_id: "agent-a".to_string(),
            },
            model: "model-a".to_string(),
            signature,
        },
        42,
        new_time,
    );
    assert!(state.contexts.len() <= MAX_PERSISTENT_CONTEXTS);
    assert!(state.contexts.contains_key("newest"));

    let call = llm_call(Some(42), Some(req));
    assert!(state.observe_run_call(Uuid::from_u128(1), &call));

    for index in 0..=MAX_RUN_CONTEXTS {
        state.run_contexts.insert(
            format!("run-{index}"),
            OslContextStats {
                samples: VecDeque::from([index as u32]),
                last_updated_at: Some(old_time),
            },
        );
    }
    assert!(state.observe_run_call(Uuid::from_u128(2), &call));
    assert!(state.run_contexts.len() <= MAX_RUN_CONTEXTS);
}

#[test]
fn observe_records_time_and_evicts_oldest_samples() {
    let mut stats = OslContextStats::default();
    let first = Utc.timestamp_millis_opt(1000).single().unwrap();
    let second = Utc.timestamp_millis_opt(2000).single().unwrap();
    let third = Utc.timestamp_millis_opt(3000).single().unwrap();

    stats.observe_with_limit(10, first, 2);
    stats.observe_with_limit(20, second, 2);
    stats.observe_with_limit(30, third, 2);

    assert_eq!(stats.sample_count(), 2);
    assert_eq!(stats.samples, VecDeque::from([20, 30]));
    assert_eq!(stats.last_updated_at, Some(third));
}

#[test]
fn observe_with_zero_limit_retains_no_samples() {
    let mut stats = OslContextStats::default();
    let observed_at = Utc.timestamp_millis_opt(1000).single().unwrap();

    stats.observe_with_limit(10, observed_at, 0);

    assert_eq!(stats.sample_count(), 0);
    assert_eq!(stats.last_updated_at, Some(observed_at));
}

#[tokio::test(flavor = "current_thread")]
async fn empirical_learner_persists_workflow_and_global_contexts() {
    let backend = InMemoryBackend::new();
    let hot_cache = empty_hot_cache();
    let learner = OslEmpiricalLearner::new("agent-a");
    let req = request(
        vec![Message::User {
            content: MessageContent::Text("question".to_string()),
            name: None,
        }],
        Some(vec![tool_definition()]),
    );
    let signature = OslRequestSignature::from_request(&req);
    let run = sample_run(vec![
        llm_call(Some(100), Some(req.clone())),
        llm_call(Some(180), Some(req)),
    ]);

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let state = backend
        .load_osl_empirical_state("agent-a")
        .await
        .unwrap()
        .unwrap();
    let workflow_key = OslContextKey {
        scope: OslContextScope::Workflow {
            agent_id: "agent-a".to_string(),
        },
        model: "model-a".to_string(),
        signature,
    }
    .storage_key();
    let global_key = OslContextKey {
        scope: OslContextScope::Global,
        model: "model-a".to_string(),
        signature,
    }
    .storage_key();

    assert_eq!(
        state.contexts[&workflow_key].samples,
        VecDeque::from([100, 180])
    );
    assert_eq!(
        state.contexts[&global_key].samples,
        VecDeque::from([100, 180])
    );
    assert_eq!(
        hot_cache
            .read()
            .unwrap()
            .osl_empirical
            .as_ref()
            .unwrap()
            .contexts[&workflow_key]
            .samples,
        VecDeque::from([100, 180])
    );
}

#[tokio::test(flavor = "current_thread")]
async fn empirical_learner_skips_invalid_completed_call_samples() {
    let backend = InMemoryBackend::new();
    let hot_cache = empty_hot_cache();
    let learner = OslEmpiricalLearner::new("agent-a");
    let valid_req = request(
        vec![Message::User {
            content: MessageContent::Text("question".to_string()),
            name: None,
        }],
        None,
    );
    let mut no_model_req = valid_req.clone();
    no_model_req.model = None;
    let mut length_call = llm_call(Some(300), Some(valid_req.clone()));
    length_call.finish_reason = Some(FinishReason::Length);

    let run = sample_run(vec![
        llm_call(Some(120), Some(valid_req)),
        llm_call(None, Some(request(vec![], None))),
        llm_call(Some(200), None),
        CallRecord {
            model_name: None,
            ..llm_call(Some(220), Some(no_model_req))
        },
        length_call,
        CallRecord {
            kind: CallKind::Tool,
            ..llm_call(Some(500), Some(request(vec![], None)))
        },
    ]);

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let state = backend
        .load_osl_empirical_state("agent-a")
        .await
        .unwrap()
        .unwrap();
    let mut retained_samples: Vec<u32> = state
        .contexts
        .values()
        .flat_map(|stats| stats.samples.iter().copied())
        .collect();
    retained_samples.sort_unstable();
    assert_eq!(retained_samples, vec![120, 120]);
}
