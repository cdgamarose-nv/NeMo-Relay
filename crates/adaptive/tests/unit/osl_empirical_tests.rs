// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;

use chrono::{TimeZone, Utc};
use nemo_flow::codec::request::{
    AnnotatedLlmRequest, FunctionDefinition, Message, MessageContent, ToolDefinition,
};
use uuid::Uuid;

use super::{
    OSL_MAX_SPREAD_MULTIPLIER, OslContextKey, OslContextScope, OslContextStats, OslEmpiricalState,
    OslMessageRole, OslRequestSignature,
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
