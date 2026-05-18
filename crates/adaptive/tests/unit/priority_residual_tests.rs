// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use chrono::{Duration, Utc};
use uuid::Uuid;

use super::*;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::records::{BackendTiming, CallAdaptiveHints, CallKind, CallRecord, RunRecord};

fn empty_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        dag_cpm: None,
        priority_residual: None,
        acg_profiles: HashMap::new(),
        acg_profile_observation_counts: HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

#[test]
fn residual_decision_uses_observed_pressure_and_criticality() {
    let now = Utc::now();
    let mut state = PriorityResidualState::new("agent");
    assert!(state.decision(0.9, "nvidia/Nemotron").is_none());

    assert!(state.observe_pressure("nvidia/Nemotron", 1200.0, now));
    let decision = state.decision(0.9, "nvidia/Nemotron").unwrap();
    assert_eq!(
        decision.key,
        "crit:high|pressure:high|model:nvidia-nemotron"
    );
    assert_eq!(decision.action, PriorityResidualAction::Noop);

    assert!(state.observe_loss(&decision.key, decision.action.arm(), 2.0, now));
    let decision = state.decision(0.9, "nvidia/Nemotron").unwrap();
    assert_eq!(decision.action, PriorityResidualAction::Up);
}

#[test]
fn residual_decision_uses_ucb_exploration_after_initial_samples() {
    let now = Utc::now();
    let mut state = PriorityResidualState::new("agent");
    assert!(state.observe_pressure("model", 1200.0, now));
    let key = "crit:high|pressure:high|model:model".to_string();
    let mut context = PriorityResidualContextState {
        total_pulls: 201,
        last_updated_at: Some(now),
        ..PriorityResidualContextState::default()
    };
    context.arms[PriorityResidualAction::Down.arm() as usize] = PriorityResidualArmState {
        pulls: 100,
        effective_pulls: 100.0,
        mean_loss: 1.0,
    };
    context.arms[PriorityResidualAction::Noop.arm() as usize] = PriorityResidualArmState {
        pulls: 100,
        effective_pulls: 100.0,
        mean_loss: 0.9,
    };
    context.arms[PriorityResidualAction::Up.arm() as usize] = PriorityResidualArmState {
        pulls: 1,
        effective_pulls: 1.0,
        mean_loss: 0.92,
    };
    state.contexts.insert(key, context);

    let decision = state.decision(0.9, "model").unwrap();

    assert_eq!(decision.action, PriorityResidualAction::Up);
}

#[test]
fn residual_loss_charges_only_queue_wait_beyond_slack_plus_priority_spend() {
    let loss = priority_residual_loss(3000.0, 5000.0, 3, 5).unwrap();
    assert!((loss - 0.03).abs() < 1e-9);

    let loss = priority_residual_loss(3000.0, 500.0, 3, 5).unwrap();
    assert!((loss - 2.53).abs() < 1e-9);
}

#[tokio::test(flavor = "current_thread")]
async fn priority_residual_learner_persists_real_dynamo_timing_feedback() {
    let now = Utc::now();
    let call_id = Uuid::new_v4();
    let run = RunRecord {
        id: Uuid::new_v4(),
        agent_id: "agent".to_string(),
        started_at: now,
        ended_at: Some(now + Duration::seconds(2)),
        calls: vec![CallRecord {
            kind: CallKind::Llm,
            name: "nemotron".to_string(),
            started_at: now,
            ended_at: Some(now + Duration::seconds(2)),
            scope_uuid: call_id,
            backend_timing: Some(BackendTiming {
                prefill_wait_time_ms: Some(1200.0),
                total_time_ms: Some(2000.0),
                ..BackendTiming::default()
            }),
            adaptive_hints: Some(CallAdaptiveHints {
                selected_priority_residual_arm: Some(PriorityResidualAction::Noop.arm()),
                selected_priority_residual_key: Some(
                    "crit:high|pressure:high|model:nemotron".to_string(),
                ),
                emitted_priority: Some(3),
                priority_cap: Some(5),
                ..CallAdaptiveHints::default()
            }),
            ..CallRecord::default()
        }],
    };
    let backend = InMemoryBackend::new();
    let learner = PriorityResidualLearner::new("agent");
    let hot_cache = empty_hot_cache();

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let state = backend
        .load_priority_residual_state("agent")
        .await
        .unwrap()
        .unwrap();
    let context = &state.contexts["crit:high|pressure:high|model:nemotron"];
    let arm = context.arms[PriorityResidualAction::Noop.arm() as usize];
    assert_eq!(arm.pulls, 1);
    assert!((arm.mean_loss - 1.23).abs() < 1e-9);
    assert_eq!(
        hot_cache
            .read()
            .unwrap()
            .priority_residual
            .as_ref()
            .unwrap()
            .pressure_by_model["nemotron"]
            .prefill_wait_time_ms,
        1200.0
    );
}
