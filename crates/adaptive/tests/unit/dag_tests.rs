// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::{Arc, RwLock};

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use super::*;
use crate::learner::traits::Learner;
use crate::storage::memory::InMemoryBackend;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::records::{BackendTiming, CallRecord, RunRecord};

fn call(
    kind: CallKind,
    name: &str,
    base: DateTime<Utc>,
    start_ms: i64,
    end_ms: Option<i64>,
    parent_uuid: Option<Uuid>,
) -> CallRecord {
    CallRecord {
        kind,
        name: name.to_string(),
        started_at: base + Duration::milliseconds(start_ms),
        ended_at: end_ms.map(|offset| base + Duration::milliseconds(offset)),
        scope_uuid: Uuid::new_v4(),
        parent_uuid,
        function_path: vec!["agent".to_string()],
        ..CallRecord::default()
    }
}

fn run(calls: Vec<CallRecord>) -> RunRecord {
    let base = calls
        .first()
        .map(|call| call.started_at)
        .unwrap_or_else(Utc::now);
    let ended_at = calls.iter().filter_map(|call| call.ended_at).max();
    RunRecord {
        id: Uuid::now_v7(),
        agent_id: "agent".to_string(),
        calls,
        started_at: base,
        ended_at,
    }
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 1e-9,
        "expected {expected}, got {actual}"
    );
}

fn empty_hot_cache() -> Arc<RwLock<HotCache>> {
    Arc::new(RwLock::new(HotCache {
        plan: None,
        trie: None,
        agent_hints_default: None,
        acg_profiles: std::collections::HashMap::new(),
        acg_profile_observation_counts: std::collections::HashMap::new(),
        acg_stability: None,
        acg_observation_count: 0,
    }))
}

#[test]
fn completed_run_dag_keeps_tools_but_only_llms_emit_priority() {
    let base = Utc::now();
    let parent_uuid = Uuid::new_v4();
    let graph = build_completed_run_dag(&run(vec![
        call(
            CallKind::Llm,
            "planner",
            base,
            0,
            Some(10),
            Some(parent_uuid),
        ),
        call(
            CallKind::Tool,
            "search",
            base,
            10,
            Some(30),
            Some(parent_uuid),
        ),
        call(
            CallKind::Llm,
            "synth",
            base,
            30,
            Some(40),
            Some(parent_uuid),
        ),
    ]));

    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(
        graph
            .priority_nodes()
            .map(|node| node.name.as_str())
            .collect::<Vec<_>>(),
        vec!["planner", "synth"]
    );
    assert!(graph.nodes.iter().any(|node| node.kind == CallKind::Tool));
}

#[test]
fn completed_run_dag_infers_fanout_and_join_edges_by_parent_phase() {
    let base = Utc::now();
    let parent_uuid = Uuid::new_v4();
    let graph = build_completed_run_dag(&run(vec![
        call(
            CallKind::Llm,
            "planner",
            base,
            0,
            Some(10),
            Some(parent_uuid),
        ),
        call(
            CallKind::Tool,
            "fast",
            base,
            10,
            Some(30),
            Some(parent_uuid),
        ),
        call(
            CallKind::Tool,
            "slow",
            base,
            11,
            Some(50),
            Some(parent_uuid),
        ),
        call(
            CallKind::Llm,
            "synth",
            base,
            51,
            Some(70),
            Some(parent_uuid),
        ),
    ]));

    let edge_names = graph
        .edges
        .iter()
        .map(|edge| {
            (
                graph.nodes[edge.from].name.as_str(),
                graph.nodes[edge.to].name.as_str(),
                edge.kind,
            )
        })
        .collect::<Vec<_>>();

    assert!(edge_names.contains(&("planner", "fast", RunDagEdgeKind::SameParentPhase)));
    assert!(edge_names.contains(&("planner", "slow", RunDagEdgeKind::SameParentPhase)));
    assert!(edge_names.contains(&("fast", "synth", RunDagEdgeKind::SameParentPhase)));
    assert!(edge_names.contains(&("slow", "synth", RunDagEdgeKind::SameParentPhase)));
}

#[test]
fn completed_run_dag_uses_llm_backend_service_duration_when_available() {
    let base = Utc::now();
    let mut llm = call(CallKind::Llm, "planner", base, 0, Some(100), None);
    llm.backend_timing = Some(BackendTiming {
        total_time_ms: Some(80.0),
        prefill_wait_time_ms: Some(25.0),
        ..BackendTiming::default()
    });

    let graph = build_completed_run_dag(&run(vec![llm]));

    assert_eq!(graph.nodes[0].duration_ms, 55.0);
}

#[test]
fn completed_run_dag_keeps_run_call_index_separate_from_structural_key() {
    let base = Utc::now();
    let parent_uuid = Uuid::new_v4();
    let mut first = call(
        CallKind::Llm,
        "branch",
        base,
        0,
        Some(50),
        Some(parent_uuid),
    );
    first.run_call_index = Some(3);
    let mut second = call(
        CallKind::Llm,
        "branch",
        base,
        1,
        Some(10),
        Some(parent_uuid),
    );
    second.run_call_index = Some(4);

    let graph = build_completed_run_dag(&run(vec![first, second]));

    assert_eq!(graph.nodes[0].structural_key, "agent/llm:branch");
    assert_eq!(graph.nodes[1].structural_key, "agent/llm:branch");
    assert_eq!(graph.nodes[0].run_call_index, Some(3));
    assert_eq!(graph.nodes[1].run_call_index, Some(4));
}

#[test]
fn completed_run_dag_only_uses_completed_parent_scope_as_dependency() {
    let base = Utc::now();
    let completed_parent = call(CallKind::Tool, "outer", base, 0, Some(10), None);
    let completed_child = call(
        CallKind::Llm,
        "after_outer",
        base,
        11,
        Some(20),
        Some(completed_parent.scope_uuid),
    );
    let overlapping_parent = call(CallKind::Tool, "wrapper", base, 30, Some(80), None);
    let overlapping_child = call(
        CallKind::Llm,
        "inside_wrapper",
        base,
        40,
        Some(60),
        Some(overlapping_parent.scope_uuid),
    );
    let graph = build_completed_run_dag(&run(vec![
        completed_parent,
        completed_child,
        overlapping_parent,
        overlapping_child,
    ]));

    let parent_edges = graph
        .edges
        .iter()
        .filter(|edge| edge.kind == RunDagEdgeKind::ParentScope)
        .map(|edge| {
            (
                graph.nodes[edge.from].name.as_str(),
                graph.nodes[edge.to].name.as_str(),
            )
        })
        .collect::<Vec<_>>();

    assert_eq!(parent_edges, vec![("outer", "after_outer")]);
}

#[test]
fn completed_run_dag_does_not_infer_phase_edges_for_unknown_parent_calls() {
    let base = Utc::now();
    let graph = build_completed_run_dag(&run(vec![
        call(CallKind::Llm, "first", base, 0, Some(10), None),
        call(CallKind::Llm, "second", base, 20, Some(30), None),
    ]));

    assert!(
        graph.edges.is_empty(),
        "unknown parent UUID does not prove shared same-parent execution"
    );
}

#[test]
fn completed_run_dag_skips_incomplete_calls() {
    let base = Utc::now();
    let graph = build_completed_run_dag(&run(vec![
        call(CallKind::Llm, "complete", base, 0, Some(10), None),
        call(CallKind::Tool, "incomplete", base, 10, None, None),
    ]));

    assert_eq!(graph.nodes.len(), 1);
    assert_eq!(graph.nodes[0].name, "complete");
}

#[test]
fn completed_run_cpm_computes_slack_and_criticality_from_fanout_join() {
    let base = Utc::now();
    let parent_uuid = Uuid::new_v4();
    let graph = build_completed_run_dag(&run(vec![
        call(
            CallKind::Llm,
            "planner",
            base,
            0,
            Some(1000),
            Some(parent_uuid),
        ),
        call(
            CallKind::Tool,
            "fast",
            base,
            1000,
            Some(2000),
            Some(parent_uuid),
        ),
        call(
            CallKind::Tool,
            "slow",
            base,
            1000,
            Some(3000),
            Some(parent_uuid),
        ),
        call(
            CallKind::Llm,
            "synth",
            base,
            3000,
            Some(4000),
            Some(parent_uuid),
        ),
    ]));

    let cpm = compute_cpm(&graph, Some(1000.0)).expect("graph is acyclic");

    assert_close(cpm.workflow_finish_ms, 4000.0);
    assert_close(cpm.queue_horizon_ms, 1000.0);

    let by_name = graph
        .nodes
        .iter()
        .zip(cpm.nodes.iter())
        .map(|(node, cpm_node)| (node.name.as_str(), cpm_node))
        .collect::<std::collections::HashMap<_, _>>();

    assert_close(by_name["planner"].slack_ms, 0.0);
    assert_close(by_name["planner"].criticality, 1.0);
    assert_close(by_name["fast"].slack_ms, 1000.0);
    assert_close(by_name["fast"].criticality, 0.0);
    assert_close(by_name["slow"].slack_ms, 0.0);
    assert_close(by_name["slow"].criticality, 1.0);
    assert_close(by_name["synth"].slack_ms, 0.0);
    assert_close(by_name["synth"].criticality, 1.0);
}

#[test]
fn completed_run_cpm_clamps_queue_horizon() {
    let base = Utc::now();
    let graph = build_completed_run_dag(&run(vec![call(
        CallKind::Llm,
        "single",
        base,
        0,
        Some(10),
        None,
    )]));

    assert_close(compute_cpm(&graph, None).unwrap().queue_horizon_ms, 3000.0);
    assert_close(
        compute_cpm(&graph, Some(4.0)).unwrap().queue_horizon_ms,
        1000.0,
    );
    assert_close(
        compute_cpm(&graph, Some(15_000.0))
            .unwrap()
            .queue_horizon_ms,
        9000.0,
    );
    assert_close(
        compute_cpm(&graph, Some(f64::NAN))
            .unwrap()
            .queue_horizon_ms,
        3000.0,
    );
}

#[test]
fn completed_run_cpm_priority_nodes_match_llm_nodes() {
    let base = Utc::now();
    let parent_uuid = Uuid::new_v4();
    let graph = build_completed_run_dag(&run(vec![
        call(
            CallKind::Llm,
            "planner",
            base,
            0,
            Some(10),
            Some(parent_uuid),
        ),
        call(
            CallKind::Tool,
            "search",
            base,
            10,
            Some(20),
            Some(parent_uuid),
        ),
        call(
            CallKind::Llm,
            "synth",
            base,
            20,
            Some(30),
            Some(parent_uuid),
        ),
    ]));
    let cpm = compute_cpm(&graph, None).expect("graph is acyclic");

    assert_eq!(
        cpm.priority_nodes(&graph)
            .map(|(node, _)| node.name.as_str())
            .collect::<Vec<_>>(),
        vec!["planner", "synth"]
    );
}

#[test]
fn completed_run_cpm_returns_none_for_cyclic_graph() {
    let base = Utc::now();
    let mut graph = build_completed_run_dag(&run(vec![
        call(CallKind::Llm, "a", base, 0, Some(10), None),
        call(CallKind::Llm, "b", base, 10, Some(20), None),
    ]));
    graph.edges = vec![
        RunDagEdge {
            from: 0,
            to: 1,
            kind: RunDagEdgeKind::SameParentPhase,
        },
        RunDagEdge {
            from: 1,
            to: 0,
            kind: RunDagEdgeKind::SameParentPhase,
        },
    ];

    assert!(compute_cpm(&graph, None).is_none());
}

#[test]
fn dag_cpm_state_updates_compact_llm_aggregates_only() {
    let base = Utc::now();
    let parent_uuid = Uuid::new_v4();
    let updated_at = base + Duration::milliseconds(5000);

    let mut planner = call(
        CallKind::Llm,
        "planner",
        base,
        0,
        Some(1000),
        Some(parent_uuid),
    );
    planner.backend_timing = Some(BackendTiming {
        prefill_wait_time_ms: Some(1200.0),
        total_time_ms: Some(2200.0),
        ..BackendTiming::default()
    });

    let mut synth = call(
        CallKind::Llm,
        "synth",
        base,
        3000,
        Some(4000),
        Some(parent_uuid),
    );
    synth.backend_timing = Some(BackendTiming {
        prefill_wait_time_ms: Some(2000.0),
        total_time_ms: Some(3000.0),
        ..BackendTiming::default()
    });

    let run = run(vec![
        planner,
        call(
            CallKind::Tool,
            "fast",
            base,
            1000,
            Some(2000),
            Some(parent_uuid),
        ),
        call(
            CallKind::Tool,
            "slow",
            base,
            1000,
            Some(3000),
            Some(parent_uuid),
        ),
        synth,
    ]);

    let mut state = DagCpmState::new("agent");

    assert!(state.update_from_run(&run, updated_at));
    assert_eq!(state.nodes.len(), 2);
    assert!(!state.nodes.contains_key("agent/tool:fast"));
    assert!(!state.nodes.contains_key("agent/tool:slow"));

    let planner = &state.nodes["agent/llm:planner"];
    assert_eq!(planner.observation_count, 1);
    assert_close(planner.duration_ms_ewma, 1000.0);
    assert_close(planner.slack_ms_ewma, 0.0);
    assert_close(planner.criticality_ewma, 1.0);
    assert_close(planner.queue_horizon_ms_ewma, 2000.0);
    assert_eq!(planner.last_updated_at, Some(updated_at));

    let synth = &state.nodes["agent/llm:synth"];
    assert_eq!(synth.observation_count, 1);
    assert_close(synth.duration_ms_ewma, 1000.0);
    assert_close(synth.slack_ms_ewma, 0.0);
    assert_close(synth.criticality_ewma, 1.0);
    assert_close(synth.queue_horizon_ms_ewma, 2000.0);
    assert_eq!(synth.last_updated_at, Some(updated_at));
}

#[test]
fn dag_cpm_state_updates_existing_nodes_with_ewma() {
    let base = Utc::now();
    let mut state = DagCpmState::new("agent");

    assert!(state.update_from_run(
        &run(vec![call(
            CallKind::Llm,
            "single",
            base,
            0,
            Some(1000),
            None,
        )]),
        base + Duration::milliseconds(1000),
    ));
    assert!(state.update_from_run(
        &run(vec![call(
            CallKind::Llm,
            "single",
            base,
            0,
            Some(3000),
            None,
        )]),
        base + Duration::milliseconds(3000),
    ));

    let node = &state.nodes["agent/llm:single"];
    assert_eq!(node.observation_count, 2);
    assert_close(
        node.duration_ms_ewma,
        1000.0 + DAG_CPM_EWMA_ALPHA * (3000.0 - 1000.0),
    );
    assert_close(node.slack_ms_ewma, 0.0);
    assert_close(node.criticality_ewma, 1.0);
    assert_close(node.queue_horizon_ms_ewma, DEFAULT_QUEUE_HORIZON_MS);
}

#[tokio::test(flavor = "current_thread")]
async fn dag_cpm_learner_persists_compact_state() {
    let base = Utc::now();
    let backend = InMemoryBackend::new();
    let hot_cache = empty_hot_cache();
    let learner = DagCpmLearner::new("agent");
    let run = run(vec![call(
        CallKind::Llm,
        "single",
        base,
        0,
        Some(1000),
        None,
    )]);

    learner
        .process_run(&run, &backend, &hot_cache)
        .await
        .unwrap();

    let state = backend.load_dag_state("agent").await.unwrap().unwrap();
    assert_eq!(state.agent_id, "agent");
    assert_eq!(state.nodes.len(), 1);
    assert_eq!(state.nodes["agent/llm:single"].observation_count, 1);
}
