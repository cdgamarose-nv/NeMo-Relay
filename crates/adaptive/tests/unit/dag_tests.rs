// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use chrono::{DateTime, Duration, Utc};
use uuid::Uuid;

use super::*;
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
