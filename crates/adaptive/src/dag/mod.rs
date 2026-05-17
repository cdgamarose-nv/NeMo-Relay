// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DAG CPM state and completed-run graph construction.

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{HashMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::records::{CallKind, CallRecord, RunRecord};

/// Learned DAG CPM summary for one agent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DagCpmState {
    /// Agent identifier the state belongs to.
    pub agent_id: String,
    /// Learned per-node summaries keyed by stable structural path.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub nodes: HashMap<String, DagCpmNodeState>,
}

/// Learned CPM statistics for one structural workflow node.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DagCpmNodeState {
    /// Number of observations incorporated into this node.
    pub observation_count: u32,
    /// Exponential moving average of model service duration in milliseconds.
    pub duration_ms_ewma: f64,
    /// Exponential moving average of CPM slack in milliseconds.
    pub slack_ms_ewma: f64,
    /// Exponential moving average of normalized CPM criticality in `[0, 1]`.
    pub criticality_ewma: f64,
    /// Exponential moving average of the queue-delay horizon used for normalization.
    pub queue_horizon_ms_ewma: f64,
    /// Last update timestamp for retention and staleness checks.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_updated_at: Option<DateTime<Utc>>,
}

/// Completed run graph used as the CPM learner input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompletedRunDag {
    /// Run identifier the graph was built from.
    pub(crate) run_id: Uuid,
    /// Completed tool and LLM calls observed in the run.
    pub(crate) nodes: Vec<RunDagNode>,
    /// Dependency edges between completed calls.
    pub(crate) edges: Vec<RunDagEdge>,
}

impl CompletedRunDag {
    /// Iterate only the nodes that may emit backend priority hints.
    pub(crate) fn priority_nodes(&self) -> impl Iterator<Item = &RunDagNode> {
        self.nodes.iter().filter(|node| node.emits_priority)
    }
}

/// One completed tool or LLM call in a run-local DAG.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RunDagNode {
    /// Run-local node index.
    pub(crate) index: usize,
    /// Original call position in the run record.
    pub(crate) call_index: usize,
    /// Run-local call ordinal assigned by the telemetry drain, when available.
    pub(crate) run_call_index: Option<u32>,
    /// Runtime scope UUID for the call.
    pub(crate) scope_uuid: Uuid,
    /// Runtime parent scope UUID, if present.
    pub(crate) parent_uuid: Option<Uuid>,
    /// Tool or LLM call type.
    pub(crate) kind: CallKind,
    /// Runtime call name.
    pub(crate) name: String,
    /// Stable structural key used for learned DAG state lookup.
    pub(crate) structural_key: String,
    /// Observed start timestamp.
    pub(crate) started_at: DateTime<Utc>,
    /// Observed end timestamp.
    pub(crate) ended_at: DateTime<Utc>,
    /// Service duration used by CPM, in milliseconds.
    pub(crate) duration_ms: f64,
    /// Whether this node is allowed to emit a backend priority hint.
    pub(crate) emits_priority: bool,
}

/// Run-local dependency edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct RunDagEdge {
    /// Predecessor node index.
    pub(crate) from: usize,
    /// Successor node index.
    pub(crate) to: usize,
    /// Why the edge was inferred.
    pub(crate) kind: RunDagEdgeKind,
}

/// Source of an inferred run-local dependency edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum RunDagEdgeKind {
    /// A represented parent call completed before this child call started.
    ParentScope,
    /// Calls were observed as non-overlapping phases inside the same parent scope.
    SameParentPhase,
}

/// Build a conservative run-local DAG from one completed run.
///
/// Tool nodes are retained because they consume workflow time and can sit on the
/// critical path. Only LLM nodes are marked as priority-emitting.
pub(crate) fn build_completed_run_dag(run: &RunRecord) -> CompletedRunDag {
    let mut completed_calls: Vec<(usize, &CallRecord, DateTime<Utc>)> = run
        .calls
        .iter()
        .enumerate()
        .filter_map(|(call_index, call)| call.ended_at.map(|ended_at| (call_index, call, ended_at)))
        .collect();

    completed_calls.sort_by(|left, right| {
        let (left_index, left_call, left_ended_at) = left;
        let (right_index, right_call, right_ended_at) = right;
        left_call
            .started_at
            .cmp(&right_call.started_at)
            .then(left_ended_at.cmp(right_ended_at))
            .then(left_index.cmp(right_index))
    });

    let mut nodes = Vec::with_capacity(completed_calls.len());
    for (node_index, (call_index, call, ended_at)) in completed_calls.into_iter().enumerate() {
        nodes.push(RunDagNode {
            index: node_index,
            call_index,
            run_call_index: call.run_call_index,
            scope_uuid: call.scope_uuid,
            parent_uuid: call.parent_uuid,
            kind: call.kind,
            name: call.name.clone(),
            structural_key: structural_key(call),
            started_at: call.started_at,
            ended_at,
            duration_ms: cpm_duration_ms(call, ended_at),
            emits_priority: call.kind == CallKind::Llm,
        });
    }

    let mut edges = Vec::new();
    let mut seen_edges = HashSet::new();
    add_parent_scope_edges(&nodes, &mut edges, &mut seen_edges);
    add_same_parent_phase_edges(&nodes, &mut edges, &mut seen_edges);
    edges.sort_by_key(|edge| (edge.from, edge.to, edge.kind as u8));

    CompletedRunDag {
        run_id: run.id,
        nodes,
        edges,
    }
}

fn structural_key(call: &CallRecord) -> String {
    let kind = match call.kind {
        CallKind::Llm => "llm",
        CallKind::Tool => "tool",
    };
    let path = if call.function_path.is_empty() {
        "root".to_string()
    } else {
        call.function_path.join("/")
    };
    format!("{path}/{kind}:{}", call.name)
}

fn cpm_duration_ms(call: &CallRecord, ended_at: DateTime<Utc>) -> f64 {
    if call.kind == CallKind::Llm
        && let Some(ref timing) = call.backend_timing
        && let Some(total_time_ms) = timing.total_time_ms
    {
        let wait_ms = timing.prefill_wait_time_ms.unwrap_or(0.0);
        let service_ms = total_time_ms - wait_ms;
        if service_ms.is_finite() && service_ms >= 0.0 {
            return service_ms;
        }
    }
    observed_duration_ms(call.started_at, ended_at)
}

fn observed_duration_ms(started_at: DateTime<Utc>, ended_at: DateTime<Utc>) -> f64 {
    ended_at
        .signed_duration_since(started_at)
        .num_milliseconds()
        .max(0) as f64
}

fn add_parent_scope_edges(
    nodes: &[RunDagNode],
    edges: &mut Vec<RunDagEdge>,
    seen_edges: &mut HashSet<(usize, usize, RunDagEdgeKind)>,
) {
    let nodes_by_scope: HashMap<Uuid, usize> = nodes
        .iter()
        .map(|node| (node.scope_uuid, node.index))
        .collect();

    for node in nodes {
        let Some(parent_uuid) = node.parent_uuid else {
            continue;
        };
        let Some(&parent_index) = nodes_by_scope.get(&parent_uuid) else {
            continue;
        };
        let parent = &nodes[parent_index];
        if parent.index == node.index || parent.ended_at > node.started_at {
            continue;
        }
        push_edge(
            edges,
            seen_edges,
            parent.index,
            node.index,
            RunDagEdgeKind::ParentScope,
        );
    }
}

fn add_same_parent_phase_edges(
    nodes: &[RunDagNode],
    edges: &mut Vec<RunDagEdge>,
    seen_edges: &mut HashSet<(usize, usize, RunDagEdgeKind)>,
) {
    let mut groups: HashMap<Uuid, Vec<usize>> = HashMap::new();
    for node in nodes {
        if let Some(parent_uuid) = node.parent_uuid {
            groups.entry(parent_uuid).or_default().push(node.index);
        }
    }

    for mut group in groups.into_values() {
        group.sort_by_key(|&index| (nodes[index].started_at, nodes[index].ended_at, index));
        add_phase_edges_for_group(nodes, &group, edges, seen_edges);
    }
}

fn add_phase_edges_for_group(
    nodes: &[RunDagNode],
    group: &[usize],
    edges: &mut Vec<RunDagEdge>,
    seen_edges: &mut HashSet<(usize, usize, RunDagEdgeKind)>,
) {
    let mut current_phase: Vec<usize> = Vec::new();
    let mut phase_predecessors: Vec<usize> = Vec::new();

    for &node_index in group {
        if current_phase.is_empty() {
            current_phase.push(node_index);
            continue;
        }

        if current_phase
            .iter()
            .all(|&active_index| nodes[active_index].ended_at <= nodes[node_index].started_at)
        {
            phase_predecessors = current_phase;
            for &predecessor in &phase_predecessors {
                push_edge(
                    edges,
                    seen_edges,
                    predecessor,
                    node_index,
                    RunDagEdgeKind::SameParentPhase,
                );
            }
            current_phase = vec![node_index];
        } else {
            for &predecessor in &phase_predecessors {
                push_edge(
                    edges,
                    seen_edges,
                    predecessor,
                    node_index,
                    RunDagEdgeKind::SameParentPhase,
                );
            }
            current_phase.push(node_index);
        }
    }
}

fn push_edge(
    edges: &mut Vec<RunDagEdge>,
    seen_edges: &mut HashSet<(usize, usize, RunDagEdgeKind)>,
    from: usize,
    to: usize,
    kind: RunDagEdgeKind,
) {
    if from == to || !seen_edges.insert((from, to, kind)) {
        return;
    }
    edges.push(RunDagEdge { from, to, kind });
}

#[cfg(test)]
#[path = "../../tests/unit/dag_tests.rs"]
mod tests;
