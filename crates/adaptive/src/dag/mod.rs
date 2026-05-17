// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DAG CPM state and completed-run graph construction.

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{HashMap, HashSet, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::records::{CallKind, CallRecord, RunRecord};

const JUMP_UNIT_MS: f64 = 1000.0;
const DEFAULT_QUEUE_HORIZON_MS: f64 = 3000.0;
const MAX_QUEUE_HORIZON_MS: f64 = 9000.0;

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

/// CPM result for one completed run graph.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CompletedRunCpm {
    /// Run identifier the CPM result was computed from.
    pub(crate) run_id: Uuid,
    /// Workflow finish estimate from the CPM forward pass.
    pub(crate) workflow_finish_ms: f64,
    /// Queue-delay horizon used to convert slack to criticality.
    pub(crate) queue_horizon_ms: f64,
    /// Per-node CPM metrics in the same order as [`CompletedRunDag::nodes`].
    pub(crate) nodes: Vec<RunDagCpmNode>,
}

impl CompletedRunCpm {
    /// Iterate only CPM nodes that correspond to priority-emitting DAG nodes.
    pub(crate) fn priority_nodes<'a>(
        &'a self,
        dag: &'a CompletedRunDag,
    ) -> impl Iterator<Item = (&'a RunDagNode, &'a RunDagCpmNode)> {
        dag.nodes
            .iter()
            .zip(self.nodes.iter())
            .filter(|(node, _)| node.emits_priority)
    }
}

/// CPM metrics for one run-local DAG node.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RunDagCpmNode {
    /// Run-local node index.
    pub(crate) node_index: usize,
    /// Earliest start from the CPM forward pass.
    pub(crate) earliest_start_ms: f64,
    /// Earliest finish from the CPM forward pass.
    pub(crate) earliest_finish_ms: f64,
    /// Latest start from the CPM backward pass.
    pub(crate) latest_start_ms: f64,
    /// Latest finish from the CPM backward pass.
    pub(crate) latest_finish_ms: f64,
    /// CPM slack in milliseconds.
    pub(crate) slack_ms: f64,
    /// Normalized criticality in `[0, 1]`.
    pub(crate) criticality: f64,
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

/// Compute CPM slack and normalized criticality over a completed-run DAG.
///
/// Returns `None` when the graph contains invalid edges or a cycle. The DAG
/// builder should not produce either, so this is a guardrail for future callers.
pub(crate) fn compute_cpm(
    dag: &CompletedRunDag,
    observed_queue_horizon_ms: Option<f64>,
) -> Option<CompletedRunCpm> {
    let queue_horizon_ms = queue_horizon_ms(observed_queue_horizon_ms);
    let (predecessors, successors) = adjacency(dag)?;
    let topology = topological_order(&predecessors, &successors)?;
    let node_count = dag.nodes.len();

    let mut earliest_start_ms = vec![0.0; node_count];
    let mut earliest_finish_ms = vec![0.0; node_count];
    for &node_index in &topology {
        let es = predecessors[node_index]
            .iter()
            .map(|&predecessor| earliest_finish_ms[predecessor])
            .fold(0.0, f64::max);
        earliest_start_ms[node_index] = es;
        earliest_finish_ms[node_index] = es + dag.nodes[node_index].duration_ms;
    }

    let workflow_finish_ms = earliest_finish_ms.iter().copied().fold(0.0, f64::max);
    let mut latest_finish_ms = vec![workflow_finish_ms; node_count];
    let mut latest_start_ms = vec![workflow_finish_ms; node_count];

    for &node_index in topology.iter().rev() {
        let lf = if successors[node_index].is_empty() {
            workflow_finish_ms
        } else {
            successors[node_index]
                .iter()
                .map(|&successor| latest_start_ms[successor])
                .fold(f64::INFINITY, f64::min)
        };
        latest_finish_ms[node_index] = lf;
        latest_start_ms[node_index] = lf - dag.nodes[node_index].duration_ms;
    }

    let nodes = (0..node_count)
        .map(|node_index| {
            let slack_ms = latest_start_ms[node_index] - earliest_start_ms[node_index];
            RunDagCpmNode {
                node_index,
                earliest_start_ms: earliest_start_ms[node_index],
                earliest_finish_ms: earliest_finish_ms[node_index],
                latest_start_ms: latest_start_ms[node_index],
                latest_finish_ms: latest_finish_ms[node_index],
                slack_ms,
                criticality: criticality_from_slack(slack_ms, queue_horizon_ms),
            }
        })
        .collect();

    Some(CompletedRunCpm {
        run_id: dag.run_id,
        workflow_finish_ms,
        queue_horizon_ms,
        nodes,
    })
}

fn queue_horizon_ms(observed_queue_horizon_ms: Option<f64>) -> f64 {
    let horizon = observed_queue_horizon_ms
        .filter(|value| value.is_finite())
        .unwrap_or(DEFAULT_QUEUE_HORIZON_MS);
    horizon.clamp(JUMP_UNIT_MS, MAX_QUEUE_HORIZON_MS)
}

fn criticality_from_slack(slack_ms: f64, queue_horizon_ms: f64) -> f64 {
    1.0 - (slack_ms.max(0.0) / queue_horizon_ms).clamp(0.0, 1.0)
}

fn topological_order(predecessors: &[Vec<usize>], successors: &[Vec<usize>]) -> Option<Vec<usize>> {
    if predecessors.len() != successors.len() {
        return None;
    }
    let node_count = predecessors.len();
    let mut incoming_count = predecessors.iter().map(Vec::len).collect::<Vec<_>>();

    let mut ready: VecDeque<usize> = incoming_count
        .iter()
        .enumerate()
        .filter_map(|(index, &count)| (count == 0).then_some(index))
        .collect();
    let mut order = Vec::with_capacity(node_count);

    while let Some(node_index) = ready.pop_front() {
        order.push(node_index);
        for &successor in &successors[node_index] {
            incoming_count[successor] -= 1;
            if incoming_count[successor] == 0 {
                ready.push_back(successor);
            }
        }
    }

    (order.len() == node_count).then_some(order)
}

fn adjacency(dag: &CompletedRunDag) -> Option<(Vec<Vec<usize>>, Vec<Vec<usize>>)> {
    let mut predecessors = vec![Vec::new(); dag.nodes.len()];
    let mut successors = vec![Vec::new(); dag.nodes.len()];

    for edge in &dag.edges {
        if edge.from >= dag.nodes.len() || edge.to >= dag.nodes.len() {
            return None;
        }
        predecessors[edge.to].push(edge.from);
        successors[edge.from].push(edge.to);
    }

    Some((predecessors, successors))
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
