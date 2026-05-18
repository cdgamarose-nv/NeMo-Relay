// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DAG CPM state and completed-run graph construction.

use std::collections::{HashMap, HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Duration, Utc};
use nemo_flow::codec::request::Message;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{AdaptiveError, Result};
use crate::learner::traits::Learner;
use crate::model::model_bucket;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::records::{CallKind, CallRecord, RunRecord};

const JUMP_UNIT_MS: f64 = 1000.0;
const DEFAULT_QUEUE_HORIZON_MS: f64 = 3000.0;
const MAX_QUEUE_HORIZON_MS: f64 = 9000.0;
const DAG_CPM_EWMA_ALPHA: f64 = 0.2;
const DAG_CPM_RETENTION_DAYS: i64 = 14;
const DAG_CPM_MAX_NODES: usize = 512;
const DAG_CPM_MAX_QUEUE_MODELS: usize = 64;
const DAG_CPM_QUEUE_WAIT_WINDOW: usize = 256;
const DAG_CPM_MIN_QUEUE_WAIT_SAMPLES: usize = 4;
pub(crate) const DEFAULT_DAG_PRIORITY_CAP: u32 = 3;
pub(crate) const MAX_DAG_PRIORITY_CAP: u32 = 5;

/// Learned DAG CPM summary for one agent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DagCpmState {
    /// Agent identifier the state belongs to.
    pub agent_id: String,
    /// Learned per-node summaries keyed by stable structural path.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub nodes: HashMap<String, DagCpmNodeState>,
    /// Rolling queue-wait observations by normalized model.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub queue_wait_by_model: HashMap<String, DagQueueWaitWindow>,
    /// Rolling queue-wait observations across all models.
    #[serde(default, skip_serializing_if = "DagQueueWaitWindow::is_empty")]
    pub global_queue_wait: DagQueueWaitWindow,
}

impl DagCpmState {
    /// Create an empty DAG CPM state for one agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            nodes: HashMap::new(),
            queue_wait_by_model: HashMap::new(),
            global_queue_wait: DagQueueWaitWindow::default(),
        }
    }

    /// Update compact aggregates from one completed run.
    ///
    /// Returns `true` when at least one priority-emitting node was incorporated.
    pub(crate) fn update_from_run(&mut self, run: &RunRecord, updated_at: DateTime<Utc>) -> bool {
        let dag = build_completed_run_dag(run);
        let Some(cpm) = self.compute_cpm_for_dag(&dag) else {
            return false;
        };
        let samples = cpm_samples_from(&dag, &cpm);
        let updated = self.update_from_samples(&samples, updated_at);
        let observed_queue = self.observe_queue_waits(run, updated_at);
        let pruned = self.prune(updated_at);
        updated || observed_queue || pruned
    }

    fn update_from_samples(
        &mut self,
        samples: &[DagCpmCallSample],
        updated_at: DateTime<Utc>,
    ) -> bool {
        let mut updated = false;
        for sample in samples {
            if !is_valid_sample(
                sample.duration_ms,
                sample.slack_ms,
                sample.criticality,
                sample.queue_horizon_ms,
            ) {
                continue;
            }
            self.nodes
                .entry(sample.structural_key.clone())
                .or_default()
                .observe(
                    sample.duration_ms,
                    sample.slack_ms,
                    sample.criticality,
                    sample.queue_horizon_ms,
                    updated_at,
                );
            updated = true;
        }
        updated
    }

    fn prune(&mut self, now: DateTime<Utc>) -> bool {
        let cutoff = now - Duration::days(DAG_CPM_RETENTION_DAYS);
        let initial_len = self.nodes.len();
        let initial_models = self.queue_wait_by_model.len();
        self.nodes.retain(|_, node| {
            node.last_updated_at
                .is_none_or(|updated_at| updated_at >= cutoff)
        });
        self.queue_wait_by_model.retain(|_, window| {
            window
                .last_updated_at
                .is_none_or(|updated_at| updated_at >= cutoff)
        });

        prune_queue_wait_models(&mut self.queue_wait_by_model, DAG_CPM_MAX_QUEUE_MODELS);

        let pruned_nodes = prune_nodes(&mut self.nodes, DAG_CPM_MAX_NODES);
        self.nodes.len() != initial_len
            || self.queue_wait_by_model.len() != initial_models
            || pruned_nodes
    }

    fn compute_cpm_for_dag(&self, dag: &CompletedRunDag) -> Option<CompletedRunCpm> {
        let global_horizon = self.queue_horizon_for_model(None);
        compute_cpm_with_horizons(dag, global_horizon, |node| {
            if node.kind == CallKind::Llm {
                self.queue_horizon_for_model(Some(&node.model_bucket))
            } else {
                global_horizon
            }
        })
    }

    fn queue_horizon_for_model(&self, model: Option<&str>) -> f64 {
        model
            .and_then(|model| self.queue_wait_by_model.get(model))
            .and_then(DagQueueWaitWindow::p90_if_enough)
            .or_else(|| self.global_queue_wait.p90_if_enough())
            .unwrap_or(DEFAULT_QUEUE_HORIZON_MS)
            .clamp(JUMP_UNIT_MS, MAX_QUEUE_HORIZON_MS)
    }

    fn observe_queue_waits(&mut self, run: &RunRecord, updated_at: DateTime<Utc>) -> bool {
        let mut updated = false;
        for call in run.calls.iter().filter(|call| call.kind == CallKind::Llm) {
            let Some(wait_ms) = backend_prefill_wait_ms(call) else {
                continue;
            };
            let model = call_model_bucket(call);
            self.global_queue_wait.observe(wait_ms, updated_at);
            self.queue_wait_by_model
                .entry(model)
                .or_default()
                .observe(wait_ms, updated_at);
            updated = true;
        }
        updated
    }
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

/// Bounded rolling queue-wait samples used to estimate scheduler pressure.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DagQueueWaitWindow {
    /// Recent backend queue/admission waits in milliseconds.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub samples_ms: Vec<f64>,
    /// Last update timestamp for retention and pruning.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_updated_at: Option<DateTime<Utc>>,
}

impl DagQueueWaitWindow {
    fn observe(&mut self, wait_ms: f64, updated_at: DateTime<Utc>) {
        if !wait_ms.is_finite() || wait_ms < 0.0 {
            return;
        }
        self.samples_ms.push(wait_ms);
        if self.samples_ms.len() > DAG_CPM_QUEUE_WAIT_WINDOW {
            let remove_count = self.samples_ms.len() - DAG_CPM_QUEUE_WAIT_WINDOW;
            self.samples_ms.drain(0..remove_count);
        }
        self.last_updated_at = Some(updated_at);
    }

    fn p90_if_enough(&self) -> Option<f64> {
        if self.samples_ms.len() < DAG_CPM_MIN_QUEUE_WAIT_SAMPLES {
            return None;
        }
        p90(self.samples_ms.iter().copied())
    }

    fn is_empty(&self) -> bool {
        self.samples_ms.is_empty() && self.last_updated_at.is_none()
    }
}

impl DagCpmNodeState {
    fn observe(
        &mut self,
        duration_ms: f64,
        slack_ms: f64,
        criticality: f64,
        queue_horizon_ms: f64,
        updated_at: DateTime<Utc>,
    ) {
        if self.observation_count == 0 {
            self.duration_ms_ewma = duration_ms;
            self.slack_ms_ewma = slack_ms;
            self.criticality_ewma = criticality;
            self.queue_horizon_ms_ewma = queue_horizon_ms;
        } else {
            self.duration_ms_ewma = ewma(self.duration_ms_ewma, duration_ms);
            self.slack_ms_ewma = ewma(self.slack_ms_ewma, slack_ms);
            self.criticality_ewma = ewma(self.criticality_ewma, criticality);
            self.queue_horizon_ms_ewma = ewma(self.queue_horizon_ms_ewma, queue_horizon_ms);
        }
        self.observation_count = self.observation_count.saturating_add(1);
        self.last_updated_at = Some(updated_at);
    }
}

/// Learner that persists compact DAG CPM aggregates from completed runs.
pub struct DagCpmLearner {
    agent_id: String,
}

impl DagCpmLearner {
    /// Create a new DAG CPM learner for one agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
        }
    }
}

impl Learner for DagCpmLearner {
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut state = backend
                .load_dag_state(&self.agent_id)
                .await?
                .unwrap_or_else(|| DagCpmState::new(&self.agent_id));
            state.agent_id = self.agent_id.clone();

            let updated_at = run.ended_at.unwrap_or_else(Utc::now);
            if state.update_from_run(run, updated_at) {
                backend.store_dag_state(&self.agent_id, &state).await?;
                let mut guard = hot_cache.write().map_err(|error| {
                    AdaptiveError::Internal(format!("hot cache lock poisoned: {error}"))
                })?;
                guard.dag_cpm = Some(state);
            }
            Ok(())
        })
    }
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
    /// Queue-delay horizon used for this node's criticality.
    pub(crate) queue_horizon_ms: f64,
    /// Normalized criticality in `[0, 1]`.
    pub(crate) criticality: f64,
}

/// CPM facts for one priority-emitting LLM call in a completed run.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DagCpmCallSample {
    /// Runtime scope UUID for the original LLM call.
    pub(crate) scope_uuid: Uuid,
    /// Stable structural key used by the hot path.
    pub(crate) structural_key: String,
    /// Model service duration used by CPM.
    pub(crate) duration_ms: f64,
    /// CPM slack for the call.
    pub(crate) slack_ms: f64,
    /// Normalized criticality in `[0, 1]`.
    pub(crate) criticality: f64,
    /// Queue-delay horizon used to compute criticality.
    pub(crate) queue_horizon_ms: f64,
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
    /// Normalized model bucket for pressure lookup.
    pub(crate) model_bucket: String,
    /// Observed start timestamp.
    pub(crate) started_at: DateTime<Utc>,
    /// Observed end timestamp.
    pub(crate) ended_at: DateTime<Utc>,
    /// Service duration used by CPM, in milliseconds.
    pub(crate) duration_ms: f64,
    /// Whether this node is allowed to emit a backend priority hint.
    pub(crate) emits_priority: bool,
    /// Provider tool-call id carried by tool execution events.
    pub(crate) tool_call_id: Option<String>,
    /// Tool-call ids emitted by this LLM response.
    pub(crate) emitted_tool_call_ids: Vec<String>,
    /// Tool-call ids consumed as tool result messages by this LLM request.
    pub(crate) consumed_tool_call_ids: Vec<String>,
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
    /// An LLM request consumed a tool result emitted by an earlier LLM response.
    LlmToolResult,
    /// An LLM response requested a tool call later executed by a tool node.
    LlmToolCall,
    /// A later LLM request consumed a tool result produced by a tool node.
    ToolResult,
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
            model_bucket: call_model_bucket(call),
            started_at: call.started_at,
            ended_at,
            duration_ms: cpm_duration_ms(call, ended_at),
            emits_priority: call.kind == CallKind::Llm,
            tool_call_id: call.tool_call_id.clone(),
            emitted_tool_call_ids: emitted_tool_call_ids(call),
            consumed_tool_call_ids: consumed_tool_call_ids(call),
        });
    }

    let mut edges = Vec::new();
    let mut seen_edges = HashSet::new();
    add_parent_scope_edges(&nodes, &mut edges, &mut seen_edges);
    add_tool_call_edges(&nodes, &mut edges, &mut seen_edges);
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
    compute_cpm_with_horizons(dag, queue_horizon_ms, |_| queue_horizon_ms)
}

fn compute_cpm_with_horizons(
    dag: &CompletedRunDag,
    summary_queue_horizon_ms: f64,
    horizon_for_node: impl Fn(&RunDagNode) -> f64,
) -> Option<CompletedRunCpm> {
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
            let queue_horizon_ms = clamp_queue_horizon(horizon_for_node(&dag.nodes[node_index]));
            RunDagCpmNode {
                node_index,
                earliest_start_ms: earliest_start_ms[node_index],
                earliest_finish_ms: earliest_finish_ms[node_index],
                latest_start_ms: latest_start_ms[node_index],
                latest_finish_ms: latest_finish_ms[node_index],
                slack_ms,
                queue_horizon_ms,
                criticality: criticality_from_slack(slack_ms, queue_horizon_ms),
            }
        })
        .collect();

    Some(CompletedRunCpm {
        run_id: dag.run_id,
        workflow_finish_ms,
        queue_horizon_ms: summary_queue_horizon_ms,
        nodes,
    })
}

/// Compute reusable CPM samples for priority-emitting LLM calls in a run.
pub(crate) fn cpm_call_samples(run: &RunRecord) -> Option<Vec<DagCpmCallSample>> {
    let dag = build_completed_run_dag(run);
    let cpm = compute_cpm(&dag, None)?;
    Some(cpm_samples_from(&dag, &cpm))
}

fn cpm_samples_from(dag: &CompletedRunDag, cpm: &CompletedRunCpm) -> Vec<DagCpmCallSample> {
    cpm.priority_nodes(dag)
        .map(|(node, cpm_node)| DagCpmCallSample {
            scope_uuid: node.scope_uuid,
            structural_key: node.structural_key.clone(),
            duration_ms: node.duration_ms,
            slack_ms: cpm_node.slack_ms,
            criticality: cpm_node.criticality,
            queue_horizon_ms: cpm_node.queue_horizon_ms,
        })
        .collect()
}

fn queue_horizon_ms(observed_queue_horizon_ms: Option<f64>) -> f64 {
    let horizon = observed_queue_horizon_ms
        .filter(|value| value.is_finite())
        .unwrap_or(DEFAULT_QUEUE_HORIZON_MS);
    clamp_queue_horizon(horizon)
}

fn clamp_queue_horizon(horizon: f64) -> f64 {
    if !horizon.is_finite() {
        return DEFAULT_QUEUE_HORIZON_MS;
    }
    horizon.clamp(JUMP_UNIT_MS, MAX_QUEUE_HORIZON_MS)
}

fn criticality_from_slack(slack_ms: f64, queue_horizon_ms: f64) -> f64 {
    1.0 - (slack_ms.max(0.0) / queue_horizon_ms).clamp(0.0, 1.0)
}

fn p90(values: impl Iterator<Item = f64>) -> Option<f64> {
    let mut waits = values
        .filter(|value| value.is_finite() && *value >= 0.0)
        .collect::<Vec<_>>();
    if waits.is_empty() {
        return None;
    }
    waits.sort_by(f64::total_cmp);
    let p90_index = (waits.len() * 9).div_ceil(10).saturating_sub(1);
    waits.get(p90_index).copied()
}

fn is_valid_sample(
    duration_ms: f64,
    slack_ms: f64,
    criticality: f64,
    queue_horizon_ms: f64,
) -> bool {
    duration_ms.is_finite()
        && duration_ms >= 0.0
        && slack_ms.is_finite()
        && criticality.is_finite()
        && (0.0..=1.0).contains(&criticality)
        && queue_horizon_ms.is_finite()
        && queue_horizon_ms > 0.0
}

fn ewma(previous: f64, sample: f64) -> f64 {
    previous + DAG_CPM_EWMA_ALPHA * (sample - previous)
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
    structural_key_parts(kind, &call.function_path, &call.name)
}

fn call_model_bucket(call: &CallRecord) -> String {
    let model_name = call
        .annotated_request
        .as_ref()
        .and_then(|request| request.model.as_deref())
        .or(call.model_name.as_deref())
        .unwrap_or(call.name.as_str());
    model_bucket(model_name)
}

/// Build the learned structural key for a hot-path LLM request.
pub(crate) fn llm_structural_key(function_path: &[String], name: &str) -> String {
    structural_key_parts("llm", function_path, name)
}

/// Project learned CPM criticality onto the bounded backend priority scale.
///
/// `priority_adjustment` is the optional correction layer. It defaults to `0`
/// until a separate adjustment learner is wired.
pub(crate) fn project_priority_prior(
    state: &DagCpmNodeState,
    priority_adjustment: i32,
    priority_cap: u32,
) -> Option<u32> {
    let criticality = state.criticality_ewma;
    if !criticality.is_finite() {
        return None;
    }
    let priority_cap = priority_cap.min(MAX_DAG_PRIORITY_CAP);
    let priority_prior = (priority_cap as f64 * criticality.clamp(0.0, 1.0)).round();
    let priority = priority_prior as i32 + priority_adjustment;
    Some(priority.clamp(0, priority_cap as i32) as u32)
}

fn structural_key_parts(kind: &str, function_path: &[String], name: &str) -> String {
    let path = if function_path.is_empty() {
        "root".to_string()
    } else {
        function_path.join("/")
    };
    format!("{path}/{kind}:{name}")
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

fn emitted_tool_call_ids(call: &CallRecord) -> Vec<String> {
    call.annotated_response
        .as_ref()
        .and_then(|response| response.tool_calls.as_ref())
        .map(|tool_calls| {
            tool_calls
                .iter()
                .map(|tool_call| tool_call.id.clone())
                .collect()
        })
        .unwrap_or_default()
}

fn consumed_tool_call_ids(call: &CallRecord) -> Vec<String> {
    call.annotated_request
        .as_ref()
        .map(|request| {
            request
                .messages
                .iter()
                .filter_map(|message| match message {
                    Message::Tool { tool_call_id, .. } => Some(tool_call_id.clone()),
                    _ => None,
                })
                .collect()
        })
        .unwrap_or_default()
}

fn observed_duration_ms(started_at: DateTime<Utc>, ended_at: DateTime<Utc>) -> f64 {
    ended_at
        .signed_duration_since(started_at)
        .num_milliseconds()
        .max(0) as f64
}

fn backend_prefill_wait_ms(call: &CallRecord) -> Option<f64> {
    call.backend_timing
        .as_ref()
        .and_then(|timing| timing.prefill_wait_time_ms)
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn prune_nodes(nodes: &mut HashMap<String, DagCpmNodeState>, max_nodes: usize) -> bool {
    if nodes.len() <= max_nodes {
        return false;
    }

    let mut entries = nodes
        .iter()
        .map(|(key, node)| (key.clone(), node.last_updated_at, node.observation_count))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(_, last_updated_at, observation_count)| {
        (*last_updated_at, *observation_count)
    });

    let remove_count = nodes.len() - max_nodes;
    for (key, _, _) in entries.into_iter().take(remove_count) {
        nodes.remove(&key);
    }
    true
}

fn prune_queue_wait_models(
    windows: &mut HashMap<String, DagQueueWaitWindow>,
    max_models: usize,
) -> bool {
    if windows.len() <= max_models {
        return false;
    }

    let mut entries = windows
        .iter()
        .map(|(key, window)| (key.clone(), window.last_updated_at, window.samples_ms.len()))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(_, last_updated_at, sample_count)| (*last_updated_at, *sample_count));

    let remove_count = windows.len() - max_models;
    for (key, _, _) in entries.into_iter().take(remove_count) {
        windows.remove(&key);
    }
    true
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

fn add_tool_call_edges(
    nodes: &[RunDagNode],
    edges: &mut Vec<RunDagEdge>,
    seen_edges: &mut HashSet<(usize, usize, RunDagEdgeKind)>,
) {
    let tools_by_call_id = nodes
        .iter()
        .filter_map(|node| {
            let tool_call_id = node.tool_call_id.as_ref()?;
            Some((tool_call_id.as_str(), node.index))
        })
        .collect::<HashMap<_, _>>();
    let llms_by_emitted_tool_call_id = nodes
        .iter()
        .filter(|node| node.kind == CallKind::Llm)
        .flat_map(|node| {
            node.emitted_tool_call_ids
                .iter()
                .map(move |tool_call_id| (tool_call_id.as_str(), node.index))
        })
        .collect::<HashMap<_, _>>();

    for node in nodes {
        for tool_call_id in &node.emitted_tool_call_ids {
            let Some(&tool_index) = tools_by_call_id.get(tool_call_id.as_str()) else {
                continue;
            };
            if node.ended_at <= nodes[tool_index].started_at {
                push_edge(
                    edges,
                    seen_edges,
                    node.index,
                    tool_index,
                    RunDagEdgeKind::LlmToolCall,
                );
            }
        }

        for tool_call_id in &node.consumed_tool_call_ids {
            if node.kind == CallKind::Llm
                && let Some(&llm_index) = llms_by_emitted_tool_call_id.get(tool_call_id.as_str())
                && nodes[llm_index].ended_at <= node.started_at
            {
                push_edge(
                    edges,
                    seen_edges,
                    llm_index,
                    node.index,
                    RunDagEdgeKind::LlmToolResult,
                );
            }

            let Some(&tool_index) = tools_by_call_id.get(tool_call_id.as_str()) else {
                continue;
            };
            if nodes[tool_index].ended_at <= node.started_at {
                push_edge(
                    edges,
                    seen_edges,
                    tool_index,
                    node.index,
                    RunDagEdgeKind::ToolResult,
                );
            }
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
