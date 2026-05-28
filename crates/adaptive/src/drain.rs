// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Background drain task for async telemetry processing.

use std::collections::HashMap;
use std::sync::{
    Arc, RwLock,
    atomic::{AtomicUsize, Ordering},
};

use nemo_flow::api::event::{Event, ScopeCategory};
use nemo_flow::api::scope::ScopeType;
use uuid::Uuid;

use crate::learner::traits::Learner;
use crate::scope_metadata::scope_graph_metadata;
use crate::storage::traits::StorageBackendDyn;
use crate::subscriber::{event_to_call_record, is_run_boundary};
use crate::types::cache::HotCache;
use crate::types::records::{BackendTiming, CallKind, CallRecord, GraphCallContext, RunRecord};

const MAX_SCOPE_FACT_DEPTH: usize = 64;

#[derive(Debug, Clone, Copy)]
struct CallLocator {
    root_uuid: Uuid,
    call_index: usize,
}

#[derive(Debug, Default)]
struct RunEventOutcome {
    completed_run: Option<RunRecord>,
    completed_llm_call: Option<CompletedRunCall>,
    finished_root_uuid: Option<Uuid>,
}

#[derive(Debug, Clone)]
struct CompletedRunCall {
    root_uuid: Uuid,
    call: CallRecord,
}

#[derive(Debug, Clone)]
struct ScopeFacts {
    parent_uuid: Option<Uuid>,
    name: String,
    is_graph_scope: bool,
    is_graph_node: bool,
    graph_node_name: Option<String>,
    graph_task_id: Option<String>,
    graph_depends_on_task_ids: Vec<String>,
}

pub(crate) struct RunAccumulator {
    agent_id: String,
    open_runs: HashMap<Uuid, RunRecord>,
    event_roots: HashMap<Uuid, Uuid>,
    open_call_index: HashMap<Uuid, CallLocator>,
    scope_facts: HashMap<Uuid, ScopeFacts>,
}

impl RunAccumulator {
    pub(crate) fn new(agent_id: String) -> Self {
        Self {
            agent_id,
            open_runs: HashMap::new(),
            event_roots: HashMap::new(),
            open_call_index: HashMap::new(),
            scope_facts: HashMap::new(),
        }
    }

    #[cfg(test)]
    pub(crate) fn open_run_count(&self) -> usize {
        self.open_runs.len()
    }

    #[cfg(test)]
    pub(crate) fn process_event(
        &mut self,
        event: &Event,
        scope_path: &[String],
    ) -> Option<RunRecord> {
        self.process_event_outcome(event, scope_path).completed_run
    }

    fn process_event_outcome(&mut self, event: &Event, scope_path: &[String]) -> RunEventOutcome {
        if let Some(boundary_result) = self.process_run_boundary(event) {
            return boundary_result;
        }

        match (event.scope_category(), event.scope_type()) {
            (Some(ScopeCategory::Start), Some(ScopeType::Tool | ScopeType::Llm)) => {
                self.track_call_start(event, scope_path);
                RunEventOutcome::default()
            }
            (Some(ScopeCategory::End), Some(ScopeType::Tool | ScopeType::Llm)) => RunEventOutcome {
                completed_llm_call: self.track_call_end(event),
                ..RunEventOutcome::default()
            },
            (Some(ScopeCategory::Start), Some(_)) => {
                self.track_nested_scope_start(event);
                RunEventOutcome::default()
            }
            (Some(ScopeCategory::End), Some(_)) => {
                self.track_nested_scope_end(event);
                RunEventOutcome::default()
            }
            _ => RunEventOutcome::default(),
        }
    }

    fn process_run_boundary(&mut self, event: &Event) -> Option<RunEventOutcome> {
        if !is_run_boundary(event) {
            return None;
        }

        if event.scope_category() == Some(ScopeCategory::Start) {
            self.start_run(event);
            return Some(RunEventOutcome::default());
        }

        let (root_uuid, completed_run) = self.finish_run(event)?;
        Some(RunEventOutcome {
            completed_run: Some(completed_run),
            finished_root_uuid: Some(root_uuid),
            ..RunEventOutcome::default()
        })
    }

    fn start_run(&mut self, event: &Event) {
        let root_uuid = event.uuid();
        self.event_roots.insert(root_uuid, root_uuid);
        self.scope_facts.insert(root_uuid, scope_facts(event));
        let run = RunRecord {
            id: Uuid::now_v7(),
            agent_id: self.agent_id.clone(),
            calls: vec![],
            started_at: *event.timestamp(),
            ended_at: None,
        };
        self.open_runs.insert(root_uuid, run);
    }

    fn finish_run(&mut self, event: &Event) -> Option<(Uuid, RunRecord)> {
        let root_uuid = self
            .event_roots
            .remove(&event.uuid())
            .unwrap_or_else(|| event.uuid());
        self.open_call_index
            .retain(|_, locator| locator.root_uuid != root_uuid);
        self.event_roots
            .retain(|_, mapped_root| *mapped_root != root_uuid);
        self.scope_facts
            .retain(|scope_uuid, _| self.event_roots.contains_key(scope_uuid));
        let mut run = self.open_runs.remove(&root_uuid)?;
        run.ended_at = Some(*event.timestamp());
        Some((root_uuid, run))
    }

    fn track_nested_scope_start(&mut self, event: &Event) -> Option<()> {
        let root_uuid = self.infer_root_uuid(event)?;
        self.event_roots.insert(event.uuid(), root_uuid);
        self.scope_facts.insert(event.uuid(), scope_facts(event));
        Some(())
    }

    fn track_nested_scope_end(&mut self, event: &Event) {
        self.event_roots.remove(&event.uuid());
        self.scope_facts.remove(&event.uuid());
    }

    fn track_call_start(&mut self, event: &Event, scope_path: &[String]) -> Option<()> {
        let root_uuid = self.infer_root_uuid(event)?;
        self.event_roots.insert(event.uuid(), root_uuid);
        let graph = self.graph_call_context(event.parent_uuid());
        if let Some(mut record) = event_to_call_record(event, scope_path)
            && let Some(run) = self.open_runs.get_mut(&root_uuid)
        {
            let call_index = run.calls.len();
            record.parent_uuid = event.parent_uuid();
            record.graph = graph;
            record.run_call_index = Some(call_index as u32 + 1);
            run.calls.push(record);
            self.open_call_index.insert(
                event.uuid(),
                CallLocator {
                    root_uuid,
                    call_index,
                },
            );
        }
        Some(())
    }

    fn track_call_end(&mut self, event: &Event) -> Option<CompletedRunCall> {
        let root_uuid = self.infer_root_uuid(event)?;
        let mut completed_call = None;
        if let Some(locator) = self.open_call_index.remove(&event.uuid())
            && let Some(call) = self.call_mut(locator)
        {
            call.ended_at = Some(*event.timestamp());
            apply_llm_end_metadata(call, event);
            if call.kind == CallKind::Llm {
                completed_call = Some(call.clone());
            }
        } else if let Some(run) = self.open_runs.get_mut(&root_uuid)
            && let Some(call) = find_open_call(run, event.name())
        {
            call.ended_at = Some(*event.timestamp());
            apply_llm_end_metadata(call, event);
            if call.kind == CallKind::Llm {
                completed_call = Some(call.clone());
            }
        }
        self.event_roots.remove(&event.uuid());
        completed_call.map(|call| CompletedRunCall { root_uuid, call })
    }

    fn call_mut(&mut self, locator: CallLocator) -> Option<&mut CallRecord> {
        self.open_runs
            .get_mut(&locator.root_uuid)
            .and_then(|run| run.calls.get_mut(locator.call_index))
    }

    fn infer_root_uuid(&self, event: &Event) -> Option<Uuid> {
        self.event_roots.get(&event.uuid()).copied().or_else(|| {
            event
                .parent_uuid()
                .and_then(|parent_uuid| self.event_roots.get(&parent_uuid).copied())
        })
    }

    fn graph_call_context(&self, parent_uuid: Option<Uuid>) -> Option<GraphCallContext> {
        let mut current_uuid = parent_uuid?;
        let mut graph_name = None;
        let mut node_name = None;
        let mut task_id = None;
        let mut depends_on_task_ids = Vec::new();

        for _ in 0..MAX_SCOPE_FACT_DEPTH {
            let facts = self.scope_facts.get(&current_uuid)?;
            if graph_name.is_none() && facts.is_graph_scope {
                graph_name = Some(facts.name.clone());
            }
            if task_id.is_none() && facts.is_graph_node {
                task_id = facts.graph_task_id.clone();
                depends_on_task_ids = facts.graph_depends_on_task_ids.clone();
                node_name = facts
                    .graph_node_name
                    .clone()
                    .or_else(|| Some(facts.name.clone()));
            }
            if graph_name.is_some() && task_id.is_some() && node_name.is_some() {
                break;
            }
            let Some(parent_uuid) = facts.parent_uuid else {
                break;
            };
            current_uuid = parent_uuid;
        }

        Some(GraphCallContext {
            graph_name,
            node_name: node_name?,
            task_id: task_id?,
            depends_on_task_ids,
        })
    }
}

fn scope_facts(event: &Event) -> ScopeFacts {
    let graph = scope_graph_metadata(event);
    ScopeFacts {
        parent_uuid: event.parent_uuid(),
        name: event.name().to_string(),
        is_graph_scope: graph.is_graph_scope,
        is_graph_node: graph.is_graph_node,
        graph_node_name: graph.node_name,
        graph_task_id: graph.task_id,
        graph_depends_on_task_ids: graph.depends_on_task_ids,
    }
}

fn find_open_call<'a>(run: &'a mut RunRecord, event_name: &str) -> Option<&'a mut CallRecord> {
    run.calls
        .iter_mut()
        .rev()
        .find(|call| call.name == event_name && call.ended_at.is_none())
}

fn apply_llm_end_metadata(call: &mut CallRecord, event: &Event) {
    if event.category().map(|category| category.as_str()) != Some("llm") {
        return;
    }
    call.annotated_response = event.annotated_response().cloned();
    let Some(ref annotated) = call.annotated_response else {
        return;
    };

    if let Some(ref usage) = annotated.usage {
        call.output_tokens = usage.completion_tokens.map(|tokens| tokens as u32);
        call.prompt_tokens = usage.prompt_tokens.map(|tokens| tokens as u32);
        call.total_tokens = usage.total_tokens.map(|tokens| tokens as u32);
    }
    call.model_name = annotated.model.clone();
    call.tool_call_count = annotated
        .tool_calls
        .as_ref()
        .map(|calls| calls.len() as u32);
    call.finish_reason = annotated.finish_reason.clone();
    call.backend_timing = backend_timing_from_response_extra(&annotated.extra);
}

fn backend_timing_from_response_extra(
    extra: &serde_json::Map<String, serde_json::Value>,
) -> Option<BackendTiming> {
    let timing = extra
        .get("nvext")
        .and_then(serde_json::Value::as_object)
        .and_then(|nvext| nvext.get("timing"))
        .and_then(serde_json::Value::as_object)?;
    let backend_timing = BackendTiming {
        prefill_wait_time_ms: read_nonnegative_f64(timing.get("prefill_wait_time_ms")),
        prefill_time_ms: read_nonnegative_f64(timing.get("prefill_time_ms")),
        ttft_ms: read_nonnegative_f64(timing.get("ttft_ms")),
        total_time_ms: read_nonnegative_f64(timing.get("total_time_ms")),
        router_queue_depth: read_nonnegative_f64(timing.get("router_queue_depth")),
    };
    (!backend_timing.is_empty()).then_some(backend_timing)
}

fn read_nonnegative_f64(value: Option<&serde_json::Value>) -> Option<f64> {
    let ms = value?.as_f64()?;
    ms.is_finite().then_some(ms.max(0.0))
}

async fn store_run(
    backend: &Arc<dyn StorageBackendDyn + Send + Sync>,
    completed_run: &RunRecord,
) -> bool {
    if let Err(error) = backend.store_run_dyn(completed_run).await {
        eprintln!("nemo-flow-adaptive drain: store_run failed: {error}");
        return false;
    }
    true
}

async fn run_learners(
    learners: &[Box<dyn Learner>],
    completed_run: &RunRecord,
    backend: &Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: &Arc<RwLock<HotCache>>,
) {
    for learner in learners {
        if let Err(error) = learner
            .process_run(completed_run, backend.as_ref(), hot_cache)
            .await
        {
            eprintln!("nemo-flow-adaptive drain: learner failed: {error}");
        }
    }
}

async fn refresh_hot_cache_plan(
    backend: &Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: &Arc<RwLock<HotCache>>,
    agent_id: &str,
) {
    match backend.load_plan_dyn(agent_id).await {
        Ok(plan) => {
            if let Ok(mut guard) = hot_cache.write() {
                guard.plan = plan;
            }
        }
        Err(error) => eprintln!("nemo-flow-adaptive drain: load_plan failed: {error}"),
    }
}

fn observe_run_local_osl(
    hot_cache: &Arc<RwLock<HotCache>>,
    root_uuid: Uuid,
    completed_call: &CallRecord,
) {
    if let Ok(mut guard) = hot_cache.write()
        && let Some(state) = guard.osl_empirical.as_mut()
    {
        state.observe_run_call(root_uuid, completed_call);
    }
}

fn clear_run_local_osl(hot_cache: &Arc<RwLock<HotCache>>, root_uuid: Uuid) {
    if let Ok(mut guard) = hot_cache.write()
        && let Some(state) = guard.osl_empirical.as_mut()
    {
        state.clear_run_contexts(root_uuid);
    }
}

/// Background task that drains events from the telemetry channel, accumulates
/// them into [`RunRecord`]s, stores completed runs, and refreshes the hot cache.
///
/// Exits cleanly when the channel sender is dropped (adaptive shutting down).
///
/// Convenience wrapper around [`drain_task_with_counter`] used by tests. The
/// adaptive runtime spawns `drain_task_with_counter` directly so it can observe
/// the in-flight event counter.
#[cfg(test)]
pub(crate) async fn drain_task(
    rx: tokio::sync::mpsc::UnboundedReceiver<(Event, Vec<String>)>,
    backend: Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: Arc<RwLock<HotCache>>,
    agent_id: String,
    learners: Vec<Box<dyn Learner>>,
) {
    drain_task_with_counter(
        rx,
        backend,
        hot_cache,
        Arc::new(AtomicUsize::new(0)),
        agent_id,
        learners,
    )
    .await;
}

pub(crate) async fn drain_task_with_counter(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<(Event, Vec<String>)>,
    backend: Arc<dyn StorageBackendDyn + Send + Sync>,
    hot_cache: Arc<RwLock<HotCache>>,
    pending_events: Arc<AtomicUsize>,
    agent_id: String,
    learners: Vec<Box<dyn Learner>>,
) {
    let mut accumulator = RunAccumulator::new(agent_id.clone());

    while let Some((event, scope_path)) = rx.recv().await {
        let outcome = accumulator.process_event_outcome(&event, &scope_path);

        if let Some(completed_call) = outcome.completed_llm_call.as_ref() {
            observe_run_local_osl(&hot_cache, completed_call.root_uuid, &completed_call.call);
        }

        if let Some(completed_run) = outcome.completed_run {
            if !store_run(&backend, &completed_run).await {
                if let Some(root_uuid) = outcome.finished_root_uuid {
                    clear_run_local_osl(&hot_cache, root_uuid);
                }
                pending_events.fetch_sub(1, Ordering::SeqCst);
                continue;
            }

            run_learners(&learners, &completed_run, &backend, &hot_cache).await;
            refresh_hot_cache_plan(&backend, &hot_cache, &agent_id).await;
            if let Some(root_uuid) = outcome.finished_root_uuid {
                clear_run_local_osl(&hot_cache, root_uuid);
            }
        }
        pending_events.fetch_sub(1, Ordering::SeqCst);
    }
}

#[cfg(test)]
#[path = "../tests/unit/drain_tests.rs"]
mod tests;
