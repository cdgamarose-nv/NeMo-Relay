// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Context helpers for reading scope metadata on the intercept hot path.
//!
//! These functions read from the NeMo Flow scope stack (via [`current_scope_stack`])
//! to extract information needed by the LLM request intercept:
//!
//! - [`extract_scope_path`]: collects function names from the scope stack for trie lookup
//! - [`extract_graph_call_context`]: reads generic graph/node metadata for DAG CPM lookup
//! - [`read_manual_latency_sensitivity`]: walks all scopes for manual `latency_sensitive` annotations
//! - [`read_workflow_class`]: reads workflow scheduling class metadata for priority caps
//! - [`resolve_agent_id`]: returns the first Agent scope name from the scope stack
//! - [`resolve_root_scope_uuid`]: returns the current execution tree root UUID
//! - [`resolve_run_boundary_scope_uuid`]: returns the active adaptive run UUID
//!
//! All functions are safe to call from sync contexts (intercepts are sync closures).
//! They acquire a read lock on the scope stack, which is always fast.
//!
//! # Metadata Convention
//!
//! Manual latency sensitivity is stored in scope metadata under the JSON path
//! `/nemo_flow_adaptive/latency_sensitivity` as a positive integer.
//!
//! Workflow class is stored in scope metadata under
//! `/nemo_flow_adaptive/workflow_class` or the legacy
//! `/nemo_flow_adaptive/scheduling_class` as `"interactive"`, `"standard"`, or
//! `"background"`.

use nemo_flow::api::runtime::current_scope_stack;
use nemo_flow::api::scope::ScopeType;
use uuid::Uuid;

use crate::scope_metadata::scope_graph_metadata_from_value;
use crate::types::records::GraphCallContext;

/// Metadata key path for manual latency sensitivity annotation.
pub const LATENCY_SENSITIVITY_POINTER: &str = "/nemo_flow_adaptive/latency_sensitivity";
/// Metadata key path for workflow class annotation.
pub const WORKFLOW_CLASS_POINTER: &str = "/nemo_flow_adaptive/workflow_class";
/// Metadata key path for legacy workflow scheduling class annotation.
pub const SCHEDULING_CLASS_POINTER: &str = "/nemo_flow_adaptive/scheduling_class";
/// Metadata key path for adaptive run-boundary override.
pub const RUN_BOUNDARY_POINTER: &str = "/nemo_flow.run_boundary";

/// Workflow-level scheduling class visible on the current scope stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkflowClass {
    /// User-visible request expected to return quickly.
    Interactive,
    /// Ordinary workflow with no special latency class.
    Standard,
    /// Background workflow.
    Background,
}

impl WorkflowClass {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "interactive" => Some(Self::Interactive),
            "standard" => Some(Self::Standard),
            "background" => Some(Self::Background),
            _ => None,
        }
    }
}

/// Session-local scope identity used to coordinate warm-first cohorts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedParentScopeIdentity {
    /// UUID of the root scope for the current execution tree.
    pub root_uuid: Uuid,
    /// UUID of the parent scope shared by sibling fan-out work.
    pub shared_parent_uuid: Uuid,
}

/// Extracts the current function call path from the NeMo Flow scope stack.
///
/// Walks all scopes from root to top, skipping the root scope (index 0),
/// and collects names of Agent and Function scopes. This path is used
/// for prediction trie lookup.
///
/// # Returns
/// A vector of scope names from the current Agent and Function scope path.
/// Returns an empty vector when the scope stack cannot be read safely.
///
/// # Notes
/// The implicit root scope is always skipped.
pub fn extract_scope_path() -> Vec<String> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return vec![],
    };
    stack
        .scopes()
        .iter()
        .skip(1) // skip root
        .filter(|s| matches!(s.scope_type, ScopeType::Agent | ScopeType::Function))
        .map(|s| s.name.clone())
        .collect()
}

/// Extracts compact graph context for the current LLM call, when present.
///
/// This reads generic `nemo_flow.graph.*` scope metadata emitted by framework
/// integrations. The returned context matches completed-run call records:
/// graph name is optional, while node name and task id are required.
pub(crate) fn extract_graph_call_context() -> Option<GraphCallContext> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };

    let mut graph_name = None;
    let mut node_name = None;
    let mut task_id = None;
    let mut depends_on_task_ids = Vec::new();
    for scope in stack.scopes().iter().rev() {
        let graph = scope_graph_metadata_from_value(&scope.name, scope.metadata.as_ref());
        if task_id.is_none() && graph.is_graph_node {
            task_id = graph.task_id;
            node_name = graph.node_name;
            depends_on_task_ids = graph.depends_on_task_ids;
        }
        if graph_name.is_none() && graph.is_graph_scope {
            graph_name = Some(scope.name.clone());
        }
        if graph_name.is_some() && task_id.is_some() && node_name.is_some() {
            break;
        }
    }

    Some(GraphCallContext {
        graph_name,
        node_name: node_name?,
        task_id: task_id?,
        depends_on_task_ids,
    })
}

/// Reads the maximum manual latency sensitivity from all scopes in the current scope stack.
///
/// Walks all scopes and checks metadata for `/nemo_flow_adaptive/latency_sensitivity`.
/// Uses max-merge semantics: if multiple scopes have annotations, the highest wins.
///
/// # Returns
/// The highest manual latency sensitivity annotation visible on the current
/// scope stack, or `None` when no annotation exists.
///
/// # Notes
/// Returns `None` when the scope stack cannot be read safely.
pub fn read_manual_latency_sensitivity() -> Option<u32> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };
    let mut max_val: Option<u32> = None;
    for scope in stack.scopes() {
        if let Some(ref meta) = scope.metadata
            && let Some(val) = meta
                .pointer(LATENCY_SENSITIVITY_POINTER)
                .and_then(|v| v.as_u64())
        {
            let val = val as u32;
            max_val = Some(max_val.map_or(val, |prev: u32| prev.max(val)));
        }
    }
    max_val
}

/// Reads the nearest workflow class from visible scope metadata.
///
/// The deepest scope wins, which lets a workflow mark a nested branch as
/// background or interactive without changing the root workflow metadata.
pub fn read_workflow_class() -> Option<WorkflowClass> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };

    let mut workflow_class = None;
    for scope in stack.scopes() {
        if let Some(ref meta) = scope.metadata {
            let parsed = meta
                .pointer(WORKFLOW_CLASS_POINTER)
                .or_else(|| meta.pointer(SCHEDULING_CLASS_POINTER))
                .and_then(|value| value.as_str())
                .and_then(WorkflowClass::parse);
            if parsed.is_some() {
                workflow_class = parsed;
            }
        }
    }
    workflow_class
}

/// Sets latency sensitivity on the current (top) scope using max-merge semantics.
///
/// If the current scope already has a latency_sensitivity value, the new value
/// is only applied if it is greater than the existing one.
///
/// # Parameters
/// - `value`: New non-negative latency sensitivity hint (`>= 0`) for the
///   current top scope.
///
/// # Returns
/// `Ok(())` when the current scope metadata has been updated or left unchanged.
///
/// # Errors
/// Returns an error string when the scope stack lock is poisoned.
///
/// # Notes
/// Existing non-negative latency sensitivity values are updated using
/// max-merge semantics.
pub fn set_latency_sensitivity(value: u32) -> std::result::Result<(), String> {
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle
        .write()
        .map_err(|e| format!("scope stack lock poisoned: {e}"))?;
    let scope = stack.top_mut();

    let existing = scope
        .metadata
        .as_ref()
        .and_then(|m| m.pointer(LATENCY_SENSITIVITY_POINTER))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32);

    let effective = match existing {
        Some(prev) if prev >= value => return Ok(()),
        _ => value,
    };

    let meta = scope.metadata.get_or_insert_with(|| serde_json::json!({}));
    if let Some(obj) = meta.as_object_mut() {
        let nemo_flow_adaptive = obj
            .entry("nemo_flow_adaptive")
            .or_insert_with(|| serde_json::json!({}));
        if let Some(np_obj) = nemo_flow_adaptive.as_object_mut() {
            np_obj.insert(
                "latency_sensitivity".to_string(),
                serde_json::json!(effective),
            );
        }
    }
    Ok(())
}

/// Resolves the agent ID from the current scope stack.
///
/// Walks all scopes from root to top, skipping the implicit root scope
/// (index 0, name="root"), and returns the name of the first Agent-typed scope.
///
/// # Returns
/// The first Agent scope name found on the current stack, or `None` when no
/// Agent scope is active.
///
/// # Notes
/// Returns `None` when the scope stack cannot be read safely.
pub fn resolve_agent_id() -> Option<String> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };
    stack
        .scopes()
        .iter()
        .skip(1) // skip implicit root
        .find(|s| matches!(s.scope_type, ScopeType::Agent))
        .map(|s| s.name.clone())
}

/// Resolves the root scope UUID for the current execution tree.
///
/// Returns `None` if the scope stack cannot be read.
pub fn resolve_root_scope_uuid() -> Option<Uuid> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };
    Some(stack.root_uuid())
}

/// Resolves the active adaptive run-boundary scope UUID.
///
/// This mirrors telemetry drain semantics: Agent scopes are run boundaries by
/// default, unless their metadata explicitly sets `nemo_flow.run_boundary` to
/// `false`. The deepest active boundary wins. If no explicit Agent boundary is
/// visible, fall back to the implicit scope-stack root.
pub fn resolve_run_boundary_scope_uuid() -> Option<Uuid> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };

    stack
        .scopes()
        .iter()
        .skip(1)
        .rev()
        .find(|scope| {
            matches!(scope.scope_type, ScopeType::Agent)
                && scope
                    .metadata
                    .as_ref()
                    .and_then(|metadata| metadata.pointer(RUN_BOUNDARY_POINTER))
                    .and_then(|value| value.as_bool())
                    .unwrap_or(true)
        })
        .map(|scope| scope.uuid)
        .or_else(|| Some(stack.root_uuid()))
}

/// Resolves the session-local identity used by warm-first cohort coordination.
///
/// The shared parent must come from the parent scope, not the current scope's
/// own UUID, so siblings under the same fan-out coordinate with one another.
/// Returns `None` if the scope stack cannot be read.
pub fn resolve_shared_parent_scope_identity() -> Option<SharedParentScopeIdentity> {
    let stack_handle = current_scope_stack();
    let stack = match stack_handle.read() {
        Ok(s) => s,
        Err(_) => return None,
    };

    let root_uuid = stack.root_uuid();
    let shared_parent_uuid = stack.top().parent_uuid.unwrap_or(root_uuid);

    Some(SharedParentScopeIdentity {
        root_uuid,
        shared_parent_uuid,
    })
}

#[cfg(test)]
#[path = "../tests/unit/context_helpers_tests.rs"]
mod tests;
