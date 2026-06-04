// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for context helpers in the NeMo Flow adaptive crate.

use super::*;
use nemo_flow::api::runtime::{create_scope_stack, set_thread_scope_stack};

#[test]
fn test_latency_sensitivity_pointer_is_valid_json_pointer() {
    // JSON pointer must start with /
    assert!(LATENCY_SENSITIVITY_POINTER.starts_with('/'));
}

#[test]
fn test_set_latency_sensitivity_basic() {
    // Sets value on the thread-local scope stack's root scope
    set_latency_sensitivity(3).unwrap();
    assert_eq!(read_manual_latency_sensitivity(), Some(3));

    // Clean up: reset root scope metadata
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_set_latency_sensitivity_max_merge_higher_wins() {
    set_latency_sensitivity(3).unwrap();
    set_latency_sensitivity(5).unwrap();
    assert_eq!(read_manual_latency_sensitivity(), Some(5));

    // Clean up
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_set_latency_sensitivity_max_merge_lower_noop() {
    set_latency_sensitivity(5).unwrap();
    set_latency_sensitivity(3).unwrap();
    // Lower value should not override
    assert_eq!(read_manual_latency_sensitivity(), Some(5));

    // Clean up
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_set_latency_sensitivity_read_roundtrip() {
    // Ensure read_manual_latency_sensitivity reads what set_latency_sensitivity writes
    set_latency_sensitivity(7).unwrap();
    assert_eq!(read_manual_latency_sensitivity(), Some(7));

    // Clean up
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_read_workflow_class_reads_nearest_scope_metadata() {
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = Some(serde_json::json!({
        "nemo_flow_adaptive": {
            "workflow_class": "standard",
        }
    }));
    drop(stack);
    assert_eq!(read_workflow_class(), Some(WorkflowClass::Standard));

    let function_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("branch")
            .scope_type(ScopeType::Function)
            .metadata(serde_json::json!({
                "nemo_flow_adaptive": {
                    "scheduling_class": "background",
                }
            }))
            .build(),
    )
    .unwrap();
    assert_eq!(read_workflow_class(), Some(WorkflowClass::Background));

    nemo_flow::api::scope::pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&function_scope.uuid)
            .build(),
    )
    .unwrap();
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = None;
}

#[test]
fn test_extract_graph_call_context_reads_generic_scope_metadata() {
    set_thread_scope_stack(create_scope_stack());
    let graph_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("research_graph")
            .scope_type(ScopeType::Agent)
            .metadata(serde_json::json!({
                "nemo_flow.graph.scope": true,
            }))
            .build(),
    )
    .unwrap();
    let node_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("researcher")
            .scope_type(ScopeType::Agent)
            .metadata(serde_json::json!({
                "nemo_flow.graph.node": true,
                "nemo_flow.graph.node_name": "researcher",
                "nemo_flow.graph.task_id": "task-1",
                "nemo_flow.graph.depends_on_task_ids": ["task-0"],
            }))
            .build(),
    )
    .unwrap();

    let context = extract_graph_call_context().unwrap();

    assert_eq!(context.graph_name.as_deref(), Some("research_graph"));
    assert_eq!(context.node_name, "researcher");
    assert_eq!(context.task_id, "task-1");
    assert_eq!(context.depends_on_task_ids, vec!["task-0"]);

    nemo_flow::api::scope::pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&node_scope.uuid)
            .build(),
    )
    .unwrap();
    nemo_flow::api::scope::pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&graph_scope.uuid)
            .build(),
    )
    .unwrap();
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn test_extract_graph_call_context_requires_task_id() {
    set_thread_scope_stack(create_scope_stack());
    let _node_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("researcher")
            .scope_type(ScopeType::Agent)
            .metadata(serde_json::json!({
                "nemo_flow.graph.node": true,
            }))
            .build(),
    )
    .unwrap();

    assert_eq!(extract_graph_call_context(), None);
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn test_helpers_return_defaults_when_scope_stack_lock_is_poisoned() {
    let poisoned = create_scope_stack();
    let poisoned_for_panic = poisoned.clone();
    let _ = std::panic::catch_unwind(move || {
        let _guard = poisoned_for_panic.write().unwrap();
        panic!("poison scope stack");
    });

    set_thread_scope_stack(poisoned);
    assert!(extract_scope_path().is_empty());
    assert_eq!(read_manual_latency_sensitivity(), None);
    assert_eq!(read_workflow_class(), None);
    assert_eq!(extract_graph_call_context(), None);
    assert_eq!(resolve_agent_id(), None);
    assert_eq!(resolve_root_scope_uuid(), None);

    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn test_resolve_root_scope_uuid_reads_current_stack_root() {
    let stack_handle = current_scope_stack();
    let root_uuid = stack_handle.read().unwrap().root_uuid();

    assert_eq!(resolve_root_scope_uuid(), Some(root_uuid));
}

#[test]
fn test_resolve_run_boundary_scope_uuid_matches_active_agent_boundary() {
    set_thread_scope_stack(create_scope_stack());
    let stack_root_uuid = current_scope_stack().read().unwrap().root_uuid();
    let agent_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("aiq.chat")
            .scope_type(ScopeType::Agent)
            .build(),
    )
    .unwrap();
    let graph_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("LangGraph")
            .scope_type(ScopeType::Agent)
            .metadata(serde_json::json!({"nemo_flow.run_boundary": false}))
            .build(),
    )
    .unwrap();
    let node_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("researcher")
            .scope_type(ScopeType::Agent)
            .metadata(serde_json::json!({"nemo_flow.run_boundary": false}))
            .build(),
    )
    .unwrap();

    assert_eq!(resolve_root_scope_uuid(), Some(stack_root_uuid));
    assert_eq!(resolve_run_boundary_scope_uuid(), Some(agent_scope.uuid));

    nemo_flow::api::scope::pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&node_scope.uuid)
            .build(),
    )
    .unwrap();
    nemo_flow::api::scope::pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&graph_scope.uuid)
            .build(),
    )
    .unwrap();
    nemo_flow::api::scope::pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&agent_scope.uuid)
            .build(),
    )
    .unwrap();
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn test_resolve_run_boundary_scope_uuid_falls_back_to_stack_root() {
    set_thread_scope_stack(create_scope_stack());
    let stack_root_uuid = current_scope_stack().read().unwrap().root_uuid();
    let graph_scope = nemo_flow::api::scope::push_scope(
        nemo_flow::api::scope::PushScopeParams::builder()
            .name("LangGraph")
            .scope_type(ScopeType::Agent)
            .metadata(serde_json::json!({"nemo_flow.run_boundary": false}))
            .build(),
    )
    .unwrap();

    assert_eq!(resolve_run_boundary_scope_uuid(), Some(stack_root_uuid));

    nemo_flow::api::scope::pop_scope(
        nemo_flow::api::scope::PopScopeParams::builder()
            .handle_uuid(&graph_scope.uuid)
            .build(),
    )
    .unwrap();
    set_thread_scope_stack(create_scope_stack());
}

#[test]
fn test_set_latency_sensitivity_ignores_non_object_metadata() {
    let stack_handle = current_scope_stack();
    let mut stack = stack_handle.write().unwrap();
    stack.top_mut().metadata = Some(serde_json::json!("metadata"));
    drop(stack);

    set_latency_sensitivity(9).unwrap();

    let mut stack = stack_handle.write().unwrap();
    assert_eq!(
        stack.top_mut().metadata,
        Some(serde_json::json!("metadata"))
    );
    stack.top_mut().metadata = None;
}
