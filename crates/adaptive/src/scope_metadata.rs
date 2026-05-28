// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic scope metadata keys consumed by adaptive telemetry.

use nemo_flow::api::event::Event;

const RUN_BOUNDARY_KEY: &str = "nemo_flow.run_boundary";
const GRAPH_SCOPE_KEY: &str = "nemo_flow.graph.scope";
const GRAPH_NODE_KEY: &str = "nemo_flow.graph.node";
const GRAPH_NODE_NAME_KEY: &str = "nemo_flow.graph.node_name";
const GRAPH_TASK_ID_KEY: &str = "nemo_flow.graph.task_id";

#[derive(Debug, Clone)]
pub(crate) struct ScopeGraphMetadata {
    pub(crate) is_graph_scope: bool,
    pub(crate) is_graph_node: bool,
    pub(crate) node_name: Option<String>,
    pub(crate) task_id: Option<String>,
}

pub(crate) fn run_boundary_override(event: &Event) -> Option<bool> {
    metadata_bool(event, RUN_BOUNDARY_KEY)
}

pub(crate) fn scope_graph_metadata(event: &Event) -> ScopeGraphMetadata {
    ScopeGraphMetadata {
        is_graph_scope: metadata_bool(event, GRAPH_SCOPE_KEY).unwrap_or(false),
        is_graph_node: metadata_bool(event, GRAPH_NODE_KEY).unwrap_or(false),
        node_name: metadata_str(event, GRAPH_NODE_NAME_KEY),
        task_id: metadata_str(event, GRAPH_TASK_ID_KEY),
    }
}

fn metadata_bool(event: &Event, key: &str) -> Option<bool> {
    event
        .metadata()
        .and_then(serde_json::Value::as_object)
        .and_then(|object| object.get(key))
        .and_then(serde_json::Value::as_bool)
}

fn metadata_str(event: &Event, key: &str) -> Option<String> {
    event
        .metadata()
        .and_then(serde_json::Value::as_object)
        .and_then(|object| object.get(key))
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
}
