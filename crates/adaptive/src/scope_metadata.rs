// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic scope metadata keys consumed by adaptive telemetry.

use nemo_flow::api::event::Event;
use serde_json::Value;

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
    scope_graph_metadata_from_value(event.name(), event.metadata())
}

pub(crate) fn scope_graph_metadata_from_value(
    scope_name: &str,
    metadata: Option<&Value>,
) -> ScopeGraphMetadata {
    let is_graph_node = metadata_bool_value(metadata, GRAPH_NODE_KEY).unwrap_or(false);
    ScopeGraphMetadata {
        is_graph_scope: metadata_bool_value(metadata, GRAPH_SCOPE_KEY).unwrap_or(false),
        is_graph_node,
        node_name: metadata_str_value(metadata, GRAPH_NODE_NAME_KEY)
            .or_else(|| is_graph_node.then(|| scope_name.to_string())),
        task_id: metadata_str_value(metadata, GRAPH_TASK_ID_KEY),
    }
}

fn metadata_bool(event: &Event, key: &str) -> Option<bool> {
    metadata_bool_value(event.metadata(), key)
}

fn metadata_bool_value(metadata: Option<&Value>, key: &str) -> Option<bool> {
    metadata
        .and_then(Value::as_object)
        .and_then(|object| object.get(key))
        .and_then(Value::as_bool)
}

fn metadata_str_value(metadata: Option<&Value>, key: &str) -> Option<String> {
    metadata
        .and_then(Value::as_object)
        .and_then(|object| object.get(key))
        .and_then(Value::as_str)
        .map(str::to_string)
}
