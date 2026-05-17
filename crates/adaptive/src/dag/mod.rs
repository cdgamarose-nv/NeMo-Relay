// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! DAG CPM state shared by the learner, storage backends, and hot path.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

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
