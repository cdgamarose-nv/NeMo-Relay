// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hot-cache state shared by adaptive runtime features.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dag::DagCpmState;
use crate::osl_empirical::OslEmpiricalState;
use crate::priority_residual::PriorityResidualState;
use crate::types::metadata::AgentHints;
use crate::types::plan::ExecutionPlan;

/// In-memory cache of adaptive artifacts needed on the hot path.
///
/// The adaptive runtime keeps this structure in an [`std::sync::RwLock`] so
/// intercepts and event-processing tasks can exchange recently learned plans,
/// trie state, and Adaptive Cache Governor (ACG) summaries without hitting the
/// configured backend on every request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotCache {
    /// Current execution plan used for tool parallelism hints.
    pub plan: Option<ExecutionPlan>,
    /// Prediction trie used to derive default latency sensitivity hints.
    pub trie: Option<crate::trie::data_models::PredictionTrieNode>,
    /// Default agent-level hints computed from the prediction trie.
    pub agent_hints_default: Option<AgentHints>,
    /// Learned DAG CPM aggregates used for hot-path priority projection.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dag_cpm: Option<DagCpmState>,
    /// Learned residual correction state for DAG CPM priority hints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority_residual: Option<PriorityResidualState>,
    /// Learned empirical output-token samples for optional OSL hints.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub osl_empirical: Option<OslEmpiricalState>,
    /// Per-profile ACG stability results keyed by derived profile identifier.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub acg_profiles: HashMap<String, crate::acg::stability::StabilityAnalysisResult>,
    /// Observation counts corresponding to entries in [`Self::acg_profiles`].
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub acg_profile_observation_counts: HashMap<String, u32>,
    /// Aggregate ACG stability result used for warm-first eligibility checks.
    pub acg_stability: Option<crate::acg::stability::StabilityAnalysisResult>,
    /// Observation count associated with [`Self::acg_stability`].
    pub acg_observation_count: u32,
}
