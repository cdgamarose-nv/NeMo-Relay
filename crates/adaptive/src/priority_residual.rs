// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Closed-loop residual correction for DAG CPM priority hints.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::dag::cpm_call_samples;
use crate::error::{AdaptiveError, Result};
use crate::learner::traits::Learner;
use crate::model::model_bucket;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::records::{CallKind, CallRecord, RunRecord};

const PRIORITY_RESIDUAL_RETENTION_DAYS: i64 = 14;
const PRIORITY_RESIDUAL_MAX_CONTEXTS: usize = 128;
const PRIORITY_RESIDUAL_MAX_MODELS: usize = 64;
const PRIORITY_SPEND_WEIGHT: f64 = 0.05;
const QUEUE_HARM_UNIT_MS: f64 = 1000.0;
const PRIORITY_RESIDUAL_DISCOUNT: f64 = 0.95;
const PRIORITY_RESIDUAL_UCB_EXPLORATION: f64 = 0.5;

/// Learned residual-priority state for one agent.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PriorityResidualState {
    /// Agent identifier the state belongs to.
    pub agent_id: String,
    /// Loss summaries keyed by coarse scheduler context.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub contexts: HashMap<String, PriorityResidualContextState>,
    /// Latest observed Dynamo pressure by normalized model name.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub pressure_by_model: HashMap<String, PressureState>,
}

impl PriorityResidualState {
    /// Create an empty residual-priority state for one agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            contexts: HashMap::new(),
            pressure_by_model: HashMap::new(),
        }
    }

    /// Select a residual action for a hot-path request.
    pub(crate) fn decision(
        &self,
        criticality: f64,
        model_name: &str,
    ) -> Option<PriorityResidualDecision> {
        let model = model_bucket(model_name);
        let pressure = self.pressure_by_model.get(&model)?;
        let key = residual_context_key(criticality, pressure.prefill_wait_time_ms, &model)?;
        let action = self
            .contexts
            .get(&key)
            .map(PriorityResidualContextState::select_action)
            .unwrap_or(PriorityResidualAction::Noop);
        Some(PriorityResidualDecision {
            key,
            action,
            delta: action.delta(),
        })
    }

    fn observe_pressure(
        &mut self,
        model_name: &str,
        prefill_wait_time_ms: f64,
        updated_at: DateTime<Utc>,
    ) -> bool {
        if !prefill_wait_time_ms.is_finite() || prefill_wait_time_ms < 0.0 {
            return false;
        }
        let model = model_bucket(model_name);
        self.pressure_by_model.insert(
            model,
            PressureState {
                prefill_wait_time_ms,
                last_updated_at: Some(updated_at),
            },
        );
        true
    }

    fn observe_loss(&mut self, key: &str, arm: u8, loss: f64, updated_at: DateTime<Utc>) -> bool {
        let Some(action) = PriorityResidualAction::from_arm(arm) else {
            return false;
        };
        if !loss.is_finite() || loss < 0.0 {
            return false;
        }

        self.contexts
            .entry(key.to_string())
            .or_default()
            .observe(action, loss, updated_at);
        true
    }

    fn prune(&mut self, now: DateTime<Utc>) -> bool {
        let cutoff = now - Duration::days(PRIORITY_RESIDUAL_RETENTION_DAYS);
        let initial_contexts = self.contexts.len();
        let initial_models = self.pressure_by_model.len();

        self.contexts.retain(|_, context| {
            context
                .last_updated_at
                .is_none_or(|updated| updated >= cutoff)
        });
        self.pressure_by_model.retain(|_, pressure| {
            pressure
                .last_updated_at
                .is_none_or(|updated| updated >= cutoff)
        });

        prune_contexts(&mut self.contexts, PRIORITY_RESIDUAL_MAX_CONTEXTS);
        prune_pressure(&mut self.pressure_by_model, PRIORITY_RESIDUAL_MAX_MODELS);

        self.contexts.len() != initial_contexts || self.pressure_by_model.len() != initial_models
    }
}

/// Loss summaries for one residual context.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PriorityResidualContextState {
    /// Arm summaries. Indexes are `down=0`, `noop=1`, `up=2`.
    pub arms: [PriorityResidualArmState; PRIORITY_RESIDUAL_ARM_COUNT],
    /// Last update timestamp for retention and staleness checks.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_updated_at: Option<DateTime<Utc>>,
    /// Total observations incorporated into this context.
    pub total_pulls: u32,
}

impl PriorityResidualContextState {
    fn observe(&mut self, action: PriorityResidualAction, loss: f64, updated_at: DateTime<Utc>) {
        for arm in &mut self.arms {
            arm.discount();
        }
        self.arms[action.arm() as usize].observe(loss);
        self.total_pulls = self.total_pulls.saturating_add(1);
        self.last_updated_at = Some(updated_at);
    }

    fn select_action(&self) -> PriorityResidualAction {
        for action in [
            PriorityResidualAction::Noop,
            PriorityResidualAction::Up,
            PriorityResidualAction::Down,
        ] {
            if self.arms[action.arm() as usize].pulls == 0 {
                return action;
            }
        }

        let total_effective_pulls = self
            .arms
            .iter()
            .map(PriorityResidualArmState::selection_pulls)
            .sum::<f64>();
        if total_effective_pulls <= 0.0 || !total_effective_pulls.is_finite() {
            return PriorityResidualAction::Noop;
        }

        [
            PriorityResidualAction::Down,
            PriorityResidualAction::Noop,
            PriorityResidualAction::Up,
        ]
        .into_iter()
        .min_by(|left, right| {
            let left_score = self.arms[left.arm() as usize].ucb_loss_score(total_effective_pulls);
            let right_score = self.arms[right.arm() as usize].ucb_loss_score(total_effective_pulls);
            left_score.total_cmp(&right_score)
        })
        .unwrap_or(PriorityResidualAction::Noop)
    }
}

/// Running mean loss for one residual arm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PriorityResidualArmState {
    /// Number of observations for this arm.
    pub pulls: u32,
    /// Discounted observation weight used for exploration.
    #[serde(default)]
    pub effective_pulls: f64,
    /// Mean observed loss. Lower is better.
    pub mean_loss: f64,
}

impl PriorityResidualArmState {
    fn discount(&mut self) {
        self.effective_pulls *= PRIORITY_RESIDUAL_DISCOUNT;
    }

    fn observe(&mut self, loss: f64) {
        let old_weight = self.selection_pulls();
        self.pulls = self.pulls.saturating_add(1);
        let new_weight = old_weight + 1.0;
        self.mean_loss = if old_weight == 0.0 {
            loss
        } else {
            (self.mean_loss * old_weight + loss) / new_weight
        };
        self.effective_pulls = new_weight;
    }

    fn ucb_loss_score(&self, total_effective_pulls: f64) -> f64 {
        let exploration =
            (total_effective_pulls.ln_1p() / self.selection_pulls().max(f64::EPSILON)).sqrt();
        self.mean_loss - PRIORITY_RESIDUAL_UCB_EXPLORATION * exploration
    }

    fn selection_pulls(&self) -> f64 {
        if self.effective_pulls.is_finite() && self.effective_pulls > 0.0 {
            self.effective_pulls
        } else {
            self.pulls as f64
        }
    }
}

/// Latest observed Dynamo pressure for a model.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct PressureState {
    /// Latest per-request Dynamo prefill/admission wait in milliseconds.
    pub prefill_wait_time_ms: f64,
    /// Last update timestamp for retention and staleness checks.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub last_updated_at: Option<DateTime<Utc>>,
}

/// Request-time residual decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PriorityResidualDecision {
    /// Context key used for feedback attribution.
    pub(crate) key: String,
    /// Selected residual action.
    pub(crate) action: PriorityResidualAction,
    /// Priority delta applied to the DAG CPM prior.
    pub(crate) delta: i32,
}

const PRIORITY_RESIDUAL_ARM_COUNT: usize = 3;

/// Residual action applied to the DAG CPM priority prior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum PriorityResidualAction {
    /// Lower the structural prior by one queue-jump unit.
    Down = 0,
    /// Keep the structural prior unchanged.
    Noop = 1,
    /// Raise the structural prior by one queue-jump unit.
    Up = 2,
}

impl PriorityResidualAction {
    pub(crate) fn from_arm(arm: u8) -> Option<Self> {
        match arm {
            0 => Some(Self::Down),
            1 => Some(Self::Noop),
            2 => Some(Self::Up),
            _ => None,
        }
    }

    pub(crate) fn arm(self) -> u8 {
        self as u8
    }

    fn delta(self) -> i32 {
        match self {
            Self::Down => -1,
            Self::Noop => 0,
            Self::Up => 1,
        }
    }
}

/// Learner that updates residual-priority feedback from completed runs.
pub struct PriorityResidualLearner {
    agent_id: String,
}

impl PriorityResidualLearner {
    /// Create a new residual-priority learner for one agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
        }
    }
}

impl Learner for PriorityResidualLearner {
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut state = backend
                .load_priority_residual_state(&self.agent_id)
                .await?
                .unwrap_or_else(|| PriorityResidualState::new(&self.agent_id));
            state.agent_id = self.agent_id.clone();

            let updated_at = run.ended_at.unwrap_or_else(Utc::now);
            let samples_by_scope = cpm_call_samples(run)
                .unwrap_or_default()
                .into_iter()
                .map(|sample| (sample.scope_uuid, sample))
                .collect::<HashMap<Uuid, _>>();

            let mut updated = false;
            for call in run.calls.iter().filter(|call| call.kind == CallKind::Llm) {
                let observed_at = call.ended_at.unwrap_or(updated_at);
                if let Some(prefill_wait_ms) = backend_prefill_wait_ms(call) {
                    updated |= state.observe_pressure(
                        &model_name_for_call(call),
                        prefill_wait_ms,
                        observed_at,
                    );
                }

                let Some(sample) = samples_by_scope.get(&call.scope_uuid) else {
                    continue;
                };
                let Some(hints) = call.adaptive_hints.as_ref() else {
                    continue;
                };
                let Some(prefill_wait_ms) = backend_prefill_wait_ms(call) else {
                    continue;
                };
                let (Some(arm), Some(key), Some(emitted_priority), Some(priority_cap)) = (
                    hints.selected_priority_residual_arm,
                    hints.selected_priority_residual_key.as_deref(),
                    hints.emitted_priority,
                    hints.priority_cap,
                ) else {
                    continue;
                };
                let Some(loss) = priority_residual_loss(
                    prefill_wait_ms,
                    sample.slack_ms,
                    emitted_priority,
                    priority_cap,
                ) else {
                    continue;
                };
                updated |= state.observe_loss(key, arm, loss, observed_at);
            }

            let pruned = state.prune(updated_at);
            if updated || pruned {
                backend
                    .store_priority_residual_state(&self.agent_id, &state)
                    .await?;
                let mut guard = hot_cache.write().map_err(|error| {
                    AdaptiveError::Internal(format!("hot cache lock poisoned: {error}"))
                })?;
                guard.priority_residual = Some(state);
            }
            Ok(())
        })
    }
}

fn backend_prefill_wait_ms(call: &CallRecord) -> Option<f64> {
    call.backend_timing
        .as_ref()
        .and_then(|timing| timing.prefill_wait_time_ms)
        .filter(|value| value.is_finite() && *value >= 0.0)
}

fn priority_residual_loss(
    prefill_wait_ms: f64,
    slack_ms: f64,
    emitted_priority: u32,
    priority_cap: u32,
) -> Option<f64> {
    if !prefill_wait_ms.is_finite() || !slack_ms.is_finite() || priority_cap == 0 {
        return None;
    }
    let queue_harm_ms = (prefill_wait_ms - slack_ms.max(0.0)).max(0.0);
    let priority_spend = emitted_priority.min(priority_cap) as f64 / priority_cap as f64;
    Some(queue_harm_ms / QUEUE_HARM_UNIT_MS + PRIORITY_SPEND_WEIGHT * priority_spend)
}

fn model_name_for_call(call: &CallRecord) -> String {
    call.annotated_request
        .as_ref()
        .and_then(|request| request.model.clone())
        .or_else(|| call.model_name.clone())
        .unwrap_or_else(|| call.name.clone())
}

fn residual_context_key(criticality: f64, prefill_wait_ms: f64, model: &str) -> Option<String> {
    Some(format!(
        "crit:{}|pressure:{}|model:{}",
        criticality_band(criticality)?,
        pressure_band(prefill_wait_ms)?,
        model
    ))
}

fn criticality_band(criticality: f64) -> Option<&'static str> {
    if !criticality.is_finite() {
        return None;
    }
    if criticality >= 0.75 {
        Some("high")
    } else if criticality >= 0.33 {
        Some("medium")
    } else if criticality >= 0.0 {
        Some("low")
    } else {
        None
    }
}

fn pressure_band(prefill_wait_ms: f64) -> Option<&'static str> {
    if !prefill_wait_ms.is_finite() || prefill_wait_ms < 0.0 {
        return None;
    }
    if prefill_wait_ms >= 1000.0 {
        Some("high")
    } else if prefill_wait_ms >= 250.0 {
        Some("moderate")
    } else {
        Some("low")
    }
}

fn prune_contexts(
    contexts: &mut HashMap<String, PriorityResidualContextState>,
    max_contexts: usize,
) {
    if contexts.len() <= max_contexts {
        return;
    }
    let mut entries = contexts
        .iter()
        .map(|(key, context)| (key.clone(), context.last_updated_at, context.total_pulls))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(_, last_updated_at, total_pulls)| (*last_updated_at, *total_pulls));
    for (key, _, _) in entries.into_iter().take(contexts.len() - max_contexts) {
        contexts.remove(&key);
    }
}

fn prune_pressure(pressure: &mut HashMap<String, PressureState>, max_models: usize) {
    if pressure.len() <= max_models {
        return;
    }
    let mut entries = pressure
        .iter()
        .map(|(key, pressure)| (key.clone(), pressure.last_updated_at))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(_, last_updated_at)| *last_updated_at);
    for (key, _) in entries.into_iter().take(pressure.len() - max_models) {
        pressure.remove(&key);
    }
}

#[cfg(test)]
#[path = "../tests/unit/priority_residual_tests.rs"]
mod tests;
