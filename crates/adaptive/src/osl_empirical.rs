// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online empirical OSL learner state and request-signature types.

use std::collections::{HashMap, VecDeque};
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use nemo_flow::codec::request::{AnnotatedLlmRequest, Message};
use nemo_flow::codec::response::FinishReason;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{AdaptiveError, Result};
use crate::learner::traits::Learner;
use crate::storage::traits::StorageBackendDyn;
use crate::types::cache::HotCache;
use crate::types::records::{CallKind, CallRecord, RunRecord};

/// Maximum retained output-token samples per empirical OSL context.
pub const MAX_SAMPLES_PER_CONTEXT: usize = 128;
/// Minimum samples before run-local empirical OSL can emit.
pub const RUN_MIN_SAMPLES: usize = 3;
/// Minimum samples before persistent empirical OSL can emit.
pub const PERSISTENT_MIN_SAMPLES: usize = 10;
/// Percentile used for emitted empirical OSL predictions.
pub const OSL_PREDICTION_PERCENTILE: u32 = 85;
/// Lower percentile used for spread confidence.
pub const OSL_SPREAD_LOW_PERCENTILE: u32 = 50;
/// Upper percentile used for spread confidence.
pub const OSL_SPREAD_HIGH_PERCENTILE: u32 = 90;
/// Maximum allowed ratio between p90 and p50, represented as an integer multiplier.
pub const OSL_MAX_SPREAD_MULTIPLIER: u32 = 4;

/// Message role used in the empirical OSL request signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OslMessageRole {
    /// System instruction message.
    System,
    /// User message.
    User,
    /// Assistant message.
    Assistant,
    /// Tool result message.
    Tool,
    /// No messages or an unrecognized shape.
    Unknown,
}

impl OslMessageRole {
    fn from_message(message: &Message) -> Self {
        match message {
            Message::System { .. } => Self::System,
            Message::User { .. } => Self::User,
            Message::Assistant { .. } => Self::Assistant,
            Message::Tool { .. } => Self::Tool,
        }
    }

    fn as_key_part(self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
            Self::Unknown => "unknown",
        }
    }
}

/// Minimal request signature used for online empirical OSL lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OslRequestSignature {
    /// Role of the final message in the request, or unknown when there are no messages.
    pub last_message_role: OslMessageRole,
    /// Whether the request includes any assistant message in the visible history.
    pub has_prior_assistant_message: bool,
    /// Whether the request includes any tool result message in the visible history.
    pub has_prior_tool_message: bool,
    /// Whether the request exposes any tool definitions to the model.
    pub tools_defined: bool,
}

impl OslRequestSignature {
    /// Derive the empirical OSL signature directly from normalized request messages.
    pub fn from_request(request: &AnnotatedLlmRequest) -> Self {
        let last_message_role = request
            .messages
            .last()
            .map(OslMessageRole::from_message)
            .unwrap_or(OslMessageRole::Unknown);
        let has_prior_assistant_message = request
            .messages
            .iter()
            .any(|message| matches!(message, Message::Assistant { .. }));
        let has_prior_tool_message = request
            .messages
            .iter()
            .any(|message| matches!(message, Message::Tool { .. }));
        let tools_defined = request
            .tools
            .as_ref()
            .is_some_and(|tools| !tools.is_empty());

        Self {
            last_message_role,
            has_prior_assistant_message,
            has_prior_tool_message,
            tools_defined,
        }
    }
}

/// Scope used to select empirical OSL history.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OslContextScope {
    /// Temporary run-local context keyed by the active run root UUID.
    Run {
        /// Root scope UUID for the active run.
        root_uuid: Uuid,
    },
    /// Persistent context for the configured agent/workflow.
    Workflow {
        /// Agent/workflow identifier.
        agent_id: String,
    },
    /// Persistent model-global context.
    Global,
}

/// Typed empirical OSL context before conversion to a stable storage key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OslContextKey {
    /// Context scope.
    pub scope: OslContextScope,
    /// Normalized model name.
    pub model: String,
    /// Message-derived request signature.
    pub signature: OslRequestSignature,
}

impl OslContextKey {
    /// Build a stable string key for in-memory maps and JSON persistence.
    pub fn storage_key(&self) -> String {
        let mut key = String::new();
        match &self.scope {
            OslContextScope::Run { root_uuid } => {
                key.push_str("scope=run:");
                key.push_str(&root_uuid.to_string());
            }
            OslContextScope::Workflow { agent_id } => {
                key.push_str("scope=workflow:");
                append_len_prefixed(&mut key, agent_id);
            }
            OslContextScope::Global => key.push_str("scope=global"),
        }

        key.push('|');
        key.push_str("model=");
        append_len_prefixed(&mut key, &self.model);
        key.push_str("|last=");
        key.push_str(self.signature.last_message_role.as_key_part());
        key.push_str("|prior_asst=");
        key.push(if self.signature.has_prior_assistant_message {
            '1'
        } else {
            '0'
        });
        key.push_str("|prior_tool=");
        key.push(if self.signature.has_prior_tool_message {
            '1'
        } else {
            '0'
        });
        key.push_str("|tools=");
        key.push(if self.signature.tools_defined {
            '1'
        } else {
            '0'
        });
        key
    }
}

fn append_len_prefixed(output: &mut String, value: &str) {
    output.push_str(&value.len().to_string());
    output.push(':');
    output.push_str(value);
}

/// Retained empirical output-token samples for one context.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OslContextStats {
    /// Recent observed completion token counts.
    #[serde(default, skip_serializing_if = "VecDeque::is_empty")]
    pub samples: VecDeque<u32>,
    /// Last observation timestamp for pruning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_updated_at: Option<DateTime<Utc>>,
}

impl OslContextStats {
    /// Record one exact observed output-token sample using the default retention limit.
    pub fn observe(&mut self, output_tokens: u32, observed_at: DateTime<Utc>) {
        self.observe_with_limit(output_tokens, observed_at, MAX_SAMPLES_PER_CONTEXT);
    }

    /// Record one exact observed output-token sample using a caller-provided retention limit.
    pub fn observe_with_limit(
        &mut self,
        output_tokens: u32,
        observed_at: DateTime<Utc>,
        max_samples: usize,
    ) {
        self.last_updated_at = Some(observed_at);
        if max_samples == 0 {
            self.samples.clear();
            return;
        }

        self.samples.push_back(output_tokens);
        while self.samples.len() > max_samples {
            self.samples.pop_front();
        }
    }

    /// Return the number of retained samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Compute the nearest-rank empirical quantile over retained samples.
    ///
    /// Returns `None` for empty samples or percentiles outside `1..=100`.
    pub fn nearest_rank_quantile(&self, percentile: u32) -> Option<u32> {
        nearest_rank_quantile(self.samples.iter().copied(), percentile)
    }

    /// Whether retained samples are sufficient and stable enough for OSL emission.
    pub fn is_confident(&self, min_samples: usize) -> bool {
        if self.samples.len() < min_samples {
            return false;
        }

        let Some(p50) = self.nearest_rank_quantile(OSL_SPREAD_LOW_PERCENTILE) else {
            return false;
        };
        let Some(p90) = self.nearest_rank_quantile(OSL_SPREAD_HIGH_PERCENTILE) else {
            return false;
        };

        u64::from(p90) <= u64::from(OSL_MAX_SPREAD_MULTIPLIER) * u64::from(p50.max(1))
    }

    /// Return the p85 empirical OSL prediction when confidence passes.
    pub fn predict_p85(&self, min_samples: usize) -> Option<u32> {
        self.is_confident(min_samples).then(|| {
            self.nearest_rank_quantile(OSL_PREDICTION_PERCENTILE)
                .expect("confidence requires non-empty samples")
        })
    }
}

fn nearest_rank_quantile<I>(samples: I, percentile: u32) -> Option<u32>
where
    I: IntoIterator<Item = u32>,
{
    if !(1..=100).contains(&percentile) {
        return None;
    }

    let mut sorted: Vec<u32> = samples.into_iter().collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_unstable();

    let n = sorted.len();
    let rank = ((u128::from(percentile) * n as u128) + 99) / 100;
    let index = usize::try_from(rank - 1).ok()?;
    sorted.get(index).copied()
}

/// Empirical OSL learner state.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OslEmpiricalState {
    /// Agent/workflow identifier this state belongs to.
    pub agent_id: String,
    /// Persistent context samples keyed by [`OslContextKey::storage_key`].
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub contexts: HashMap<String, OslContextStats>,
    /// Temporary run-local samples. These are not persisted.
    #[serde(default, skip)]
    pub run_contexts: HashMap<String, OslContextStats>,
}

impl OslEmpiricalState {
    /// Create empty empirical OSL state for an agent/workflow.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            contexts: HashMap::new(),
            run_contexts: HashMap::new(),
        }
    }

    fn observe_context(
        &mut self,
        key: OslContextKey,
        output_tokens: u32,
        observed_at: DateTime<Utc>,
    ) {
        self.contexts
            .entry(key.storage_key())
            .or_default()
            .observe(output_tokens, observed_at);
    }

    /// Predict persistent empirical OSL from workflow history, then global history.
    pub(crate) fn predict_persistent(
        &self,
        agent_id: &str,
        model: &str,
        signature: OslRequestSignature,
    ) -> Option<u32> {
        let workflow_key = OslContextKey {
            scope: OslContextScope::Workflow {
                agent_id: agent_id.to_string(),
            },
            model: model.to_string(),
            signature,
        }
        .storage_key();
        self.contexts
            .get(&workflow_key)
            .and_then(|stats| stats.predict_p85(PERSISTENT_MIN_SAMPLES))
            .or_else(|| {
                let global_key = OslContextKey {
                    scope: OslContextScope::Global,
                    model: model.to_string(),
                    signature,
                }
                .storage_key();
                self.contexts
                    .get(&global_key)
                    .and_then(|stats| stats.predict_p85(PERSISTENT_MIN_SAMPLES))
            })
    }
}

/// Apply an explicit request output-token cap to an empirical OSL prediction.
pub(crate) fn cap_osl_to_request_limit(predicted_osl: u32, request: &AnnotatedLlmRequest) -> u32 {
    let Some(limit) = request
        .params
        .as_ref()
        .and_then(|params| params.max_tokens)
        .or(request.max_output_tokens)
    else {
        return predicted_osl;
    };
    let capped = u64::from(predicted_osl).min(limit);
    u32::try_from(capped).unwrap_or(u32::MAX)
}

/// Learner that stores empirical output-token samples from completed runs.
pub struct OslEmpiricalLearner {
    agent_id: String,
}

impl OslEmpiricalLearner {
    /// Create an empirical OSL learner for one agent/workflow.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
        }
    }
}

impl Learner for OslEmpiricalLearner {
    fn process_run<'a>(
        &'a self,
        run: &'a RunRecord,
        backend: &'a dyn StorageBackendDyn,
        hot_cache: &'a Arc<RwLock<HotCache>>,
    ) -> Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let mut state = backend
                .load_osl_empirical_state(&self.agent_id)
                .await?
                .unwrap_or_else(|| OslEmpiricalState::new(&self.agent_id));
            state.agent_id = self.agent_id.clone();

            let mut observed = 0usize;
            for call in &run.calls {
                if let Some(sample) = completed_call_sample(&self.agent_id, call, run.ended_at) {
                    state.observe_context(
                        sample.workflow_key,
                        sample.output_tokens,
                        sample.observed_at,
                    );
                    state.observe_context(
                        sample.global_key,
                        sample.output_tokens,
                        sample.observed_at,
                    );
                    observed += 1;
                }
            }

            if observed == 0 {
                return Ok(());
            }

            backend
                .store_osl_empirical_state(&self.agent_id, &state)
                .await?;
            let mut guard = hot_cache.write().map_err(|error| {
                AdaptiveError::Internal(format!("hot cache lock poisoned: {error}"))
            })?;
            guard.osl_empirical = Some(state);

            Ok(())
        })
    }
}

struct CompletedCallSample {
    output_tokens: u32,
    observed_at: DateTime<Utc>,
    workflow_key: OslContextKey,
    global_key: OslContextKey,
}

fn completed_call_sample(
    agent_id: &str,
    call: &CallRecord,
    run_ended_at: Option<DateTime<Utc>>,
) -> Option<CompletedCallSample> {
    if call.kind != CallKind::Llm {
        return None;
    }
    if matches!(call.finish_reason.as_ref(), Some(FinishReason::Length)) {
        return None;
    }

    let output_tokens = call.output_tokens?;
    let request = call.annotated_request.as_deref()?;
    let model = request
        .model
        .as_deref()
        .or(call.model_name.as_deref())?
        .to_string();
    let signature = OslRequestSignature::from_request(request);
    let observed_at = call.ended_at.or(run_ended_at).unwrap_or_else(Utc::now);

    Some(CompletedCallSample {
        output_tokens,
        observed_at,
        workflow_key: OslContextKey {
            scope: OslContextScope::Workflow {
                agent_id: agent_id.to_string(),
            },
            model: model.clone(),
            signature,
        },
        global_key: OslContextKey {
            scope: OslContextScope::Global,
            model,
            signature,
        },
    })
}

#[cfg(test)]
#[path = "../tests/unit/osl_empirical_tests.rs"]
mod tests;
