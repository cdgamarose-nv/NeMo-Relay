// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Online empirical OSL learner state and request-signature types.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use nemo_flow::codec::request::{AnnotatedLlmRequest, Message};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

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
}

#[cfg(test)]
#[path = "../tests/unit/osl_empirical_tests.rs"]
mod tests;
