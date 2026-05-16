// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request-surface feature extraction shared by hot-path prediction and telemetry.

use nemo_flow::codec::request::{ContentPart, Message, MessageContent};

/// Message-derived features that distinguish phases within a reused graph node.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct RequestMessageFeatures {
    pub prompt_tokens_estimate: u32,
    pub has_prompt_content: bool,
    pub message_count: u32,
    pub assistant_message_count: u32,
    pub tool_message_count: u32,
    pub tool_message_tokens_estimate: u32,
    pub trailing_tool_message_count: u32,
    pub trailing_tool_message_tokens_estimate: u32,
    pub last_message_role: Option<&'static str>,
}

impl RequestMessageFeatures {
    pub(crate) fn last_message_is_tool(&self) -> bool {
        self.last_message_role == Some("tool")
    }

    pub(crate) fn prompt_tokens_estimate_nonzero(&self) -> Option<u32> {
        self.has_prompt_content
            .then_some(self.prompt_tokens_estimate)
    }

    pub(crate) fn tool_result_join_width(&self) -> Option<u32> {
        self.last_message_is_tool()
            .then_some(self.trailing_tool_message_count)
            .filter(|count| *count > 0)
    }
}

/// Derive prompt-size and conversation-stage features from request messages.
pub(crate) fn extract_message_features(messages: &[Message]) -> RequestMessageFeatures {
    let mut prompt_chars = 0usize;
    let mut tool_chars = 0usize;
    let mut trailing_tool_chars = 0usize;
    let mut assistant_message_count = 0u32;
    let mut tool_message_count = 0u32;
    let mut trailing_tool_message_count = 0u32;
    let mut last_message_role = None;

    for message in messages {
        last_message_role = Some(message_role(message));
        let message_chars = message_char_len(message);
        prompt_chars += message_chars;
        match message {
            Message::Assistant { .. } => {
                assistant_message_count += 1;
                trailing_tool_message_count = 0;
                trailing_tool_chars = 0;
            }
            Message::Tool { .. } => {
                tool_message_count += 1;
                tool_chars += message_chars;
                trailing_tool_message_count += 1;
                trailing_tool_chars += message_chars;
            }
            Message::System { .. } | Message::User { .. } => {
                trailing_tool_message_count = 0;
                trailing_tool_chars = 0;
            }
        }
    }

    RequestMessageFeatures {
        prompt_tokens_estimate: estimate_tokens_floor(prompt_chars),
        has_prompt_content: prompt_chars > 0,
        message_count: messages.len() as u32,
        assistant_message_count,
        tool_message_count,
        tool_message_tokens_estimate: estimate_tokens_optional(tool_chars),
        trailing_tool_message_count,
        trailing_tool_message_tokens_estimate: estimate_tokens_optional(trailing_tool_chars),
        last_message_role,
    }
}

fn estimate_tokens_floor(chars: usize) -> u32 {
    ((chars / 4).max(1)) as u32
}

fn estimate_tokens_optional(chars: usize) -> u32 {
    if chars == 0 {
        0
    } else {
        estimate_tokens_floor(chars)
    }
}

fn message_role(message: &Message) -> &'static str {
    match message {
        Message::System { .. } => "system",
        Message::User { .. } => "user",
        Message::Assistant { .. } => "assistant",
        Message::Tool { .. } => "tool",
    }
}

fn message_char_len(message: &Message) -> usize {
    match message {
        Message::System { content, .. }
        | Message::User { content, .. }
        | Message::Tool { content, .. } => content_char_len(content),
        Message::Assistant { content, .. } => content.as_ref().map_or(0, content_char_len),
    }
}

fn content_char_len(content: &MessageContent) -> usize {
    match content {
        MessageContent::Text(text) => text.len(),
        MessageContent::Parts(parts) => parts
            .iter()
            .map(|part| match part {
                ContentPart::Text { text } => text.len(),
                ContentPart::ImageUrl { .. } => 0,
            })
            .sum(),
    }
}

#[cfg(test)]
#[path = "../tests/unit/request_features_tests.rs"]
mod tests;
