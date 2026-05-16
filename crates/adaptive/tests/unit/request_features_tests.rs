// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for request message feature extraction.

use super::*;
use nemo_flow::codec::request::{Message, MessageContent};

fn user(text: &str) -> Message {
    Message::User {
        content: MessageContent::Text(text.into()),
        name: None,
    }
}

fn assistant(text: &str) -> Message {
    Message::Assistant {
        content: Some(MessageContent::Text(text.into())),
        tool_calls: None,
        name: None,
    }
}

fn tool(id: &str, text: &str) -> Message {
    Message::Tool {
        content: MessageContent::Text(text.into()),
        tool_call_id: id.into(),
    }
}

#[test]
fn trailing_tool_messages_measure_current_join_width() {
    let features = extract_message_features(&[
        user("question"),
        assistant("calling tools"),
        tool("call-1", "first result"),
        tool("call-2", "second result"),
    ]);

    assert_eq!(features.tool_message_count, 2);
    assert_eq!(features.trailing_tool_message_count, 2);
    assert!(features.trailing_tool_message_tokens_estimate > 0);
    assert_eq!(features.tool_result_join_width(), Some(2));
}

#[test]
fn non_tool_tail_resets_join_width() {
    let features = extract_message_features(&[
        user("question"),
        assistant("calling tools"),
        tool("call-1", "first result"),
        user("follow-up"),
    ]);

    assert_eq!(features.tool_message_count, 1);
    assert_eq!(features.trailing_tool_message_count, 0);
    assert_eq!(features.trailing_tool_message_tokens_estimate, 0);
    assert_eq!(features.tool_result_join_width(), None);
}
