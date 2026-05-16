// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Internal scheduling-prior primitives for adaptive LLM hints.

use crate::context_helpers::SchedulingClass;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum WorkflowClassBucket {
    BackgroundOrUnspecified = 0,
    Standard = 1,
    Interactive = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum CriticalityBucket {
    Unknown = 0,
    Low = 1,
    Medium = 2,
    High = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum GateOrJoinBucket {
    Ordinary = 0,
    JoinOrSynthesis = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum AntiFanoutBucket {
    LargeFanout = 0,
    MediumFanout = 1,
    SmallOrSingleton = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum StageBucket {
    Ordinary = 0,
    ToolPlanning = 1,
    ToolResultFollowup = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub(crate) enum RepeatBucket {
    Repeated = 0,
    FirstOrEarly = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct PriorityKey {
    pub(crate) workflow_class: WorkflowClassBucket,
    pub(crate) criticality: CriticalityBucket,
    pub(crate) gate_or_join: GateOrJoinBucket,
    pub(crate) anti_fanout: AntiFanoutBucket,
    pub(crate) stage: StageBucket,
    pub(crate) repeat: RepeatBucket,
}

impl PriorityKey {
    // GateOrJoinBucket dropped RootOrEarlyGate; radix at index 2 is now 2 (was 3).
    const RADICES: [u32; 6] = [3, 4, 2, 3, 3, 2];

    fn max_ordinal() -> u32 {
        Self::RADICES.into_iter().product::<u32>().saturating_sub(1)
    }

    fn structural_levels() -> u32 {
        Self::max_ordinal().saturating_add(1)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PriorityScale {
    min_priority: u32,
    max_priority: u32,
    structural_levels: u32,
}

impl PriorityScale {
    pub(crate) fn new(min_priority: u32, max_priority: u32, structural_levels: u32) -> Self {
        Self {
            min_priority: min_priority.min(max_priority),
            max_priority: min_priority.max(max_priority),
            structural_levels: structural_levels.max(1),
        }
    }

    pub(crate) fn dynamo_router_priority_jump() -> Self {
        // Dynamo treats nvext.agent_hints.priority as a router queue jump.
        // Keep the emitted range small: 5 means about a five-second FCFS boost.
        Self::new(0, 5, PriorityKey::structural_levels())
    }

    pub(crate) fn residual_quantum(self) -> u32 {
        let range = self.max_priority.saturating_sub(self.min_priority);
        if range == 0 {
            0
        } else {
            (range / self.structural_levels).max(1)
        }
    }

    pub(crate) fn clamp(self, priority: i32) -> u32 {
        priority.clamp(self.min_priority as i32, self.max_priority as i32) as u32
    }

    pub(crate) fn clamp_to_cap(self, priority: i32, cap: u32) -> u32 {
        let cap = cap.clamp(self.min_priority, self.max_priority);
        priority.clamp(self.min_priority as i32, cap as i32) as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PriorityResidualAction {
    Down,
    Noop,
    Up,
}

impl PriorityResidualAction {
    pub(crate) fn delta(self, scale: PriorityScale) -> i32 {
        let quantum = scale.residual_quantum() as i32;
        match self {
            Self::Down => -quantum,
            Self::Noop => 0,
            Self::Up => quantum,
        }
    }

    pub(crate) fn apply(self, raw_priority: i32, scale: PriorityScale, cap: u32) -> u32 {
        scale.clamp_to_cap(raw_priority + self.delta(scale), cap)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct RuntimePriorityFeatures {
    pub(crate) scheduling_class: Option<SchedulingClass>,
    pub(crate) dag_sensitivity: Option<f64>,
    pub(crate) scope_depth: usize,
    pub(crate) call_index: u32,
    pub(crate) tools_defined: bool,
    pub(crate) last_message_is_tool: bool,
    pub(crate) tool_result_join_width: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PriorityDecision {
    pub(crate) key: PriorityKey,
    pub(crate) scale: PriorityScale,
    pub(crate) priority_raw: i32,
    pub(crate) priority_cap: u32,
    pub(crate) priority_prior: u32,
    pub(crate) residual_action: PriorityResidualAction,
    pub(crate) residual_delta: i32,
    pub(crate) priority_final: u32,
}

pub(crate) fn derive_priority_decision(
    features: RuntimePriorityFeatures,
    scale: PriorityScale,
    residual_action: PriorityResidualAction,
) -> PriorityDecision {
    let key = PriorityKey {
        workflow_class: workflow_class_bucket(features.scheduling_class),
        criticality: criticality_bucket(features.dag_sensitivity),
        gate_or_join: gate_or_join_bucket(features.last_message_is_tool),
        anti_fanout: anti_fanout_bucket(features.tool_result_join_width),
        stage: stage_bucket(features.tools_defined, features.last_message_is_tool),
        repeat: repeat_bucket(features.call_index),
    };
    let priority_raw = dynamo_queue_jump_raw(key);
    let priority_cap = dynamo_queue_jump_cap(key, scale);
    let priority_prior = scale.clamp_to_cap(priority_raw, priority_cap);
    let residual_delta = residual_action.delta(scale);
    let priority_final = residual_action.apply(priority_raw, scale, priority_cap);

    PriorityDecision {
        key,
        scale,
        priority_raw,
        priority_cap,
        priority_prior,
        residual_action,
        residual_delta,
        priority_final,
    }
}

fn dynamo_queue_jump_raw(key: PriorityKey) -> i32 {
    workflow_class_base(key.workflow_class)
        + criticality_boost(key.criticality)
        + gate_or_join_boost(key.gate_or_join)
        + stage_boost(key.stage)
        + fanout_penalty(key.anti_fanout)
}

fn dynamo_queue_jump_cap(key: PriorityKey, scale: PriorityScale) -> u32 {
    let cap = match key.workflow_class {
        WorkflowClassBucket::BackgroundOrUnspecified => 1,
        WorkflowClassBucket::Standard => 3,
        WorkflowClassBucket::Interactive => 5,
    };
    scale.clamp(cap)
}

fn workflow_class_base(workflow_class: WorkflowClassBucket) -> i32 {
    match workflow_class {
        WorkflowClassBucket::BackgroundOrUnspecified => 0,
        WorkflowClassBucket::Standard => 1,
        WorkflowClassBucket::Interactive => 2,
    }
}

fn criticality_boost(criticality: CriticalityBucket) -> i32 {
    match criticality {
        CriticalityBucket::High => 1,
        CriticalityBucket::Unknown | CriticalityBucket::Low | CriticalityBucket::Medium => 0,
    }
}

fn gate_or_join_boost(gate_or_join: GateOrJoinBucket) -> i32 {
    match gate_or_join {
        GateOrJoinBucket::JoinOrSynthesis => 1,
        GateOrJoinBucket::Ordinary => 0,
    }
}

fn stage_boost(stage: StageBucket) -> i32 {
    match stage {
        StageBucket::ToolResultFollowup => 1,
        StageBucket::Ordinary | StageBucket::ToolPlanning => 0,
    }
}

fn fanout_penalty(anti_fanout: AntiFanoutBucket) -> i32 {
    match anti_fanout {
        AntiFanoutBucket::LargeFanout => -1,
        AntiFanoutBucket::MediumFanout | AntiFanoutBucket::SmallOrSingleton => 0,
    }
}

fn workflow_class_bucket(scheduling_class: Option<SchedulingClass>) -> WorkflowClassBucket {
    match scheduling_class {
        Some(SchedulingClass::Interactive) => WorkflowClassBucket::Interactive,
        Some(SchedulingClass::Standard) => WorkflowClassBucket::Standard,
        Some(SchedulingClass::Background) | None => WorkflowClassBucket::BackgroundOrUnspecified,
    }
}

fn criticality_bucket(dag_sensitivity: Option<f64>) -> CriticalityBucket {
    match dag_sensitivity {
        None => CriticalityBucket::Unknown,
        Some(value) if value > 0.7 => CriticalityBucket::High,
        Some(value) if value > 0.3 => CriticalityBucket::Medium,
        Some(_) => CriticalityBucket::Low,
    }
}

fn gate_or_join_bucket(last_message_is_tool: bool) -> GateOrJoinBucket {
    if last_message_is_tool {
        GateOrJoinBucket::JoinOrSynthesis
    } else {
        GateOrJoinBucket::Ordinary
    }
}

fn anti_fanout_bucket(tool_result_join_width: Option<u32>) -> AntiFanoutBucket {
    match tool_result_join_width {
        Some(width) if width >= 5 => AntiFanoutBucket::LargeFanout,
        Some(width) if width >= 2 => AntiFanoutBucket::MediumFanout,
        Some(_) => AntiFanoutBucket::SmallOrSingleton,
        None => AntiFanoutBucket::SmallOrSingleton,
    }
}

fn stage_bucket(tools_defined: bool, last_message_is_tool: bool) -> StageBucket {
    if last_message_is_tool {
        StageBucket::ToolResultFollowup
    } else if tools_defined {
        StageBucket::ToolPlanning
    } else {
        StageBucket::Ordinary
    }
}

fn repeat_bucket(call_index: u32) -> RepeatBucket {
    if call_index <= 1 {
        RepeatBucket::FirstOrEarly
    } else {
        RepeatBucket::Repeated
    }
}

#[cfg(test)]
#[path = "../tests/unit/scheduling_tests.rs"]
mod tests;
