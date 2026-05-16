// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for scheduling priority primitives.

use super::*;

fn queue_jump_scale() -> PriorityScale {
    PriorityScale::dynamo_router_priority_jump()
}

fn standard_features_with_criticality(dag_sensitivity: Option<f64>) -> RuntimePriorityFeatures {
    RuntimePriorityFeatures {
        scheduling_class: Some(SchedulingClass::Standard),
        dag_sensitivity,
        scope_depth: 3,
        call_index: 1,
        tools_defined: false,
        last_message_is_tool: false,
        tool_result_join_width: None,
        ..RuntimePriorityFeatures::default()
    }
}

#[test]
fn higher_criticality_adds_one_queue_jump_unit() {
    let low = derive_priority_decision(
        standard_features_with_criticality(Some(0.1)),
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );
    let high = derive_priority_decision(
        standard_features_with_criticality(Some(0.9)),
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );

    assert_eq!(low.priority_prior, 1);
    assert_eq!(high.priority_prior, 2);
    assert_eq!(high.priority_raw - low.priority_raw, 1);
}

#[test]
fn large_fanout_is_penalized_with_lower_priority_bucket() {
    let large_fanout = PriorityKey {
        anti_fanout: AntiFanoutBucket::LargeFanout,
        workflow_class: WorkflowClassBucket::Standard,
        criticality: CriticalityBucket::Medium,
        gate_or_join: GateOrJoinBucket::Ordinary,
        stage: StageBucket::Ordinary,
        repeat: RepeatBucket::FirstOrEarly,
    };
    let small_fanout = PriorityKey {
        anti_fanout: AntiFanoutBucket::SmallOrSingleton,
        ..large_fanout
    };

    assert!(small_fanout > large_fanout);
    assert_eq!(dynamo_queue_jump_raw(large_fanout), 0);
    assert_eq!(dynamo_queue_jump_raw(small_fanout), 1);
}

#[test]
fn queue_jump_policy_has_interpretable_bands_and_caps() {
    let background_fanout = derive_priority_decision(
        RuntimePriorityFeatures {
            scheduling_class: Some(SchedulingClass::Background),
            scope_depth: 4,
            call_index: 2,
            tool_result_join_width: Some(6),
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );
    let standard_ordinary = derive_priority_decision(
        RuntimePriorityFeatures {
            scheduling_class: Some(SchedulingClass::Standard),
            scope_depth: 4,
            call_index: 2,
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );
    let interactive_high_crit_synthesis = derive_priority_decision(
        RuntimePriorityFeatures {
            scheduling_class: Some(SchedulingClass::Interactive),
            dag_sensitivity: Some(0.9),
            scope_depth: 2,
            call_index: 1,
            tools_defined: true,
            last_message_is_tool: true,
            tool_result_join_width: Some(1),
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );

    assert_eq!(background_fanout.priority_raw, -1);
    assert_eq!(background_fanout.priority_prior, 0);
    assert_eq!(background_fanout.priority_cap, 1);
    assert_eq!(standard_ordinary.priority_prior, 1);
    assert_eq!(standard_ordinary.priority_cap, 3);
    assert_eq!(interactive_high_crit_synthesis.priority_raw, 5);
    assert_eq!(interactive_high_crit_synthesis.priority_prior, 5);
    assert_eq!(interactive_high_crit_synthesis.priority_cap, 5);
}

#[test]
fn residual_quantum_is_derived_from_scale() {
    assert_eq!(PriorityScale::new(0, 1000, 100).residual_quantum(), 10);
    assert_eq!(PriorityScale::new(0, 100, 100).residual_quantum(), 1);
    assert_eq!(PriorityScale::new(0, 5, 100).residual_quantum(), 1);
    assert_eq!(PriorityScale::new(7, 7, 100).residual_quantum(), 0);
}

#[test]
fn residual_action_nudges_and_clamps_around_prior() {
    let scale = queue_jump_scale();

    assert_eq!(PriorityResidualAction::Up.apply(4, scale, 5), 5);
    assert_eq!(PriorityResidualAction::Noop.apply(4, scale, 5), 4);
    assert_eq!(PriorityResidualAction::Down.apply(4, scale, 5), 3);
    assert_eq!(PriorityResidualAction::Up.apply(-1, scale, 1), 0);
}

#[test]
fn planned_mid_buckets_are_ordered() {
    assert!(WorkflowClassBucket::Interactive > WorkflowClassBucket::Standard);
    assert!(CriticalityBucket::Medium > CriticalityBucket::Low);
    assert!(GateOrJoinBucket::JoinOrSynthesis > GateOrJoinBucket::Ordinary);
    assert!(AntiFanoutBucket::MediumFanout > AntiFanoutBucket::LargeFanout);
    assert!(StageBucket::ToolPlanning > StageBucket::Ordinary);
    assert!(PriorityResidualAction::Noop.delta(queue_jump_scale()) == 0);
}

#[test]
fn runtime_priority_decision_maps_available_runtime_facts() {
    let decision = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: Some(0.8),
            scope_depth: 3,
            call_index: 2,
            tools_defined: true,
            last_message_is_tool: false,
            tool_result_join_width: None,
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );

    assert_eq!(decision.key.criticality, CriticalityBucket::High);
    assert_eq!(decision.key.gate_or_join, GateOrJoinBucket::Ordinary);
    assert_eq!(decision.key.stage, StageBucket::ToolPlanning);
    assert_eq!(decision.key.repeat, RepeatBucket::Repeated);
    assert_eq!(decision.residual_action, PriorityResidualAction::Noop);
    assert_eq!(decision.priority_prior, decision.priority_final);
}

#[test]
fn background_no_sla_is_capped_but_raw_structural_signal_is_recorded() {
    let intent = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: None,
            scope_depth: 2,
            call_index: 10,
            tools_defined: false,
            last_message_is_tool: false,
            tool_result_join_width: None,
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );
    let tool_followup = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: Some(1.0),
            scope_depth: 2,
            call_index: 10,
            tools_defined: true,
            last_message_is_tool: true,
            tool_result_join_width: Some(1),
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );

    assert_eq!(intent.priority_cap, 1);
    assert_eq!(tool_followup.priority_cap, 1);
    assert!(tool_followup.priority_raw > intent.priority_raw);
    assert_eq!(intent.priority_prior, 0);
    assert_eq!(tool_followup.priority_prior, 1);
}

#[test]
fn priority_prior_has_no_size_dimension() {
    let baseline = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: None,
            scope_depth: 2,
            call_index: 1,
            tools_defined: false,
            last_message_is_tool: false,
            tool_result_join_width: None,
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );
    let same_features = RuntimePriorityFeatures {
        dag_sensitivity: None,
        scope_depth: 2,
        call_index: 1,
        tools_defined: false,
        last_message_is_tool: false,
        tool_result_join_width: None,
        ..RuntimePriorityFeatures::default()
    };
    let repeated = derive_priority_decision(same_features, queue_jump_scale(), PriorityResidualAction::Noop);

    assert_eq!(baseline.priority_prior, repeated.priority_prior);
    assert_eq!(baseline.key, repeated.key);
}

#[test]
fn dag_criticality_is_bounded_inside_no_sla_band() {
    let without_dag = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: None,
            scope_depth: 2,
            call_index: 1,
            tools_defined: false,
            last_message_is_tool: false,
            tool_result_join_width: None,
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );
    let with_dag = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: Some(0.9),
            scope_depth: 2,
            call_index: 1,
            tools_defined: false,
            last_message_is_tool: false,
            tool_result_join_width: None,
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );

    assert!(with_dag.priority_raw > without_dag.priority_raw);
    assert_eq!(without_dag.priority_prior, 0);
    assert_eq!(with_dag.priority_prior, 1);
    assert_eq!(with_dag.priority_cap, 1);
}

#[test]
fn tool_result_join_width_penalizes_large_batches_within_followups() {
    let singleton_join = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: None,
            scheduling_class: Some(SchedulingClass::Standard),
            scope_depth: 2,
            call_index: 2,
            tools_defined: true,
            last_message_is_tool: true,
            tool_result_join_width: Some(1),
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );
    let large_join = derive_priority_decision(
        RuntimePriorityFeatures {
            dag_sensitivity: None,
            scheduling_class: Some(SchedulingClass::Standard),
            scope_depth: 2,
            call_index: 2,
            tools_defined: true,
            last_message_is_tool: true,
            tool_result_join_width: Some(6),
            ..RuntimePriorityFeatures::default()
        },
        queue_jump_scale(),
        PriorityResidualAction::Noop,
    );

    assert_eq!(
        singleton_join.key.anti_fanout,
        AntiFanoutBucket::SmallOrSingleton
    );
    assert_eq!(large_join.key.anti_fanout, AntiFanoutBucket::LargeFanout);
    assert!(singleton_join.priority_prior > large_join.priority_prior);
    assert_eq!(singleton_join.key.stage, StageBucket::ToolResultFollowup);
    assert_eq!(large_join.key.stage, StageBucket::ToolResultFollowup);
}
