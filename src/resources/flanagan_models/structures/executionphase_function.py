def generate_execution_phase(task_name,
                             subphase_info,
                             predicted_forces,
                             predicted_positions,
                             error_tolerance,
                             corrective_actions,
                             sensor_feedback,
                             tactile_data,
                             visual_data,
                             proprio_data,
                             symbolic_goals=None):
    return {
        "ExecutionPhase": {
            "SubPhases": [
                {
                    "name": sub["name"],
                    "description": sub["description"],
                    "goalState": sub["goalState"]
                } for sub in subphase_info
            ],
            "SymbolicGoals": symbolic_goals or {
                "ToolState": { "ReadyForCut": True },
                "TargetObjectState": { "PreparedForCut": True }
            },
            "FeedforwardControl": {
                "predicted_forces": {
                    "initial_force": predicted_forces["initial"],
                    "cutting_force_range": predicted_forces["range"]
                },
                "predicted_positions": {
                    "start": predicted_positions["start"],
                    "end": predicted_positions["end"]
                },
                "error_tolerance": {
                    "position_error": error_tolerance["position"],
                    "force_error": error_tolerance["force"]
                }
            },
            "FeedbackControl": {
                "corrective_actions": corrective_actions,
                "sensor_feedback": sensor_feedback
            },
            "SensoryIntegration": {
                "tactile": tactile_data,
                "visual": visual_data,
                "proprioceptive": proprio_data
            },
            "SemanticAnnotation": f"PhaseClass:ExecutionFor:{task_name.replace(' ', '')}"
        }
    }


subphases = [
    {
        "name": "AlignToolWithTarget",
        "description": "Align tool with cut start location and orientation",
        "goalState": { "ToolState": { "Aligned": True } }
    },
    {
        "name": "ApproachTarget",
        "description": "Move tool into initial contact with object",
        "goalState": {
            "ToolState": { "Engaged": True },
            "TargetObjectState": { "Contacted": True }
        }
    }
]

execution_phase = generate_execution_phase(
    task_name="Cut Apple",
    subphase_info=subphases,
    predicted_forces={"initial": 2.0, "range": [1.5, 3.0]},
    predicted_positions={"start": [0.5, -0.2, 0.3], "end": [0.5, -0.2, 0.1]},
    error_tolerance={"position": 0.01, "force": 0.1},
    corrective_actions={"adjust_trajectory": True, "increase_force": True},
    sensor_feedback={"force": 2.3, "position_error": 0.02, "slip_detected": False},
    tactile_data={"gripper_force": 5.0, "cutting_force": 2.3},
    visual_data={"object_center": [0.5, 0.2, 0.1], "knife_position": [0.5, -0.2, 0.28]},
    proprio_data={"joint_positions": [45, 30], "joint_velocities": [0.2, 0.1]}
)
