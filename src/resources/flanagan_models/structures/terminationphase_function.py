def generate_termination_phase(task_name,
                                final_tool_position,
                                reset_trajectory,
                                outcome_check,
                                error_analysis,
                                symbolic_goals=None):
    return {
        "TerminationPhase": {
            "SubPhases": [
                {
                    "name": "WithdrawTool",
                    "description": "Lift tool from object",
                    "goalState": { "ToolState": { "Lifted": True } }
                },
                {
                    "name": "ReleaseTool",
                    "description": "Put down or release tool",
                    "goalState": { "ToolState": { "Released": True } }
                }
            ],
            "SymbolicGoals": symbolic_goals or {
                "TargetObjectState": { "Separated": True },
                "ToolState": { "Released": True }
            },
            "EndEffectorStabilization": {
                "final_position": {
                    "knife_position": final_tool_position
                },
                "reset_trajectory": f"Trajectory:{reset_trajectory}"
            },
            "SuccessVerification": {
                "outcome_check": outcome_check,
                "error_analysis": {
                    "cut_quality": f"CutQuality:{error_analysis['cut_quality']}",
                    "deviation_from_center": error_analysis["deviation"]
                }
            },
            "SemanticAnnotation": f"PhaseClass:TerminationFor:{task_name.replace(' ', '')}"
        }
    }


termination_phase = generate_termination_phase(
    task_name="Cut Apple",
    final_tool_position=[0.5, -0.2, 0.0],
    reset_trajectory="LinearResetToNeutral",
    outcome_check={
        "apple_cut": True,
        "halves_separated": True
    },
    error_analysis={
        "cut_quality": "Smooth",
        "deviation": 0.02
    }
)
