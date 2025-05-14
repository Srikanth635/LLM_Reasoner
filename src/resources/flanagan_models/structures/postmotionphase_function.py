def generate_post_motion_phase(task_name,
                                force_model_update,
                                trajectory_model_update,
                                reinforcement_params,
                                deviation_analysis,
                                parameter_update,
                                symbolic_goals=None):
    return {
        "PostMotionPhase": {
            "SubPhases": [
                {
                    "name": "UpdateStatus",
                    "description": "Record cut outcome and update world model",
                    "goalState": { "KnowledgeBase": { "Updated": True } }
                },
                {
                    "name": "PrepareNextTask",
                    "description": "Reset system for next task",
                    "goalState": { "SystemState": { "Initialized": True } }
                }
            ],
            "SymbolicGoals": symbolic_goals or {
                "SystemState": { "Ready": True }
            },
            "LearningUpdate": {
                "model_refinement": {
                    "force_model_update": {
                        "cutting_force": force_model_update["cutting_force"]
                    },
                    "trajectory_model_update": {
                        "deviation_correction": trajectory_model_update["deviation_correction"]
                    }
                },
                "reinforcement": {
                    "successful_cut": reinforcement_params["success"],
                    "reinforce_parameters": {
                        "grip_force": reinforcement_params["grip_force"]
                    }
                }
            },
            "ErrorCorrection": {
                "deviation_analysis": {
                    "position_error": deviation_analysis["position_error"],
                    "force_error": deviation_analysis["force_error"],
                    "slip_event": deviation_analysis["slip_event"]
                },
                "parameter_update": {
                    "cutting_force_range": parameter_update["cutting_force_range"],
                    "grip_force_range": parameter_update["grip_force_range"]
                }
            },
            "SemanticAnnotation": f"PhaseClass:PostMotionFor:{task_name.replace(' ', '')}"
        }
    }


post_motion_phase = generate_post_motion_phase(
    task_name="Cut Apple",
    force_model_update={"cutting_force": 2.5},
    trajectory_model_update={"deviation_correction": 0.01},
    reinforcement_params={"success": True, "grip_force": 5.0},
    deviation_analysis={"position_error": 0.02, "force_error": 0.1, "slip_event": False},
    parameter_update={
        "cutting_force_range": [1.6, 2.8],
        "grip_force_range": [4.8, 5.2]
    }
)
