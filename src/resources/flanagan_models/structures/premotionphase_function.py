def generate_premotion_phase(task, object_info, tool_info, expected_end_state, predictive_model, motion_plan):
    return {
        "PreMotionPhase": {
            "GoalDefinition": {
                "task": task,
                "semantic_annotation": f"TaskClass:{task.replace(' ', '')}",
                "object": {
                    "id": object_info["id"],
                    "type": object_info["type"],
                    "properties": object_info["properties"],
                    "expected_end_state": expected_end_state
                },
                "tool": {
                    "id": tool_info["id"],
                    "type": tool_info["type"],
                    "properties": tool_info["properties"]
                }
            },
            "PredictiveModel": {
                "expected_trajectory": predictive_model["expected_trajectory"],
                "expected_force": {
                    "initial_N": predictive_model["initial_force"],
                    "resistance_range_N": predictive_model["resistance_range"]
                },
                "confidence_level": predictive_model.get("confidence_level", 0.95),
                "affordance_model": predictive_model.get("affordance_model", {
                    "tool_affords_cutting": True,
                    "object_affords_slicing": True
                })
            },
            "MotionPlanning": {
                "planned_trajectory": motion_plan["trajectory"],
                "obstacle_avoidance": motion_plan["obstacle_avoidance"],
                "energy_efficiency": motion_plan["energy_efficiency"]
            }
        }
    }


object_info = {
    "id": "object_apple_01",
    "type": "Fruit",
    "properties": {
        "size": "medium",
        "texture": "smooth"
    }
}

tool_info = {
    "id": "knife_standard_15cm",
    "type": "Knife",
    "properties": {
        "sharpness": "high",
        "length_cm": 15
    }
}

expected_end_state = { "ObjectState": { "SlicedInHalf": True } }

predictive_model = {
    "expected_trajectory": "Trajectory:SmoothDownwardCut",
    "initial_force": 2.0,
    "resistance_range": [1.5, 3.0]
}

motion_plan = {
    "trajectory": "Trajectory:LinearCutDownward",
    "obstacle_avoidance": "None",
    "energy_efficiency": "Optimized"
}

premotion_phase = generate_premotion_phase(
    "Cut Apple",
    object_info,
    tool_info,
    expected_end_state,
    predictive_model,
    motion_plan
)
