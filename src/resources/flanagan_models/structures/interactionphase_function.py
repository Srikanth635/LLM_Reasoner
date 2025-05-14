def generate_interaction_phase(task_name,
                               grip_force,
                               cutting_force,
                               deformation_model,
                               material_properties,
                               symbolic_goals=None):
    return {
        "InteractionPhase": {
            "SubPhases": [
                {
                    "name": "Cut",
                    "description": "Move tool along predefined cutting path",
                    "goalState": { "MaterialState": { "BeingCut": True } }
                },
                {
                    "name": "MonitorAndControl",
                    "description": "Adjust tool motion and force as needed",
                    "goalState": { "ControlState": { "Stable": True } }
                }
            ],
            "SymbolicGoals": symbolic_goals or {
                "TaskStatus": { "Cutting": True }
            },
            "ForceAdaptation": {
                "grip_force": {
                    "initial_grip": grip_force["initial"],
                    "adjusted_grip": grip_force["adjusted"]
                },
                "cutting_force": {
                    "initial_cut": cutting_force["initial"],
                    "adaptive_cut": cutting_force["adaptive"]
                }
            },
            "ObjectModeling": {
                "deformation_model": {
                    "elasticity": deformation_model["elasticity"],
                    "strain_limit": deformation_model["strain_limit"],
                    "current_deformation": deformation_model["current"]
                },
                "material_properties": {
                    "hardness": material_properties["hardness"],
                    "friction_coefficient": material_properties["friction"]
                }
            },
            "SemanticAnnotation": f"PhaseClass:InteractionFor:{task_name.replace(' ', '')}"
        }
    }
interaction_phase = generate_interaction_phase(
    task_name="Cut Apple",
    grip_force={"initial": 4.5, "adjusted": 5.0},
    cutting_force={"initial": 2.0, "adaptive": 2.3},
    deformation_model={
        "elasticity": 0.3,
        "strain_limit": 0.5,
        "current": 0.4
    },
    material_properties={
        "hardness": 1.0,
        "friction": 0.4
    }
)


