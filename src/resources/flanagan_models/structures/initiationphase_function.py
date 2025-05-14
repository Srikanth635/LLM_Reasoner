def generate_initiation_phase(task_name, robot_pose, tool_position, object_position,
                              joint_activation, velocity_profile,
                              pregrasp_pose_reached=True, tool_ready=True):
    return {
        "InitiationPhase": {
            "InitialState": {
                "robot_pose": {
                    "position": robot_pose["position"],
                    "orientation": robot_pose.get("orientation", [0.0, 0.0, 0.0, 1.0])
                },
                "tool_position": tool_position,
                "target_object_position": object_position
            },
            "MotionInitialization": {
                "joint_activation": joint_activation,
                "velocity_profile": f"Profile:{velocity_profile}",
                "motion_priming": {
                    "pregrasp_pose_reached": pregrasp_pose_reached,
                    "tool_ready": tool_ready
                }
            },
            "SubPhases": [
                {
                    "name": "Reach",
                    "description": "Move end-effector toward the tool or object",
                    "goalState": { "ArmState": { "Aligned": True } }
                },
                {
                    "name": "Grasp",
                    "description": "Engage the tool or object using gripper",
                    "goalState": { "ToolState": { "Grasped": True } }
                }
            ],
            "SymbolicGoals": {
                "GripperStatus": { "Engaged": True },
                "ToolState": { "Grasped": True },
                "ArmState": { "Aligned": True }
            },
            "SemanticAnnotation": f"PhaseClass:InitiationFor:{task_name.replace(' ', '')}"
        }
    }

robot_pose = {
    "position": [0.0, 0.0, 0.0],
    "orientation": [0.0, 0.0, 0.0, 1.0]
}

tool_position = [0.5, -0.2, 0.3]
object_position = [0.5, 0.0, 0.1]

joint_activation = {
    "joint1": 45,
    "joint2": 30
}

velocity_profile = "LinearRampUp"

initiation_phase = generate_initiation_phase(
    task_name="Cut Apple",
    robot_pose=robot_pose,
    tool_position=tool_position,
    object_position=object_position,
    joint_activation=joint_activation,
    velocity_profile=velocity_profile
)

