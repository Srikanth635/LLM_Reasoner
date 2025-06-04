from pydantic import BaseModel, Field
from typing import Optional, Union, List, Dict, Type, Literal
from enum import Enum, auto
from src.resources.pycram.pycram_action_designators import *

# Pydantic models mirroring your exception classes
class PlanFailureModel(BaseModel):
    """Pydantic model for plan failures."""
    message: Optional[str] = None

    class Config:
        # Allow extra fields if your exceptions have additional attributes
        extra = "allow"

class FailureDiagnosisModel(PlanFailureModel):
    """Pydantic model for FailureDiagnosis exceptions."""
    pass

class TaskModel(PlanFailureModel):
    """Pydantic model for Task exceptions."""
    pass

class HighLevelFailureModel(FailureDiagnosisModel):
    """Pydantic model for HighLevelFailure exceptions."""
    pass

class ObjectPlacingErrorModel(HighLevelFailureModel):
    """Pydantic model for ObjectPlacingError exceptions."""
    obj: ObjectModel = Field(..., description="The object that should be placed")
    placing_pose: PoseStamped = Field(..., description="The target pose at which the object should be placed")
    robot: ObjectModel = Field(..., description="The robot that placed the object")
    arm: Arms = Field(..., description="The robot arm used to place the object")

class Link(BaseModel):
    name: str

class GraspingModel(TaskModel):
    """Pydantic model for Grasping exceptions."""
    obj: ObjectModel = Field(..., description="The object to be grasped")
    robot: ObjectModel = Field(..., description="The robot that should grasp the object")
    arm: Arms = Field(..., description="The arm used to grasp the object")
    grasp: Optional[Grasp] = Field(None, description="The grasp type used to grasp the object")

class ObjectStillInContactModel(ObjectPlacingErrorModel):
    """Pydantic model for ObjectStillInContact exceptions."""
    contact_links: List[Link] = Field(...,
                                      description="The links of the robot that are still in contact with the object")
    failure_type : str = "ObjectStillInContactModel"
    def __init__(self, **data):
        # Auto-generate the error message if not provided
        if 'message' not in data or data['message'] is None:
            obj = data.get('obj')
            contact_links = data.get('contact_links', [])
            placing_pose = data.get('placing_pose')
            robot = data.get('robot')
            arm = data.get('arm')

            if all([obj, contact_links, placing_pose, robot, arm]):
                contact_link_names = [link.name if isinstance(link, Link) else str(link) for link in contact_links]
                obj_name = obj.name if isinstance(obj, ObjectModel) else str(obj)
                robot_name = robot.name if isinstance(robot, ObjectModel) else str(robot)
                arm_name = arm.value if isinstance(arm, Arms) else str(arm)

                pos_list = placing_pose.position.to_list() if hasattr(placing_pose, 'position') else "unknown"
                ori_list = placing_pose.orientation.to_list() if hasattr(placing_pose, 'orientation') else "unknown"

                data['message'] = (f"Object {obj_name} is still in contact with {robot_name}, "
                                   f"the contact links are {contact_link_names}, after placing at "
                                   f"target pose {pos_list}{ori_list} using {arm_name} arm")

        super().__init__(**data)

    @property
    def args(self) -> tuple:
        """Mimic Exception.args behavior for compatibility."""
        return (self.message,) if self.message else ()

class ObjectNotPlacedAtTargetLocationModel(ObjectPlacingErrorModel):
    """Pydantic model for ObjectNotPlacedAtTargetLocation exceptions."""
    failure_type : str = "ObjectNotPlacedAtTargetLocationModel"
    def __init__(self, **data):
        # Auto-generate the error message if not provided
        if 'message' not in data or data['message'] is None:
            obj = data.get('obj')
            placing_pose = data.get('placing_pose')
            robot = data.get('robot')
            arm = data.get('arm')

            # if all([obj, placing_pose, robot, arm]):
            if obj is not None and placing_pose is not None and robot is not None and arm is not None:
                obj_name = obj.name if isinstance(obj, ObjectModel) else str(obj)
                robot_name = robot.name if isinstance(robot, ObjectModel) else str(robot)
                arm_name = arm.value if isinstance(arm, Arms) else str(arm)

                pos_list = placing_pose.position.to_list() if hasattr(placing_pose, 'position') else "unknown"
                ori_list = placing_pose.orientation.to_list() if hasattr(placing_pose, 'orientation') else "unknown"

                data['message'] = (f"Object {obj_name} was not placed at target pose {pos_list}"
                                   f"{ori_list} using {arm_name} arm of {robot_name}")

        super().__init__(**data)

    @property
    def args(self) -> tuple:
        """Mimic Exception.args behavior for compatibility."""
        return (self.message,) if self.message else ()

class ObjectNotGraspedErrorModel(GraspingModel):
    """Pydantic model for ObjectNotGraspedError exceptions."""
    failure_type : str = "ObjectNotGraspedErrorModel"
    def __init__(self, **data):
        # Auto-generate the error message if not provided
        if 'message' not in data or data['message'] is None:
            obj = data.get('obj')
            arm = data.get('arm')
            grasp = data.get('grasp')

            grasp_str = f" using {grasp.value} grasp" if grasp else ""
            obj_name = obj.name if isinstance(obj, ObjectModel) else str(obj)
            arm_name = arm.value if isinstance(arm, Arms) else str(arm)

            data['message'] = f"object {obj_name} was not grasped by {arm_name} arm{grasp_str}"

        super().__init__(**data)

    @property
    def args(self) -> tuple:
        """Mimic Exception.args behavior for compatibility."""
        return (self.message,) if self.message else ()

