from sqlalchemy import True_

from src.langchain.create_agents import create_agent
from src.resources.pycram.pycram_action_designators import (MoveTorsoAction, SetGripperAction,
        GripAction, ParkArmsAction, NavigateAction, PickUpAction, PlaceAction, ReachToPickUpAction, TransportAction,
        LookAtAction, OpenAction, CloseAction, GraspingAction, MoveAndPickUpAction, MoveAndPlaceAction, FaceAtAction,
                                                            DetectAction, SearchAction)
from src.resources.pycram.pycram_failures import *

from src.resources.pycram.pycram_action_designators import *
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.tools.structured import StructuredTool
from langchain.agents import Tool
from src.langchain.create_agents import *
from src.langchain.llm_configuration import *
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, Type, List, Literal, Union, Annotated
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentState
import json

action_classes = [PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction]

failure_reasons = [ObjectNotGraspedErrorModel,ObjectStillInContactModel,ObjectNotPlacedAtTargetLocationModel]


# Agent State ---------------------------------------------------------------------------------------------------

action_designator_type = Union[PickUpAction, NavigateAction, PlaceAction, SetGripperAction, LookAtAction,
                  MoveTorsoAction, GripAction, ParkArmsAction, MoveAndPickUpAction, MoveAndPlaceAction,
                  OpenAction, CloseAction, GraspingAction, ReachToPickUpAction, TransportAction,
                    SearchAction, FaceAtAction, str]

failure_reason_type = Union[ObjectNotGraspedErrorModel,ObjectStillInContactModel,ObjectNotPlacedAtTargetLocationModel, str]

class CustomState(AgentState):
    action_designator : action_designator_type
    reason_for_failure : failure_reason_type
    human_comment : str
    updated_action_designator : action_designator_type
    explanation : str

tool_prompt_template = """
            You are a helpful and detail-oriented assistant specialized in robotics action planning and correction. 
        Your task is to correct or modify a given action designator object based on a reported error reason and an optional human comment.

        ## Objective
        Analyze the provided `action_designator` (a structured object representing a robotic action), the `error_reason` 
        (a diagnostic explanation of why the action failed or is invalid), and a possible `human_comment` 
        (extra guidance or clarification from a human operator). Based on this information, perform **step-by-step reasoning** 
        and propose a **corrected version of the action_designator** that resolves the issue.

        ## Inputs
        - `action_designator`: A structured object with fields like action type, arm, object, grasp description, and robot state.
        - `error_reason`: A description of the validation or execution error encountered.
        - `human_comment`: (Optional) Natural language input from a human providing suggestions, hints, or additional context.

        ## Output
        A corrected or improved version of the original `action_designator` object that addresses the error. 
        If a fix is not possible due to insufficient context, suggest minimal, plausible modifications and add a note about what clarification is needed.

        ## Reasoning Chain (Apply this step-by-step logic):
        1. **Understand the action goal** from the action_type and its parameters.
        2. **Analyze the `error_reason`** to identify what went wrong (e.g., missing fields, invalid values, planning issues).
        3. **Parse the `human_comment`**, if any, for cues or corrections intended by the user.
        4. **Identify incorrect, missing, or conflicting fields** in the original `action_designator`.
        5. **Propose a corrected designator**, modifying only the necessary parts while preserving original intent.
        6. **Explain each modification** made to enhance transparency (this can be included in comments or metadata if your system allows).

        ## Style Notes
        - Be conservative: modify only what is necessary.
        - Be consistent with the domain schema (e.g., grasp types, arms, object identifiers).
        - If multiple corrections are possible, prefer the most contextually plausible.
        - Maintain syntactic and semantic integrity of the final action designator.

        Begin by reasoning through the error and then output the corrected designator.

        ---

        Perform the task mentioned for the provided 

        action_designator :{action_designator}

        reason_for_failure : {reason_for_failure}

        human_comment : {human_comment}

        ---

        return the action designator with modified attribute values as per the failure
        """

def updater_node(state : CustomState):

    structured_ollama = None
    tool_prompt = ChatPromptTemplate.from_template(tool_prompt_template)

    action_designator1 = state['action_designator']
    reason_for_failure1 = state['reason_for_failure']
    human_comment1 = state['human_comment']

    print(action_designator1)
    print(reason_for_failure1)
    print(human_comment1)
    original_action_designator = ""

    try:
        if isinstance(reason_for_failure1, str):
            failure_instance = eval(reason_for_failure1)
            failure_type = failure_instance.failure_type
            failure_cls = next((cls for cls in failure_reasons if
                                cls.__name__ == failure_type or getattr(cls, 'failure_type', None) == failure_type),
                               None)
            if failure_cls is None:
                raise ValueError(f"Unknown failure_type: {failure_type}")
            print("failure Class", failure_cls, type(failure_cls))
            error_message = failure_instance.args[0]
            print(f"Error Message {error_message}")
    except:
        print("invalid failure type")

    try:
        if isinstance(action_designator1, str):
            ad = eval(action_designator1)
            original_action_designator = str(ad)
            action_type = ad.action_type
            action_cls = next((cls for cls in action_classes if
                               cls.__name__ == action_type or getattr(cls, 'action_type', None) == action_type), None)
            if action_cls is None:
                raise ValueError(f"Unknown action_type: {action_type}")

            print("Original Action Designator : ", ad)
            print("action Class", action_cls, type(action_cls))
            structured_ollama = ollama_llm.with_structured_output(action_cls, method="json_schema")
    except:
        print("Unknown Action Designator")

    chain = tool_prompt | structured_ollama

    response = chain.invoke({"action_designator" : original_action_designator, "reason_for_failure" : reason_for_failure1,
                             "human_comment" : human_comment1})

    print("model response : ", response)

    # return response.model_dump()
    return  {"updated_action_designator" : response}

#SoleGraphbuilder

graph_builder = StateGraph(CustomState)
graph_builder.add_node("updater", updater_node)
graph_builder.set_entry_point("updater")
graph_builder.add_edge("updater", END)
sole = graph_builder.compile()

if __name__ == "__main__":
    test_obj = ObjectModel(name="cup",concept="cup", color="blue")
    test_robot = ObjectModel(name="robot", concept="robot")
    test_links = [Link(name="gripper_link"), Link(name="wrist_link")]
    test_pose = PoseStamped(pose=Pose(
        position=Vector3(x=1.0, y=2.0, z=3.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)))

    action_designator = ("PickUpAction(object_designator=test_obj, arm=Arms.LEFT, "
                         "grasp_description=GraspDescription(approach_direction=Grasp.TOP,vertical_alignment=Grasp.TOP, rotate_gripper=True))")
    grasping_error = "ObjectNotGraspedErrorModel(obj=test_obj, robot=test_robot, arm=Arms.LEFT, grasp=Grasp.TOP)"
    human_comment = "pick up the yellow bottle not the blue cup"

    for s in sole.stream({"action_designator": action_designator, "reason_for_failure": grasping_error,
                 "human_comment": human_comment}):
        print(s)



    #
    # # contact_error = ObjectStillInContactModel(
    # #     obj=test_obj,
    # #     contact_links=test_links,
    # #     placing_pose=test_pose,
    # #     robot=test_robot,
    # #     arm=Arms.RIGHT
    # # )
    # #
    # # target_location_error = ObjectNotPlacedAtTargetLocationModel(
    # #     obj=test_obj,
    # #     placing_pose=test_pose,
    # #     robot=test_robot,
    # #     arm=Arms.LEFT
    # # )
    #
    # designator_updater.invoke({"action_designator" : action_designator, "reason_for_failure" : grasping_error,
    #                            "human_comment" : "i want to pick up the yellow cup not the blue one"})



    # Tool Inspection
    # print(designator_updater.name)
    # print(designator_updater.description)
    # print(designator_updater.args)
    # print(designator_updater.return_direct)

    # print(designator_updater_dump.invoke({"action_designator" : action_designator, "reason_for_failure" : grasping_error,
    #                            "human_comment" : human_comment}))

    # config = {"configurable" : {"thread_id" : 1}}
    # print(update_agent.invoke({"action_designator" : action_designator, "reason_for_failure" : grasping_error,
    #                            "human_comment" : human_comment}))
