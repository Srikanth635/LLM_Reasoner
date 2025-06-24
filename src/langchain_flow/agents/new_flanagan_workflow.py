from src.langchain_flow.llm_configuration import *
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Literal, Dict, Union, TypedDict
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from src.langchain_flow.state_graph import *

new_flanagan_memory = MemorySaver()


class PhasePlanner(BaseModel):
    phases: List[Literal[
        "Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice",
        "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"
    ]]

class NormalizedPhases(BaseModel):
    normalized_phases: List[Literal[
        "Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice",
        "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"
    ]]

class PhasePreconditionsMap(BaseModel):
    phase_preconditions: Dict[str, Dict[str, Union[bool, str]]]

class ForceProfile(BaseModel):
    type: str
    expected_range_N: Union[List[float], None] = None
    expected_range_Nm: Union[List[float], None] = None

class ForceDynamics(BaseModel):
    contact: bool
    motion_type: str
    force_exerted: str
    force_profile: ForceProfile

class ForceDynamicsMap(BaseModel):
    force_dynamics: Dict[str, ForceDynamics]

class GoalStateMap(BaseModel):
    goal_states: Dict[str, Dict[str, Union[bool, str]]]

class SensoryFeedbackMap(BaseModel):
    sensory_feedback: Dict[str, Dict[str, Union[bool, str]]]

class FailureRecoveryMap(BaseModel):
    failure_and_recovery: Dict[str, Dict[str, List[str]]]

class PhaseTiming(BaseModel):
    max_duration_sec: float
    urgency: Literal["low", "medium", "high"]

class TemporalConstraintsMap(BaseModel):
    temporal_constraints: Dict[str, PhaseTiming]

class FlanaganStateInternal(TypedDict):
    instruction: str
    initial_phases : List[str]
    phases: List[str]
    preconditions: Dict[str, dict]
    goal_states: Dict[str, dict]
    force_dynamics: Dict[str, ForceDynamics]
    sensory_feedbacks: Dict[str, dict]
    failure_and_recovery: Dict[str, dict]
    temporal_constraints: Dict[str, PhaseTiming]
    flanagan : Dict


task_decomposer_prompt_template = """
    You are a robotic reasoning engine. Your task is to decompose a natural language instruction into a sequence of symbolic motion phases that describe how a
    robot would perform the task.

    Use only the following standardized action phase vocabulary:
    ["Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice", "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"]
    These labels represent atomic, purposeful phases in a robot’s action execution plan.
    
    Your job is to:
    
    Break the instruction down into ordered symbolic phases using only the labels above.
    
    Ensure the sequence reflects the logical and physical structure of the action.
    
    Output the result as a JSON object with a key "phases" and value as a list of strings.
    
    Do not add any explanation or extra fields. Do not invent new phase names.
    
    
    Examples:
    <instruction>: pick up the box
    <phases>: {{
      "phases": ["Approach", "Grasp", "Lift", "Withdraw"]
    }}
    
    <instruction>: cut the apple
    <phases>: {{
      "phases": ["Approach", "Grasp", "Lift", "Transport", "Stabilize", "Cut", "Slice", "Withdraw"]
    }}

    
    <instruction>: pour water into a glass
    <phases>: {{
      "phases": ["Approach", "Grasp", "Lift", "Transport", "Align", "Pour", "Reorient", "Return", "Release", "Withdraw"]
    }}

    ---
    
    Now, Perform similar operation on the given instruction :
    
    <instruction>: {instruction}
    <phases>: JSON_PHASE_OUTPUT
    
"""

phase_normalization_prompt_template = """
    You are a robotic control reasoning agent. You are given a list of raw action phases describing steps of a physical task.

    Your job is to map each raw phase to its closest symbolic match from the following normalized vocabulary:
    ["Approach", "Grasp", "Lift", "Transport", "Place", "Align", "Cut", "Slice", "Tilt", "Pour", "Insert", "Withdraw", "Release", "Reorient", "Stabilize", "Inspect"]
    
    For each raw phase, choose the closest semantic match.

    Output a list of normalized phase names in order.
    
    Do not invent new labels. Do not skip items.
    
    Return your answer as: {{ "normalized_phases": [...] }}
    
    ---
    
    Now, Perform the operation on the given list of action phases,
    
    action_phases : {action_phases}
"""

precondition_generator_prompt_template = """
    You are a robotics task planning engine.
    Given:

    - A task instruction that describes a robot's goal.
    
    - A list of normalized motion phases required to execute that task.
    
    Your job is to generate a symbolic dictionary of preconditions for each phase based on the instruction context.

    You must use structured, symbolic, key–value pairs. Output your result in this format:
    {{
      "phase_preconditions": {{
        "PhaseName1": {{ "condition1": true, "condition2": "value", ... }},
        "PhaseName2": {{ ... }},
        ...
      }}
    }}

    
    - All condition keys must be symbolic, lowercase, underscore_separated.

    - Use Boolean values or short symbolic strings.
    
    - Do not include effects or natural language commentary.
    
    - Only include conditions that must be true before each phase begins.
    
    ### Standard Format of Preconditions ###
    - Symbolic key–value pairs:
    
    ** Boolean flags (gripper_open: true) **
    
    ** Simple conditionals (pose_known) **
    
    ** Declarative robot-world relations (object_stable: true) **
    
    
    Examples:
    "phase": "Lift"
    "instruction": "pick up the box"
    OUTPUT: {{
      "preconditions": {{
        "object_grasped": true,
        "arm_vertical_clearance": true,
        "no_obstacle_above": true
      }}
    }}
    
    "phase": "Grasp"
    "instruction": "pour water into a mug"
    OUTPUT: {{
      "preconditions": {{
        "within_grasp_range": true,
        "object_stationary": true,
        "gripper_open": true,
        "grasp_pose_known": true
      }}
    }}

    ---
    
    Now perform the operation on the given instruction and list of action phases :
    
    instruction : {instruction} \n
    action_phases : {action_phases}

    
"""

force_dynamics_prompt_template = """
    You are a robotics control analyst.
    Given a robotic task instruction and a list of symbolic action phases, your job is to estimate the force dynamics involved in each phase.
    
    For each phase, return:
    
    - Whether contact occurs
    
    - The type of motion (linear, rotational, gripper, etc.)
    
    - The kind of force or torque involved
    
    - An estimated range in Newtons (expected_range_N) or Newton-meters (expected_range_Nm)
    
    - Adapt the probable force values to the object in action from the instruction
    
    Format your response like this:
    {{
      "force_dynamics": {{
        "PhaseName": {{
          "contact": true,
          "motion_type": "type",
          "force_exerted": "label",
          "force_profile": {{
            "type": "label",
            "expected_range_N": [min, max]
          }}
        }}
      }}
    }}
    
    - Use expected_range_Nm for torque-based phases (e.g. rotation).

    - Use realistic values — a human grasp might use 5–15N; pouring torque might be 0.3–0.8Nm.
    
    - Do not invent additional keys. Use only JSON.
    
    --
    
    Now perform the operation on the given context containing instruction, list of action phases and preconditions of each phase:
    
    instruction : {instruction} \n
    action_phases : {action_phases} \n
    preconditions : {preconditions}
    
"""

goal_state_generator_prompt_template = """
    You are a robotic reasoning engine.

    Given:
    
    - A task instruction
    
    - A list of motion phases
    
    - The symbolic preconditions and force dynamics already associated with each phase
    
    Your job is to produce the goal state that will be true after each phase is successfully executed.
    
    Output your result as a JSON dictionary:
    {{
      "goal_states": {{
        "PhaseName": {{
          "symbolic_condition_1": true,
          "symbolic_condition_2": "value",
          ...
        }}
      }}
    }}
    
    Use symbolic lowercase keys with underscores. Values can be booleans or symbolic strings.
    Do not repeat preconditions or force descriptions — describe the resulting world/robot state.
    
    Examples:
    Input: {{
      "instruction": "cut a cucumber",
      "phases": ["Grasp", "Align", "Cut"],
      "preconditions": {{
        "Grasp": {{ "object_stationary": true, "gripper_open": true }},
        "Align": {{ "object_grasped": true }},
        "Cut": {{ "knife_contact_ready": true, "object_aligned": true }}
      }},
      "force_dynamics": {{
        "Cut": {{
          "contact": true,
          "motion_type": "vertical_slicing_motion",
          "force_exerted": "cutting_force",
          "force_profile": {{ "type": "shear_force", "expected_range_N": [15, 25] }}
        }}
      }}
    }}
    
    Output:
    {{
      "goal_states": {{
        "Grasp": {{ "object_grasped": true }},
        "Align": {{ "object_aligned": true }},
        "Cut": {{ "object_divided": true, "cut_plane_completed": true }}
      }}
    }}

    ---
    
    Now, Perform the operation on the given context containing the instruction, action phases, already generated preconditions and force dynamics of each phase:
    
    instruction : {instruction} \n
    action_phases : {action_phases} \n
    preconditions : {preconditions} \n
    force_dynamics : {force_dynamics}
    
    
"""

sensory_feedback_predictor_prompt_template = """
    You are a robotic sensor integration assistant.

    Your task is to predict what sensor feedback should be observed during the execution of each robot motion phase.
    
    Given:
    
    - A task instruction
    
    - A list of phases
    
    - The preconditions, goal state and force dynamics for each phase
    
    Output a JSON object like this:
    {{
      "sensory_feedback": {{
        "PhaseName": {{
          "sensor_signal": value,
          ...
        }}
      }}
    }}
    
    ### Guidelines: ###

    - Use symbolic signals like tactile_contact, load_sensor_change, vision_pose_verified, audio_pour_sound, etc.
    
    - Values should be true, "detected", "within_range", etc.
    
    - Don’t repeat force or goal state information — focus only on feedback the robot should sense in real time.
    
    - Use lowercase keys with underscores.
    
    Example:
    Input = {{
      "instruction": "pour water into a mug",
      "phases": ["Grasp", "Pour"],
      "goal_states": {{
        "Grasp": {{ "object_grasped": true }},
        "Pour": {{ "fluid_transferred_to_target": true }}
      }},
      "force_dynamics": {{
        "Grasp": {{
          "contact": true,
          "force_exerted": "gripping_force"
        }},
        "Pour": {{
          "motion_type": "controlled_tilt",
          "force_exerted": "pouring_torque"
        }}
      }}
    }}
    
    Output = {{
      "sensory_feedback": {{
        "Grasp": {{
          "tactile_contact_detected": true,
          "grip_force_within_range": true
        }},
        "Pour": {{
          "fluid_flow_audio": "detected",
          "vision_confirms_fill_level": true
        }}
      }}
    }}
    
    ---
    
    Now, Perform the operation on the given context containing the instruction, action phases, already generated preconditions, force dynamics and goal states of each phase:
    
    instruction : {instruction} \n
    action_phases : {action_phases} \n
    preconditions : {preconditions} \n
    force_dynamics : {force_dynamics} \n
    goal_states : {goal_states}
    
"""

failure_recovery_prompt_template = """
    You are a robotic fault prediction and recovery reasoning assistant.

    For each motion phase in a robot task:
    
    - Identify what could go wrong based on the motion and sensory context.
    
    - Suggest symbolic recovery strategies the robot could attempt.
    
    Use symbolic phrasing like object_slipped, grip_force_too_low, no_tactile_feedback, etc.
    
    For each failure mode, suggest simple, plausible symbolic recovery strategies like:
    
    - retry_grasp
    
    - adjust_tilt_angle
    
    - pause_and_stabilize
    
    - fallback_to_known_pose
    
    Format your output like this:
    {{
      "failure_and_recovery": {{
        "PhaseName": {{
          "possible_failures": [...],
          "recovery_strategies": [...]
        }}
      }}
    }}
    
    Example:
    Input = {{
      "instruction": "cut an apple",
      "phases": ["Grasp", "Cut"],
      "force_dynamics": {{
        "Cut": {{
          "contact": true,
          "force_profile": {{
            "type": "shear_force",
            "expected_range_N": [15, 30]
          }}
        }}
      }},
      "sensory_feedback": {{
        "Cut": {{
          "resistance_detected": true
        }}
      }}
    }}
    
    Output = {{
      "failure_and_recovery": {{
        "Grasp": {{
          "possible_failures": [
            "object_out_of_grasp_range",
            "gripper_misaligned"
          ],
          "recovery_strategies": [
            "reposition_gripper",
            "adjust_grasp_pose",
            "retry_approach"
          ]
        }},
        "Cut": {{
          "possible_failures": [
            "resistance_not_detected",
            "knife_slips_off_surface"
          ],
          "recovery_strategies": [
            "increase_downward_force",
            "realign_cutting_angle",
            "reapply_contact"
          ]
        }}
      }}
    }}
    
    ---
    
    Now, Perform the operation on the given context containing the instruction, action phases, already generated force dynamics, sensory feedback, goal states
    and expected sensory feedbacks:
    
    instruction : {instruction} \n
    action_phases : {action_phases} \n
    preconditions : {preconditions} \n
    force_dynamics : {force_dynamics} \n
    goal_states : {goal_states} \n
    expected_sensory_feedbacks : {sensory_feedback}
    
"""

temporal_constraints_prompt_template = """
    You are a robotic control timing advisor.

    ### For each robot motion phase, estimate: ###
    
    - An upper bound for safe execution time (in seconds)
    
    - The urgency level of the action:
    
        - "low": no pressure or risk
    
        - "medium": should proceed smoothly
    
        - "high": time-sensitive (e.g. liquid, safety, balance)
    
    ### Consider: ###
    
    - Task instruction
    
    - Force dynamics (e.g. if high torque or fast motion is involved)
    
    - Object type and precision required
    
    Respond in this JSON format:
    {{
      "temporal_constraints": {{
        "PhaseName": {{
          "max_duration_sec": float,
          "urgency": "low" | "medium" | "high"
        }}
      }}
    }}
    
    Example:
    Input = {{
      "instruction": "cut an apple in half",
      "phases": ["Grasp", "Cut", "Withdraw"],
      "force_dynamics": {{
        "Cut": {{
          "force_profile": {{
            "type": "shear_force",
            "expected_range_N": [20, 35]
          }}
        }}
      }}
    }}
    
    Output = {{
      "temporal_constraints": {{
        "Grasp": {{
          "max_duration_sec": 1.2,
          "urgency": "medium"
        }},
        "Cut": {{
          "max_duration_sec": 2.5,
          "urgency": "high"
        }},
        "Withdraw": {{
          "max_duration_sec": 1.0,
          "urgency": "low"
        }}
      }}
    }}
    
    ---
    
    Now, Perform the operation on the given context containing the instruction, action phases, already generated preconditions, force dynamics ,
    goal states, sensory feedbacks and failure and recovery strategies of each phase:
    
    instruction : {instruction} \n
    action_phases : {action_phases} \n
    preconditions : {preconditions} \n
    force_dynamics : {force_dynamics} \n
    goal_states : {goal_states} \n
    sensory_feedback : {sensory_feedback} \n
    failure_and_recovery : {failure_and_recovery}

"""


# Nodes

def task_decomposer_node(state : FlanaganStateInternal):
    instruction = state['instruction']
    chain = ChatPromptTemplate.from_template(task_decomposer_prompt_template) | ollama_llm_small.with_structured_output(
        PhasePlanner, method="json_schema")
    chain_out = chain.invoke({'instruction': instruction})
    return {'initial_phases' : chain_out.phases}

def phase_normalization_node(state : FlanaganStateInternal):
    action_phases = state['initial_phases']
    chain2 = ChatPromptTemplate.from_template(
        phase_normalization_prompt_template) | ollama_llm_small.with_structured_output(NormalizedPhases,
                                                                                       method="json_schema")

    chain2_out = chain2.invoke({'action_phases': action_phases})
    return {'phases' : chain2_out.normalized_phases}

def precondition_generator_node(state : FlanaganStateInternal):
    instruction = state['instruction']
    normalized_phases = state['phases']
    chain3 = ChatPromptTemplate.from_template(
        precondition_generator_prompt_template) | ollama_llm_small.with_structured_output(PhasePreconditionsMap,
                                                                                          method="json_schema")

    chain3_out = chain3.invoke(
        {'instruction': instruction, 'action_phases': normalized_phases})
    return {'preconditions' : chain3_out.phase_preconditions}


def force_dynamics_node(state : FlanaganStateInternal):
    instruction = state['instruction']
    normalized_phases = state['phases']
    preconditions = state['preconditions']
    chain4 = ChatPromptTemplate.from_template(force_dynamics_prompt_template) | ollama_llm_small.with_structured_output(
        ForceDynamicsMap, method="json_schema")

    chain4_out = chain4.invoke(
        {'instruction': instruction, 'action_phases': normalized_phases,
         'preconditions': preconditions})
    return {'force_dynamics' : chain4_out.force_dynamics}


def goal_state_generator_node(state : FlanaganStateInternal):
    instruction = state['instruction']
    normalized_phases = state['phases']
    preconditions = state['preconditions']
    force_dynamics = state['force_dynamics']
    chain5 = ChatPromptTemplate.from_template(
        goal_state_generator_prompt_template) | ollama_llm_small.with_structured_output(GoalStateMap,
                                                                                        method="json_schema")

    chain5_out = chain5.invoke(
        {'instruction': instruction, 'action_phases': normalized_phases,
         'preconditions': preconditions, 'force_dynamics': force_dynamics})
    return {'goal_states' : chain5_out.goal_states}


def sensory_feedback_predictor_node(state : FlanaganStateInternal):
    instruction = state['instruction']
    normalized_phases = state['phases']
    preconditions = state['preconditions']
    force_dynamics = state['force_dynamics']
    goal_states = state['goal_states']
    chain6 = ChatPromptTemplate.from_template(
        sensory_feedback_predictor_prompt_template) | ollama_llm_small.with_structured_output(SensoryFeedbackMap,
                                                                                              method="json_schema")

    chain6_out = chain6.invoke(
        {'instruction': instruction, 'action_phases': normalized_phases,
         'preconditions': preconditions, 'force_dynamics': force_dynamics,
         'goal_states': goal_states})
    return {'sensory_feedbacks' : chain6_out.sensory_feedback}


def failure_recovery_node(state : FlanaganStateInternal):
    instruction = state['instruction']
    normalized_phases = state['phases']
    preconditions = state['preconditions']
    force_dynamics = state['force_dynamics']
    goal_states = state['goal_states']
    sensory_feedback = state['sensory_feedbacks']
    chain7 = ChatPromptTemplate.from_template(
        failure_recovery_prompt_template) | ollama_llm_small.with_structured_output(FailureRecoveryMap,
                                                                                    method="json_schema")

    chain7_out = chain7.invoke(
        {'instruction': instruction, 'action_phases': normalized_phases,
         'preconditions': preconditions, 'force_dynamics': force_dynamics,
         'sensory_feedback': sensory_feedback, 'goal_states': goal_states})
    return {'failure_and_recovery' : chain7_out.failure_and_recovery}



def temporal_constraints_node(state : FlanaganStateInternal):
    instruction = state['instruction']
    normalized_phases = state['phases']
    preconditions = state['preconditions']
    force_dynamics = state['force_dynamics']
    goal_states = state['goal_states']
    sensory_feedback = state['sensory_feedbacks']
    failure_and_recovery = state['failure_and_recovery']
    chain8 = ChatPromptTemplate.from_template(
        temporal_constraints_prompt_template) | ollama_llm_small.with_structured_output(TemporalConstraintsMap,
                                                                                        method="json_schema")

    chain8_out = chain8.invoke(
        {'instruction': instruction, 'action_phases': normalized_phases,
         'preconditions': preconditions, 'force_dynamics': force_dynamics,
         'sensory_feedback': sensory_feedback, 'failure_and_recovery': failure_and_recovery,
         'goal_states': goal_states})
    return {'temporal_constraints' : chain8_out.temporal_constraints}


def composition_node(state : FlanaganStateInternal):
    phases = state['phases']
    preconditions = state['preconditions']
    goal_states = state['goal_states']
    force_dynamics = state['force_dynamics']
    sensory_feedbacks = state['sensory_feedbacks']
    failure_and_recovery = state['failure_and_recovery']
    temporal_constraints = state['temporal_constraints']
    instruction = state['instruction']

    result = {
        "instruction": instruction,
        "phases": []
    }

    for raw_phase in phases:
        phase = raw_phase[0].upper() + raw_phase[1:]
        lowercase_phase = raw_phase.lower()

        entry = {
            "phase": phase,
            "symbol": f"→[robot performs {phase.lower()}]",
            "goal_state": goal_states.get(phase, {}),
            "preconditions": preconditions.get(phase, {}),
            "force_dynamics": (
                force_dynamics[phase].model_dump()
                if phase in force_dynamics else {}
            ),
            "sensory_feedback": sensory_feedbacks.get(lowercase_phase, {}),
            "failure_and_recovery": failure_and_recovery.get(phase, {}),
            "temporal_constraints": (
                temporal_constraints[phase].model_dump()
                if phase in temporal_constraints else {}
            )
        }

        result["phases"].append(entry)

    return {'flanagan' : result}



fbuilder = StateGraph(FlanaganStateInternal)
fbuilder.add_node("task_decomposer", task_decomposer_node)
fbuilder.add_node("phase_normalization", phase_normalization_node)
fbuilder.add_node("precondition_generator", precondition_generator_node)
fbuilder.add_node("force_dynamics_generator", force_dynamics_node)
fbuilder.add_node("goal_state_generator", goal_state_generator_node)
fbuilder.add_node("sensory_feedback_predictor", sensory_feedback_predictor_node)
fbuilder.add_node("failure_recovery_predictor", failure_recovery_node)
fbuilder.add_node("temporal_constraints_predictor", temporal_constraints_node)
fbuilder.add_node("composition", composition_node)
fbuilder.set_entry_point("task_decomposer")

fbuilder.add_edge("task_decomposer", "phase_normalization")
fbuilder.add_edge("phase_normalization", "precondition_generator")
fbuilder.add_edge("precondition_generator", "force_dynamics_generator")
fbuilder.add_edge("force_dynamics_generator", "goal_state_generator")
fbuilder.add_edge("goal_state_generator", "sensory_feedback_predictor")
fbuilder.add_edge("sensory_feedback_predictor","failure_recovery_predictor")
fbuilder.add_edge("failure_recovery_predictor", "temporal_constraints_predictor")
fbuilder.add_edge("temporal_constraints_predictor", "composition")

new_flanagan_graph = fbuilder.compile(checkpointer=new_flanagan_memory)

# class FinalPhase(BaseModel):
#     phase: str
#     symbol: str
#     goal_state: Dict[str, Union[bool, str]]
#     preconditions: Dict[str, Union[bool, str]]
#     force_dynamics: Dict[str, Union[bool, str, Dict[str, Union[List[float], str]]]]
#     sensory_feedback: Dict[str, Union[bool, str]]
#     failure_and_recovery: Dict[str, List[str]]
#     temporal_constraints: Dict[str, Union[float, str]]
#
# class FlanaganActionDescriptor(BaseModel):
#     instruction: str
#     phases: List[FinalPhase]

def normalize_key(key: str) -> str:
    """Capitalize the first letter of a phase key"""
    return key[0].upper() + key[1:] if key else key

def generate_flanagan_descriptor(instruction: str,
    phases: List[str],
    preconditions: Dict[str, dict],
    goal_states: Dict[str, dict],
    force_dynamics: Dict[str, ForceDynamics],
    sensory_feedbacks: Dict[str, dict],
    failure_and_recovery: Dict[str, dict],
    temporal_constraints: Dict[str, PhaseTiming]) -> Dict:

    result = {
        "instruction": instruction,
        "phases": []
    }

    for raw_phase in phases:
        phase = raw_phase[0].upper() + raw_phase[1:]
        lowercase_phase = raw_phase.lower()

        entry = {
            "phase": phase,
            "symbol": f"→[robot performs {phase.lower()}]",
            "goal_state": goal_states.get(phase, {}),
            "preconditions": preconditions.get(phase, {}),
            "force_dynamics": (
                force_dynamics[phase].model_dump()
                if phase in force_dynamics else {}
            ),
            "sensory_feedback": sensory_feedbacks.get(lowercase_phase, {}),
            "failure_and_recovery": failure_and_recovery.get(phase, {}),
            "temporal_constraints": (
                temporal_constraints[phase].model_dump()
                if phase in temporal_constraints else {}
            )
        }

        result["phases"].append(entry)

    return result


def new_flanagan_node(state : ModelsStateInternal):
    print("NEW FLANAGAN NODE STATE ", state)
    instruction = state['instruction']
    # action_core = state['action_core']
    # enriched_json_attributes = str(state['enriched_action_core_attributes'])
    # cram_plan_response = state['cram_plan_response']

    final_flanagan_state = new_flanagan_graph.invoke({'instruction': instruction})

    # premotion_phase = final_flanagan_state['premotion_phase']
    # phaser = final_flanagan_state['phaser']
    flanagan = final_flanagan_state['flanagan']

    return {'flanagan' : flanagan}



if __name__ == "__main__":

    chain = ChatPromptTemplate.from_template(task_decomposer_prompt_template) | ollama_llm_small.with_structured_output(PhasePlanner, method="json_schema")

    chain_out = chain.invoke({'instruction' : "pick up the box from the table"})

    chain2 = ChatPromptTemplate.from_template(phase_normalization_prompt_template) | ollama_llm_small.with_structured_output(NormalizedPhases, method="json_schema")

    chain2_out = chain2.invoke({'action_phases' : chain_out.phases})

    # print("Normalized phases",chain2_out.normalized_phases)

    chain3 = ChatPromptTemplate.from_template(precondition_generator_prompt_template) | ollama_llm_small.with_structured_output(PhasePreconditionsMap, method="json_schema")

    chain3_out = chain3.invoke({'instruction' : "pick up the box from the table", 'action_phases' : chain2_out.normalized_phases})

    # print("Preconditions",chain3_out.phase_preconditions)

    chain4 = ChatPromptTemplate.from_template(force_dynamics_prompt_template) | ollama_llm_small.with_structured_output(ForceDynamicsMap, method="json_schema")

    chain4_out = chain4.invoke({'instruction' : "pick up the box from the table", 'action_phases' : chain2_out.normalized_phases, 'preconditions' : chain3_out.phase_preconditions})

    # print("Force dynamics", chain4_out.force_dynamics)

    chain5 = ChatPromptTemplate.from_template(goal_state_generator_prompt_template) | ollama_llm_small.with_structured_output(GoalStateMap, method="json_schema")

    chain5_out = chain5.invoke({'instruction' : "pick up the box from the table", 'action_phases' : chain2_out.normalized_phases,
                                'preconditions' : chain3_out.phase_preconditions, 'force_dynamics' : chain4_out.force_dynamics})

    # print("Goal States", chain5_out.goal_states)

    chain6 = ChatPromptTemplate.from_template(sensory_feedback_predictor_prompt_template) | ollama_llm_small.with_structured_output(SensoryFeedbackMap, method="json_schema")

    chain6_out = chain6.invoke({'instruction' : "pick up the box from the table", 'action_phases' : chain2_out.normalized_phases,
                                'preconditions' : chain3_out.phase_preconditions, 'force_dynamics' : chain4_out.force_dynamics,
                                'goal_states' : chain5_out.goal_states})

    # print("Sensory feedbacks", chain6_out.sensory_feedback)

    chain7 = ChatPromptTemplate.from_template(failure_recovery_prompt_template) | ollama_llm_small.with_structured_output(FailureRecoveryMap, method="json_schema")

    chain7_out = chain7.invoke({'instruction' : "pick up the box from the table", 'action_phases' : chain2_out.normalized_phases,
                                'preconditions' : chain3_out.phase_preconditions, 'force_dynamics' : chain4_out.force_dynamics,
                                'sensory_feedback' : chain6_out.sensory_feedback, 'goal_states' : chain5_out.goal_states})

    # print("Failure and recovery", chain7_out.failure_and_recovery)

    chain8 = ChatPromptTemplate.from_template(temporal_constraints_prompt_template) | ollama_llm_small.with_structured_output(TemporalConstraintsMap, method="json_schema")

    chain8_out = chain8.invoke({'instruction' : "pick up the box from the table", 'action_phases' : chain2_out.normalized_phases,
                                'preconditions' : chain3_out.phase_preconditions, 'force_dynamics' : chain4_out.force_dynamics,
                                'sensory_feedback' : chain6_out.sensory_feedback, 'failure_and_recovery' : chain7_out.failure_and_recovery,
                                'goal_states' : chain5_out.goal_states})

    # print("Temporal constraints", chain8_out.temporal_constraints)

    flanagan = generate_flanagan_descriptor(
        "pick up the box from the table",
        phases= chain2_out.normalized_phases,
        preconditions=chain3_out.phase_preconditions,
        goal_states=chain5_out.goal_states,
        force_dynamics=chain4_out.force_dynamics,
        sensory_feedbacks=chain6_out.sensory_feedback,
        failure_and_recovery=chain7_out.failure_and_recovery,
        temporal_constraints=chain8_out.temporal_constraints
    )

    # print(flanagan)

