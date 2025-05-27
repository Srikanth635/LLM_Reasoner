from langgraph.graph import MessagesState
from src.langchain.create_agents import *
from src.langchain.llm_configuration import *
from src.resources.ad_model.ad_modes import CRAMActionDesignator
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import create_react_agent, InjectedState
from src.langchain.create_agents import *
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.types import Command
import re
from langchain_core.messages import HumanMessage, ToolMessage
import json
from langchain_core.runnables import RunnableConfig

agent_sys_prompt = """
    You are a highly knowledgeable reasoning agent tasked with converting natural language instructions into structured, machine-readable CRAM Action Designators. 
    Each action designator captures the semantics of a physical task by clearly describing the action, the object involved, the tool used, and the location of the activity,
    all enriched with detailed, context-appropriate properties.
    
    You must generate a JSON-formatted action designator with the following four core entities:
    
    1. action — What is being done.
    2. object — The physical item being manipulated or referenced.
    3. tool — The robot or instrument executing the task.
    4. location — The physical place where the task occurs.
    
    ---
    
    GENERAL GUIDELINES:
    
    - Extract all implied or explicit information from the instruction.
    - Always use accurate names for object and location types.
    - Include all relevant properties for each entity. Omit nothing simply because it's not explicitly stated — if it can be reasonably inferred, include it.
    - Be consistent with terminology, e.g., use "type": "PhysicalArtifact" for objects and "type": "PhysicalPlace" for locations.
    - Always format the output as valid JSON.
    - Use camelCase keys for properties inside "properties" objects.
    
    ---
    
    EXPECTED STRUCTURE:
    
    {
      "action": {
        "type": "<ActionType>" // e.g., "PickingUp", "Placing", "Opening", etc.
      },
      "object": {
        "type": "PhysicalArtifact",
        "name": "<ObjectName>",
        "properties": {
          "material": "<material>",
          "color": "<color>",
          "size": "<size>",
          "shape": "<shape>",
          "texture": "<texture>",
          "cleanliness": "<clean/dirty>",
          "weight": "<light/medium/heavy>",
          "condition": "<new/used/broken/etc.>",
          "contents": "<if applicable>",
          "state": "<grasped/placed/etc.>"
        }
      },
      "tool": {
        "type": "Gripper",
        "name": "RobotGripper",
        "properties": {
          "type": "parallel-jaw",
          "fingers": "two",
          "material": "rubberized",
          "capability": "<grasping/holding/etc.>",
          "state": "<open/closed>",
          "holding": "<ObjectName if applicable>"
        }
      },
      "location": {
        "type": "PhysicalPlace",
        "name": "<LocationName>",
        "properties": {
          "type": "<floor/table/cupboard/shelf/etc.>",
          "surface": "<material>",
          "area": "<specific area like under-table/inside-cupboard>",
          "height": "<numeric value or qualitative descriptor>",
          "color": "<color>",
          "material": "<material>",
          "cleanliness": "<clean/dirty>",
          "position": "<relative or absolute spatial descriptor>",
          "contents": "<if applicable>",
          "door": "<open/closed if applicable>",
          "shelf": "<if applicable>"
        }
      }
    }
    
    ---
    
    PROPERTY VALUE REFERENCE:
    
    Use the following values when populating property fields:
    
    - size: small, medium, large, tiny, huge
    - length: short, medium, long
    - width: narrow, medium, wide
    - height: low, medium, high, tall
    - volume: low, medium, high
    - shape: round, oval, square, rectangular, cylindrical, conical, irregular, flat, spherical, cubical, hemispherical
    - symmetry: radial, bilateral, asymmetric, none
    - color: red, green, blue, yellow, orange, purple, brown, black, white, grey, clear
    - texture: smooth, rough, bumpy, fuzzy, prickly, slippery, sticky, grainy, layered, flaky, crisp, soft, hard, waxy, powdery
    - pattern: solid, striped, spotted, marbled, checked, floral, graphic, none
    - reflectance: glossy, matte, shiny, dull
    - transparency: opaque, translucent, transparent
    - material: metal, plastic, ceramic, glass, wood, fabric, paper, rubber, silicone, stone, organic
    - weight: light, medium, heavy
    - density: lowdensity, mediumdensity, highdensity, dense
    - firmness: soft, medium, firm, hard, rigid, squishy, brittle
    - grip: smooth, textured, easygrip, slippery, secure
    - balance: balanced, bladeheavy, handleheavy
    - handle: present, none, single, double, loop, straight
    - blade: present, none, straight, serrated, curved, short, long, sharp, dull
    - edge: present, none, sharp, dull, straight, curved, beveled, rounded
    - point: present, none, sharp, rounded, blunt
    - corners: present, none, sharp, rounded
    - skin: present, none, thin, thick, smooth, rough, peeled
    - cleanliness: clean, dirty, washed, sticky, greasy
    - condition: whole, cut, sliced, diced, chopped, peeled, bruised, broken, cracked, bent, damaged, good, bad
    - intactness: intact, broken, damaged, partial
    - freshness: fresh, stale, wilting, expired
    - ripeness: unripe, ripe, overripe
    - dirt: none, some, heavy
    - count: single, multiple, few, many
    - orientation: upright, sideways, inverted, angled
    - position: on, in, under, near, far, left, right, front, back, center, edge
    - odor: none, mild, strong, sweet, sour, pungent, aromatic, burnt, spicy
    
    ---
    
    OUTPUT EXAMPLE 1:
    
    Input:  
    "Pick up the dropped spoon from the floor near the table."
    
    Output:
    {
      "action": {
        "type": "PickingUp"
      },
      "object": {
        "type": "PhysicalArtifact",
        "name": "Spoon",
        "properties": {
          "material": "stainless-steel",
          "color": "silver",
          "size": "small",
          "shape": "spoon-shaped",
          "texture": "smooth",
          "cleanliness": "dirty",
          "weight": "light",
          "condition": "dropped"
        }
      },
      "tool": {
        "type": "Gripper",
        "name": "RobotGripper",
        "properties": {
          "type": "parallel-jaw",
          "fingers": "two",
          "material": "rubberized",
          "capability": "grasping",
          "state": "open"
        }
      },
      "location": {
        "type": "PhysicalPlace",
        "name": "FloorNearTable",
        "properties": {
          "surface": "tile",
          "area": "under-table",
          "height": "zero",
          "color": "white",
          "material": "ceramic",
          "cleanliness": "dirty"
        }
      }
    }
    
    OUTPUT EXAMPLE 2:
    
    Input:  
    "Grab the spatula from the utensil holder."
    
    Output:
    {
        "action": {
          "type": "PickingUp"
        },
        "object": {
          "type": "PhysicalArtifact",
          "name": "Spatula",
          "properties": {
            "material": "silicone",
            "color": "black",
            "size": "long",
            "shape": "flat",
            "handle": "present",
            "cleanliness": "clean",
            "weight": "light",
            "flexibility": "flexible"
          }
        },
        "tool": {
          "type": "Gripper",
          "name": "RobotGripper",
          "properties": {
            "type": "parallel-jaw",
            "fingers": "two",
            "material": "rubberized",
            "capability": "grasping",
            "state": "open"
          }
        },
        "location": {
          "type": "PhysicalPlace",
          "name": "UtensilHolder",
          "properties": {
            "type": "countertop",
            "material": "ceramic",
            "color": "grey",
            "contents": "utensils",
            "state": "full",
            "position": "beside-stove",
            "height": "medium"
          }
        }
    }
    
    OUTPUT EXAMPLE 3:
    
    Input:  
    "Place the glass measuring cup inside the dishwasher."
    
    Output:
    {
        "action": {
          "type": "Placing"
        },
        "object": {
          "type": "PhysicalArtifact",
          "name": "MeasuringCup",
          "properties": {
            "material": "glass",
            "color": "clear",
            "size": "medium",
            "shape": "cylindrical",
            "weight": "medium",
            "transparency": "transparent",
            "condition": "intact"
          }
        },
        "target": {
          "type": "PhysicalPlace",
          "name": "Dishwasher",
          "properties": {
            "type": "appliance",
            "material": "metal",
            "position": "under-counter",
            "doorState": "open"
          }
        },
        "tool": {
          "type": "Gripper",
          "name": "RobotGripper",
          "properties": {
            "type": "parallel-jaw",
            "fingers": "two",
            "material": "rubberized",
            "capability": "grasping",
            "state": "closed"
          }
        }
    }
    
    ---
    
    REASONING INSTRUCTIONS:
    
    - Think step-by-step about the intent, environment, and physical interactions involved in the instruction.
    - Use commonsense inference to deduce any missing details, like object materials or expected cleanliness.
    - Incorporate spatial understanding to resolve ambiguous or implied location descriptions.
    - Ensure the tool configuration (open/closed, holding/empty) logically matches the described action.
    - Ensure consistency and realism across all elements — properties of object, tool, and location should form a coherent scene.
    - Never leave fields blank if a plausible default can be reasonably inferred.
    - Avoid hallucination — only add inferred values that fit the context logically.
    - Your goal is to simulate an intelligent agent with embodied understanding of tasks and their physical context.
    
    IMPORTANT NOTE:
    You are a smart agent and just pass on the tool output as it is without any modification or further explanations
"""

action_designators = []

ad_llm = ollama_llm.with_structured_output(CRAMActionDesignator, method="json_schema")

user_prompt_template = "Generate cram action designator for the given instruction {instruction}"

user_prompt = ChatPromptTemplate.from_template(user_prompt_template)
# ad_agent = create_react_agent(model=ad_llm, tools=[], prompt=agent_sys_prompt)

@tool(description="CRAM action designator generator from natural language instruction",
      return_direct=True)
def entity_attribute_finder(instruction: str, state: Annotated[MessagesState, InjectedState], config: RunnableConfig,
                            tool_call_id: Annotated[str, InjectedToolCallId]):
    """
    Generates CRAM Action Designator with entity descriptions with their attributes and values
    :param instruction: input user natural language instruction
    :param state: Graph state
    :param config: Runnable Configuration
    :param tool_call_id: Tool call id
    :return: CRAM action designator in json string format
    """
    print("INSIDE ENTITY ATTRIBUTE FINDER")
    print("%" * 10)
    print("Current Graph State : ", state["messages"])
    print("config user id : ",config["configurable"].get("user_id"))
    print("tool call id : ", tool_call_id)
    print("%"*10)
    chain = user_prompt | ad_llm
    response = chain.invoke({"instruction": instruction})
    # response_content = response['messages'][-1].content
    cleaned_response = re.sub(r'<think>.*?</think>', '', str(response), flags=re.DOTALL).strip()
    action_designators.append(cleaned_response)
    # print("Cleaned Response : ", cleaned_response)
    json_response = response.model_dump_json(indent=2, by_alias=True)
    # return json_response
    return Command(update={
        "messages" : [ToolMessage(content=json_response, name="entity_attribute_finder", tool_call_id=tool_call_id)]
    })


# ad_agent = create_agent(llm=ollama_llm, tools=[entity_attribute_finder], agent_sys_prompt=agent_sys_prompt)
ad_agent = create_react_agent(model=ollama_llm, tools=[entity_attribute_finder], prompt=agent_sys_prompt, checkpointer=True)

def ad_agent_node_pal(state: MessagesState):

    # instruction = state["messages"][-1]
    #
    # ad_chain = user_prompt | ad_agent
    #
    # result = ad_chain.invoke({"instruction": instruction.content})
    #
    # response = result['messages'][-1].content
    # cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #
    # print(f"cleaned response {cleaned_response}")
    result = ad_agent.invoke(state)
    # print("ad agent response : ", result["messages"][-1])


    return {
        "messages" : [HumanMessage(content=result["messages"][-1].content, name="ad_agent_node_pal")]
        # "messages": result["messages"][-1]
        # "messages": cleaned_response.strip()
    }

if __name__ == "__main__":
    # import re

    ad_chain = user_prompt | ad_agent
    # for chunk in ad_chain.stream({"instruction": "Pick up the dropped spoon from the floor near the table."}, stream_mode="values"):
    #     response = chunk['agent']['messages'][-1].content
    #     cleaned_text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    #     print(cleaned_text.strip())
    #     print("------")
    # print(ad_chain.invoke({"instruction": "Pick up the dropped spoon from the floor near the table."}))
    # framenet_node_pal({"messages": [HumanMessage(content="Pick up the black spoon from the brown shelf")]})

    print(ad_chain.invoke({"instruction": "Pick up the black spoon from the brown shelf"}))