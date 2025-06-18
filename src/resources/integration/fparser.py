from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
import re
from pydantic import BaseModel, Field
from typing import List


class AtomicsModel(BaseModel):
    atomics: List[str] = Field(description="Atomic robot instructions in imperative tone")


system_prompt_template = """
    You are an automated text-cleaning agent. Your sole purpose is to process an input document and remove all redundant
    information, returning only the essential, unique text.

    Your task is to:

    - Analyze the entire input context, which may include plain text, JSON, or other structured formats.

    - Identify and eliminate redundant text. Redundancy includes:

        - Exact Duplicates: Identical phrases, sentences, or paragraphs that appear more than once.

        - Semantic Duplicates: Text that conveys the exact same meaning using slightly different words. You must be vigilant in
            identifying these near-identical statements.

    - Preserve the first instance of any piece of information and remove all subsequent repetitions.

    - Return only the cleaned text.

    ### Critical Rules: ###

    - Do NOT add any new words, sentences, explanations, or summaries. Your output must consist only of content from the original input.

    - Do NOT alter the meaning of the original content. Only remove superfluous parts.

    - Do NOT explain your actions. Perform the cleaning task silently and provide only the final text as output.

    ## Example of Semantic Redundancy Detection: ##

    - If the input contains the following lines:

    Original: In Segment 4, the person is still cutting the brown onion with the knife on the wooden cutting board.

    Original: In Segment 5, the person keeps cutting the brown onion with the knife on the wooden cutting board.

    You should recognize that these two sentences describe the same action. Your processed output should retain only the first instance and discard the second.

    Input:

    In Segment 4, the person is still cutting the brown onion with the knife on the wooden cutting board.
    In Segment 5, the person keeps cutting the brown onion with the knife on the wooden cutting board.

    Correct Output:

    In Segment 4, the person is still cutting the brown onion with the knife on the wooden cutting board.

    Process the given context and return the cleaned version.

    context : {context}
"""

system_prompt = ChatPromptTemplate.from_template(system_prompt_template)

decomposer_prompt_template_cut = """
    You are a highly focused LLM agent in a LangChain workflow. Your purpose is to process human-written action descriptions
    and transform them into concise, robot-friendly instructions.

    ✅ Your Task:
    Transform the input text into a numbered list of atomic, imperative instructions with no inferred actions.

   📌 Output Rules (Strict):
    - Only extract actions explicitly described in the input.

        - Do not add implied, inferred, or auxiliary steps (e.g., “Move to Segment 8”, “Pick up the knife”) unless they are clearly stated.

        - If the input contains only one action, return only one instruction.

    - Always write in imperative tone.

        - Use direct command form: e.g., "Peel the brown onion", not "Begin peeling..." or "The person peels...".

    - Segment only when multiple actions are joined.

        - Use punctuation, conjunctions, or verb phrases to identify compound actions.

        - If the input is already atomic, do not split or expand it further.

        - Do not fabricate preparatory steps.

    - Resolve all references and pronouns.

        - Replace "it", "this", "them", etc. with the correct noun from context.

    🧾 Input
    A natural language paragraph or sentence describing a series of actions, possibly with pronouns and compound verbs.

    Example Input:
    "The person grabs the mango and kept it on the table."

    ✅ Output
    A numbered list of simplified robot instructions with no ambiguous references.

    Example Output:

    Grab the mango.

    Place the mango on the table.

    🚫 Do NOT:
    - Add steps like "Move to...", "Pick up...", "Place..." unless they are explicitly present in the input.

    - Infer tool use, intent, or sequence not clearly described.

    - Include commentary or formatting beyond the numbered steps.

    ---

    Now, perform the task on the given input instruction:

    instruction : {instruction}

    /nothink
"""

decomposer_prompt_cut = ChatPromptTemplate.from_template(decomposer_prompt_template_cut)

decomposer_prompt_template = """
    You are a task planning agent that generates clear, step-by-step robotic instructions based on video segment descriptions .

    Your goal is to:
    
    - Understand the overall objective from the video segments.
    - Generate a numbered list of robotic actions using only the allowed action classes listed below.
    - Use imperative form for each instruction (e.g., "Pick up...", "Pour...", "Place...").
    - Output only the numbered list — no explanations, no markdown, no extra text.
    - Always follow the same flow of operations using the same entities followed in the original segment data.
    
    ⚠️ Important Guidelines:
    
    - Single-Handed Robot Assumption (Default):
        - The robot has only one hand unless otherwise specified.
        - This means: the robot must place an object before picking up another .
        - Plan actions accordingly to avoid impossible sequences.
        
    -Action Sequence Matters:
        - The order of actions must reflect the chronological flow shown in the video segments.
        - If pouring into a bowl is described, ensure that picking up and placing the bowl happen before the pouring action.
        - Always maintain awareness of what the robot is holding or doing at each step.
        
    - Common Sense & Realism:
        - Ensure logical sequencing based on physical constraints (e.g., can't pour without first having the container)..
        - Avoid over-interpreting or adding steps not directly suggested.
        - Ensure realistic and physically possible sequences.
    
    - Avoid Redundancy:
        - If two or more segments describe the same ongoing action (e.g., continuous pouring), output only one representative action — typically the first one .
        - Do not repeat actions unless there’s a clear change in object, location, or action type.
        
    ✅ Allowed Action Classes:
    [Peeling, Cutting,
    PickingUp, Lifting, Opening, OperatingATap, Pipetting, Pouring, Pressing,
    Pulling, Placing, Removing, Rolling, Shaking, Spooning, Sprinkling, Stirring,
    Taking, Turning, Unscrewing, Waiting]
    
    ---
    
    You will receive multiple video segment descriptions in this format:
    
    [Video Segment Descriptions]
    Segment 1: ...
    Segment 2: ...
    ...
    Segment N: ...
    
     Based on them, output the robotic instructions as a numbered list.
    
    ✅ Example Output Format:
    - Pick up the red cup
    - Place the cup on the table
    - Pick up the milk bottle
    - Place the milk bottle near the cup
    - Pour the milk from the milk bottle into the cup
    - Evaluate the pouring process to ensure even distribution
    - Place the empty milk bottle back on the table

    🚫 Do NOT:
    - Add steps like "Move to...", "Pick up...", "Place..." unless they are explicitly present in the input.

    - Infer tool use, intent, or sequence not clearly described.

    - Include commentary or formatting beyond the numbered steps.

    ---

    Now, perform the task on the given segments data:

    segments : {segments}
    
    /nothink
"""

decomposer_prompt = ChatPromptTemplate.from_template(decomposer_prompt_template)


# decomposer_reflection_prompt_template = """
#     You are a Reflectance Agent. Your role is to refine a list of step-by-step, atomic, imperative instructions produced by another agent. These
#     instructions describe how to complete a task using only allowed action classes.
#
#     Your goal is to return a concise, non-redundant, and logically complete set of instructions that preserves task accuracy while removing unnecessary or placeholder content.
#
#     🧠 Your Responsibilities
#     - Filter Redundant Instructions
#         - Remove instructions that are unnecessarily repeated or semantically equivalent to earlier steps.
#
#     - Remove Placeholder or Non-Actionable Instructions
#         - Eliminate vague instructions like “continue pouring” or “make sure it is steady” if they do not add distinct functional value.
#         - Retain only actions that directly change the task state (e.g., grasp, lift, pour, place).
#
#     - Ensure Atomicity
#         - Confirm that each instruction remains atomic: one action on one object.
#
#     - Preserve Logical Task Flow
#         - Maintain the correct chronological and functional sequence needed to complete the inferred task.
#
#     - Preserve Specificity and Action Class Alignment
#         - Ensure each instruction is specific, refers to the correct object(s), and implicitly aligns with a valid action class from the original list.
#
#     🧾 Input Format
#     Inferred Task Goal: [One sentence]
#
#     Instructions:
#     1. [Instruction 1]
#     2. [Instruction 2]
#     ...
#
#     📤 Output Format
#     Return a cleaned version in the same format, with no extra commentary.
#
#     Inferred Task Goal: [Refined version of task goal, if needed]
#
#     Refined Instructions:
#     1. [Cleaned Instruction 1]
#     2. [Cleaned Instruction 2]
#
#     ...
#
#     ⚠️ DO NOT:
#     - Add any new instructions not in the original list
#     - Change the meaning of any instruction
#     - Generalize or paraphrase — retain exact object references and imperative tone
#
#     ---
#
#     Now, perform the task on the given instruction context:
#
#     instruction_context : {instruction_context}
# """
#
# decomposer_reflection_prompt = ChatPromptTemplate.from_template(decomposer_reflection_prompt_template)

def think_remover(res: str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()

    return cleaned_res


# try:
#     ollama_llm = ChatOllama(model="qwen3:8b")
# except:
#     print("14b model not available, using 8b model")
#     ollama_llm = ChatOllama(model="qwen3:8b")

# with open("feroz_context.txt", 'r') as f:
#     context = f.read()
#
#
# from pydantic import BaseModel, Field
#
# chain = system_prompt | ollama_llm
#
# print(chain.invoke({'context': context}))