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

decomposer_prompt_template = """
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

decomposer_prompt = ChatPromptTemplate.from_template(decomposer_prompt_template)


def think_remover(res: str):
    if re.search(r"<think>.*?</think>", res, flags=re.DOTALL):
        cleaned_res = re.sub(r"<think>.*?</think>", "", res, flags=re.DOTALL).strip()
    else:
        cleaned_res = res.strip()

    return cleaned_res


try:
    ollama_llm = ChatOllama(model="qwen3:14b")
except:
    print("14b model not available, using 8b model")
    ollama_llm = ChatOllama(model="qwen3:8b")

# with open("feroz_context.txt", 'r') as f:
#     context = f.read()
#
#
# from pydantic import BaseModel, Field
#
# chain = system_prompt | ollama_llm
#
# print(chain.invoke({'context': context}))