import re
import json


def parse_segment_data(text_data):
    """
    Parses a string containing segment data with original text and JSON
    into a list of dictionaries.

    Args:
        text_data (str): The raw string containing the segment information.

    Returns:
        list: A list of dictionaries, where each dictionary contains an
              'instruction' and an 'action_designator' key.
              Returns an empty list if no valid segments are found.
    """
    # This improved regular expression is the core of the parser.
    # It now looks for the start of each segment ('=== Segment ...') to correctly
    # isolate the instruction and the corresponding JSON block.
    # The JSON block capture is more robust, ensuring the entire object is matched.
    pattern = re.compile(
        r"=== Segment \d+ ===\s*Original:\s*(.*?)\s*Generated JSON:\s*(\{.*?\n\})",
        re.DOTALL
    )

    # Find all non-overlapping matches of the pattern in the string.
    # This returns a list of tuples, where each tuple contains the captured groups.
    matches = pattern.findall(text_data)

    parsed_list = []
    # Iterate through each match found.
    # 'instruction_text' will be the first captured group (the sentence).
    # 'json_string' will be the second captured group (the JSON object as a string).
    for instruction_text, json_string in matches:
        try:
            # Clean up the instruction text by removing leading/trailing whitespace.
            instruction = instruction_text.strip()

            # Parse the JSON string into a Python dictionary.
            action_designator = json.loads(json_string)

            # Create the final dictionary structure as requested.
            segment_dict = {
                'instruction': instruction,
                'action_designator': action_designator
            }
            parsed_list.append(segment_dict)

        except json.JSONDecodeError as e:
            # Handle cases where a JSON block might be malformed.
            print(f"Skipping a segment due to a JSON parsing error: {e}")
            continue

    return parsed_list


def parse_summary_data(text_data):
    """
    Parses summary text to extract segment descriptions and single-word actions
    into two separate dictionaries.

    Args:
        text_data (str): The raw string containing the summary lines.

    Returns:
        list: A list containing two dictionaries:
              1. A dictionary of segment descriptions.
              2. A dictionary of segment actions.
    """
    # Dictionary to hold descriptions like:
    # {'Segment 1': 'the person picks up a brown onion...'}
    descriptions_dict = {}
    # Regex to find lines starting with "In Segment..." and capture the segment
    # number and the rest of the sentence.
    description_pattern = re.compile(r"In (Segment \d+), (.*?)\s*\n")
    description_matches = description_pattern.findall(text_data)
    for segment, text in description_matches:
        descriptions_dict[segment] = text.strip()

    # Dictionary to hold actions like:
    # {'Segment 1': 'Picking'}
    actions_dict = {}
    # Regex to find lines formatted like "Segment 1: Picking" and capture
    # the segment number and the action word.
    action_pattern = re.compile(r"^(Segment \d+): (.*?)\s*$", re.MULTILINE)
    action_matches = action_pattern.findall(text_data)
    for segment, action in action_matches:
        actions_dict[segment] = action.strip()

    return [descriptions_dict, actions_dict]


def filter_redundant_actions(segment_list):
    """
    Filters a list of segment dictionaries to remove items with duplicate
    'action_designator' values. It keeps the first occurrence of each unique
    action designator.

    Args:
        segment_list (list): A list of dictionaries, where each dict has
                             at least an 'action_designator' key.

    Returns:
        list: A new list containing only the segments with unique
              action designators.
    """
    # This set will store a hashable representation of each 'action_designator'
    # dictionary that we have already seen.
    seen_actions = set()

    # This list will hold the unique dictionaries that we want to keep.
    unique_list = []

    # Iterate through each dictionary in the input list.
    for segment in segment_list:
        # Extract the 'action_designator' dictionary.
        action_designator = segment.get('action_designator')

        # If there's no action_designator, we can't process it, so skip.
        if action_designator is None:
            continue

        # Dictionaries are not hashable, so we can't add them to a set directly.
        # We convert the dictionary to a JSON string. Using sort_keys=True
        # ensures that dictionaries with the same content but different key
        # order are treated as identical.
        action_string = json.dumps(action_designator, sort_keys=True)

        # If we have NOT seen this action before...
        if action_string not in seen_actions:
            # Add its string representation to the set of seen actions.
            seen_actions.add(action_string)
            # Add the original, complete dictionary to our list of unique items.
            unique_list.append(segment)

    return unique_list


def filter_redundant_executed_actions(segment_list):
    """
    Filters a list to remove items where the combination of primary and
    secondary action names is a duplicate. It keeps the first occurrence
    of each unique action pair.

    Args:
        segment_list (list): A list of dictionaries, where each dict has
                             an 'action_designator' with an 'executed_action'.

    Returns:
        list: A new list containing only segments with unique executed
              action pairs.
    """
    # This set will store a tuple of (primary_action, secondary_action)
    # for each combination we have already encountered.
    seen_executed_actions = set()

    # This list will hold the unique dictionaries.
    unique_list = []

    for segment in segment_list:
        # Safely get the executed_action dictionary using .get() to avoid errors
        executed_action = segment.get('action_designator', {}).get('executed_action')

        if not executed_action:
            continue

        # Safely get the primary and secondary action names
        primary_name = executed_action.get('primary_action', {}).get('action_name')
        secondary_name = executed_action.get('secondary_action', {}).get('action_name')

        # If either action name is missing, we can't form a pair, so we skip it.
        if primary_name is None or secondary_name is None:
            continue

        # Create a tuple of the action names. Tuples are hashable and can be
        # stored in a set. The order is preserved as requested.
        action_pair = (primary_name, secondary_name)

        # If we have NOT seen this specific pair of actions before...
        if action_pair not in seen_executed_actions:
            # Add the pair to our set of seen actions.
            seen_executed_actions.add(action_pair)
            # Add the entire original dictionary to our unique list.
            unique_list.append(segment)

    return unique_list


def decompose_segments_with_atomic_actions(chain, segment_list):
    """
    Decomposes each segment into multiple segments based on a list of
    atomic instructions.

    Args:
        segment_list (list): The list of unique segment dictionaries.

    Returns:
        list: A new list with decomposed segments.
    """
    decomposed_list = []
    for original_segment in segment_list:
        # --- MOCK LLM CALL ---
        # This is where you would insert your call to `chain.invoke`.
        # For demonstration purposes, we'll simulate the output.
        com_ins = original_segment['instruction']
        # atomic_ins_list = []
        atomic_ins = chain.invoke({'instruction': com_ins})
        atomic_ins_list = atomic_ins.atomics
        # --- END MOCK ---

        # Iterate through the generated atomic instructions
        for atomic_instruction in atomic_ins_list:
            # Create a deep copy of the original segment to avoid modifying
            # it by reference. json.dumps/loads is a simple way to deep copy.
            new_segment = json.loads(json.dumps(original_segment))

            # Replace the instruction with the current atomic one
            new_segment['instruction'] = atomic_instruction

            # Add the new, decomposed segment to our final list
            decomposed_list.append(new_segment)

    return decomposed_list


if __name__ == "__main__":

    with open("./feroz_context.txt", 'r') as f:
        context = f.read()

    # The input text provided by the user.
    input_text = """
    Response:
    In Segment 1, the person picks up a brown onion and places it on a wooden cutting board.  
    In Segment 2, the person holds the knife and begins cutting the brown onion on the wooden cutting board.  
    In Segment 3, the person continues cutting the brown onion with the knife on the wooden cutting board.  
    In Segment 4, the person is still cutting the brown onion with the knife on the wooden cutting board.  
    In Segment 5, the person keeps cutting the brown onion with the knife on the wooden cutting board.  
    In Segment 6, the person continues cutting the brown onion with the knife on the wooden cutting board.  
    In Segment 7, the person is still cutting the brown onion with the knife on the wooden cutting board.  
    In Segment 8, the person switches to peeling the skin off the brown onion with the knife on the wooden cutting board.  
    In Segment 9, the person continues peeling the skin off the brown onion with the knife on the wooden cutting board.  
    In Segment 10, the person switches to peeling the skin off a white onion with their fingers on the wooden cutting board.  
    In Segment 11, the person continues peeling the skin off the white onion with their fingers on the wooden cutting board.
    Segment 1: Picking
    Segment 2: Holding
    Segment 3: Cutting
    Segment 4: Cutting
    Segment 5: Cutting
    Segment 6: Cutting
    Segment 7: Cutting
    Segment 8: Peeling
    Segment 9: Peeling
    Segment 10: Peeling
    Segment 11: Peeling

    === Segment 1 ===
    Original: In Segment 1, the person picks up a brown onion and places it on a wooden cutting board.  
    Generated JSON:
     {
      "component_id": "onion_01",
      "component_information": {
        "name": "onion",
        "id_number": "01",
        "component_type": "food_item",
        "shape": "spherical",
        "size": "medium",
        "handle": "no",
        "orientation": "upright",
        "weight": 1
      },
      "executed_action": {
        "primary_action": {
          "action_name": "picking_up",
          "hand": "right_hand"
        },
        "secondary_action": {
          "action_name": "placing",
          "hand": "right_hand"
        }
      },
      "grasp_descriptor": {
        "grasp_type": "spherical",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }

    === Segment 2 ===
    Original: In Segment 2, the person holds the knife and begins cutting the brown onion on the wooden cutting board.  
    Generated JSON:
     {
      "component_id": "knife_01",
      "component_information": {
        "name": "knife",
        "id_number": "01",
        "component_type": "kitchen_object",
        "shape": "flat",
        "size": "medium",
        "handle": "yes",
        "orientation": "upright",
        "weight": 1
      },
      "executed_action": {
        "primary_action": {
          "action_name": "cutting",
          "hand": "right_hand"
        },
        "secondary_action": {
          "action_name": "holding",
          "hand": "left_hand"
        }
      },
      "grasp_descriptor": {
        "grasp_type": "flat",
        "contact_points": "three_fingers",
        "holding_type": "one_handed",
        "hand_orientation": "top_to_bottom"
      },
      "environmental_factors": {
        "surface_conditions": "flat_surfaces"
      }
    }
    """

    # --- Example usage for the new function ---
    print("--- Summary Data ---")
    summary_data = parse_summary_data(context)
    # Pretty-print the final list of dictionaries for clear readability.
    if summary_data[0] and summary_data[1]:
        print(json.dumps(summary_data, indent=2))
    else:
        print("Could not parse summary data from the input text.")

    # --- Example usage for the original function ---
    print("\n--- Detailed Segment Data ---")
    detailed_data = parse_segment_data(context)
    # Pretty-print the final list of dictionaries for clear readability.
    if detailed_data:
        print(json.dumps(detailed_data, indent=2))
    else:
        print("Could not parse any detailed segments from the input text.")