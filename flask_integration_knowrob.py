from flask import Flask, request, jsonify
from src.langchain_flow.agents.models_agent import *
from src.langchain_flow.agents.old_agents.reasoner_agent import *
import requests

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/generate' , methods=['POST'])
def generate_designator():
    try:
        # Get data from request (works with JSON or form-data)
        data = request.get_json() if request.is_json else request.form

        # Extract parameters
        _instruction = data.get('instruction')

        # Validate required fields
        if not _instruction:
            return jsonify({'error': 'Input instruction is required'}), 400

        # Model Invocation
        _config = {"configurable": {"thread_id": 1}}
        final_graph_state = pal_graph.invoke({"instruction": _instruction}, config=_config, stream_mode="updates")

        instruction = _instruction
        action_core = pal_graph.get_state(_config).values["action_core"]
        enriched_action_core_attributes = pal_graph.get_state(_config).values["enriched_action_core_attributes"]
        cram_plan_response = pal_graph.get_state(_config).values["cram_plan_response"]

        new_out = {}
        try:
            new_out['instruction'] = instruction
            new_out['action_core'] = action_core
            # new_out['action_core_attributes'] = json.loads(action_core_attributes)
            new_out['enriched_action_core_attributes'] = json.loads(enriched_action_core_attributes)
            new_out['cram_plan_response'] = cram_plan_response
            print("Parsed output normally")
        except:
            new_out['instruction'] = instruction
            new_out['action_core'] = action_core
            # new_out['action_core_attributes'] = out['action_core_attributes']
            new_out['enriched_action_core_attributes'] = enriched_action_core_attributes
            new_out['cram_plan_response'] = cram_plan_response
            print("Parsed output with strings")


        # model_response = {
        #     "instruction": instruction,
        #     "action_core": action_core,
        #     "cram_plan_response": cram_plan_response,
        #     "enriched_action_core_attributes" : enriched_action_core_attributes
        # }

        try:
            display_server_url = 'http://127.0.0.1:8082/display'
            server_b_response = requests.post(display_server_url, json=new_out)
            # Check if Server B responded successfully (e.g., status code 200 OK)
            server_b_response.raise_for_status()
            print("Displayed results")
        except Exception as e:
            print(f"Error displaying results: {e}")

        return jsonify(new_out), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/build' , methods=['POST'])
def build_designator_models():
    try:
        # Get data from request (works with JSON or form-data)
        data = request.get_json() if request.is_json else request.form

        # Extract parameters
        _instruction = data.get('instruction')
        _action_core = data.get('action_core')
        _enriched_action_core_attributes = data.get('enriched_action_core_attributes')
        _cram_plan_response = data.get('cram_plan_response')


        # Validate required fields
        if not _instruction:
            return jsonify({'error': 'Input instruction is required'}), 400

        # Model Invocation
        _config = {"configurable": {"thread_id": 2}}
        final_graph_state = model_graph.invoke({"instruction": _instruction,
                                                 "action_core": _action_core,
                                                 "enriched_action_core_attributes": _enriched_action_core_attributes,
                                                 "cram_plan_response": _cram_plan_response}, config=_config, stream_mode="updates")


        flanagan = model_graph.get_state(_config).values["flanagan"]
        framenet_model = model_graph.get_state(_config).values["framenet_model"]


        model_response = {
            "flanagan" : flanagan,
            "framenet_model" : framenet_model
        }

        try:
            display_server_url = 'http://127.0.0.1:8082/display'
            server_b_response = requests.post(display_server_url, json=model_response)
            # Check if Server B responded successfully (e.g., status code 200 OK)
            server_b_response.raise_for_status()
            print("Displayed results")
        except Exception as e:
            print(f"Error displaying results: {e}")

        return jsonify(model_response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reason' , methods=['POST'])
def reasoner():
    try:
        # Get data from request (works with JSON or form-data)
        data = request.get_json() if request.is_json else request.form

        # Extract parameters
        _query = data.get('query')
        _instruction = data.get('instruction')
        _enriched_action_core_attributes = data.get('enriched_action_core_attributes')
        _cram_plan_response = data.get('cram_plan_response')
        _flanagan = data.get('flanagan')
        _framenet_model = data.get('framenet_model')

        CONTEXT = (f'Instruction: {_instruction} \n Enriched_action_core_attributes: {_enriched_action_core_attributes} '
                   f'\n Cram_plan_response: {_cram_plan_response} \n Flanagan: {_flanagan} \n Framenet_model: {_framenet_model} ')

        # print("HUGE CONTEXT : " ,CONTEXT)
        # Invoke LLM
        _config = {"configurable": {"thread_id": 3}}

        print("Invoking Reasoner")
        response = invoke_reasoner(CONTEXT, _query)

        try:
            display_server_url = 'http://127.0.0.1:8082/display'
            server_b_response = requests.post(display_server_url, json=response)
            # Check if Server B responded successfully (e.g., status code 200 OK)
            server_b_response.raise_for_status()
            print("Displayed results")
        except Exception as e:
            print(f"Error displaying results: {e}")

        print("Reasoner response", response)

        return jsonify({'response': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8081)