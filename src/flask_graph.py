from flask import Flask, request, jsonify
from langchain.parallel_workflow import *
from langchain.models_graph import *
from langchain.agents.reasoner_agent import *
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


        model_response = {
            "instruction": instruction,
            "action_core": action_core,
            "cram_plan_response": cram_plan_response,
            "enriched_action_core_attributes" : enriched_action_core_attributes
        }

        return jsonify(model_response), 200

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
        final_graph_state = models_graph.invoke({"instruction": _instruction,
                                                 "action_core": _action_core,
                                                 "enriched_action_core_attributes": _enriched_action_core_attributes,
                                                 "cram_plan_response": _cram_plan_response}, config=_config, stream_mode="updates")


        flanagan = models_graph.get_state(_config).values["flanagan"]
        framenet_model = models_graph.get_state(_config).values["framenet_model"]


        model_response = {
            "flanagan" : flanagan,
            "framenet_model" : framenet_model
        }

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

        print("HUGE CONTEXT : " ,CONTEXT)
        # Invoke LLM
        _config = {"configurable": {"thread_id": 3}}

        response = invoke_reasoner(CONTEXT, _query)

        print("Reasoner response", response)

        return jsonify({'response': response}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500




if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8081)