from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
from datetime import datetime
from src.resources.integration.fparser_python import *
from src.resources.integration.fparser import *
from src.flasking.sending_windows import *
from src.langchain.agents.enhanced_ad_agent import *
from src.langchain.models_graph import *
from src.langchain.llm_configuration import *
from pathlib import Path

graph_output = []
summary = ""
segments = ""
context = ""
final_parse = ""

# with open("./feroz_context.txt", 'r') as f:
with open("./CRAM_Plan_Executive.txt", 'r') as f:
    context = f.read()


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store data for each window
window_data = {
    'window1': [],
    'window2': [],
    'window3': [],
    'window4': []
}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/window1', methods=['POST'])
def update_window1():
    data = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    window_data['window1'].append({
        'data': data,
        'timestamp': timestamp
    })

    # Keep only last 10 entries
    window_data['window1'] = window_data['window1'][-10:]

    # Emit to all connected clients
    socketio.emit('window1_update', {
        'data': data,
        'timestamp': timestamp
    })

    return jsonify({'status': 'success', 'window': 'window1'})


@app.route('/api/window2', methods=['POST'])
def update_window2():
    data = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    window_data['window2'].append({
        'data': data,
        'timestamp': timestamp
    })

    window_data['window2'] = window_data['window2'][-10:]

    socketio.emit('window2_update', {
        'data': data,
        'timestamp': timestamp
    })

    return jsonify({'status': 'success', 'window': 'window2'})


@app.route('/api/window3', methods=['POST'])
def update_window3():
    data = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    window_data['window3'].append({
        'data': data,
        'timestamp': timestamp
    })

    window_data['window3'] = window_data['window3'][-10:]

    socketio.emit('window3_update', {
        'data': data,
        'timestamp': timestamp
    })

    return jsonify({'status': 'success', 'window': 'window3'})


@app.route('/api/window4', methods=['POST'])
def update_window4():
    data = request.json
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    window_data['window4'].append({
        'data': data,
        'timestamp': timestamp
    })

    window_data['window4'] = window_data['window4'][-10:]

    socketio.emit('window4_update', {
        'data': data,
        'timestamp': timestamp
    })

    return jsonify({'status': 'success', 'window': 'window4'})


@app.route('/api/data/<window_id>')
def get_window_data(window_id):
    if window_id in window_data:
        return jsonify(window_data[window_id])
    return jsonify({'error': 'Window not found'}), 404


@socketio.on('connect')
def handle_connect():
    print('Client connected')
    # Send current data to newly connected client
    for window_id, data in window_data.items():
        if data:
            emit(f'{window_id}_update', data[-1])


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def generate_test_data_for_window1():
    print("Windows 1 Invoked")
    global context
    global summary
    summary = parse_summary_data(context)
    global segments
    segments = parse_segment_data(context)
    # filtered_segments = filter_redundant_actions(segments)
    # final_unique_segments = filter_redundant_executed_actions(filtered_segments)
    for seg in segments:
        print("📡 Sending test data to 1st window...")
        send_data_to_window(1, seg)

def generate_test_data_for_window2():
    print("Windows 2 Invoked")
    chain = decomposer_prompt | ollama_llm.with_structured_output(AtomicsModel, method="json_schema")
    global summary
    global final_parse
    print("Invoking Model")
    final_parse = chain.invoke({'segments': summary[0]})
    final_parse = final_parse.atomics
    # final_parse = decompose_segments_with_atomic_actions(chain, final_unique_segments)
    for par in final_parse:
        print("📡 Sending test data to 2nd window...")
        send_data_to_window(2, par)

def generate_test_data_for_window3():
    # final_parse_filtered = [final_parse[0], final_parse[1], final_parse[2]]
    for fp in final_parse:
        # instruction = fp['instruction']
        # ad = fp['action_designator']
        print("Invoking Model")
        out = ad_graph.invoke({'instruction': fp, 'context': ""})
        global graph_output
        graph_output.append(out)

    for out in graph_output:
        print("<UNK> Sending test data to 3rd window...")
        new_out = {}
        try:
            new_out['instruction'] = out['instruction']
            new_out['action_core'] = out['action_core']
            new_out['action_core_attributes'] = json.loads(out['action_core_attributes'])
            new_out['enriched_action_core_attributes'] = json.loads(out['enriched_action_core_attributes'])
            new_out['cram_plan'] = out['cram_plan_response']
            print("Parsed output normally")
        except:
            new_out['instruction'] = out['instruction']
            new_out['action_core'] = out['action_core']
            new_out['action_core_attributes'] = out['action_core_attributes']
            new_out['enriched_action_core_attributes'] = out['enriched_action_core_attributes']
            new_out['cram_plan'] = out['cram_plan_response']
            print("Parsed output with strings")

        send_data_to_window(3, new_out)

def generate_test_data_for_window4():
    config = {"configurable" : {"thread_id" : 1}}
    framenet_model = ""
    flanagan = ""
    for out in graph_output[:3]:
        print("<UNK> Sending test data to 4th window...")
        instruction = out['instruction']
        action_core = out['action_core']
        action_core_attributes = out['action_core_attributes']
        enriched_action_core_attributes = json.loads(out['enriched_action_core_attributes'])
        cram_plan = out['cram_plan_response']
        final_graph_state = model_graph.invoke({"instruction": instruction,
                                                "action_core": action_core,
                                                "enriched_action_core_attributes": enriched_action_core_attributes,
                                                "cram_plan_response": cram_plan}, config=config, stream_mode="updates")
        flanagan = model_graph.get_state(config).values["flanagan"]
        framenet_model = model_graph.get_state(config).values["framenet_model"]
        new_out = {}
        try:
            flanagan_json = json.loads(flanagan)
            framenet_model_json = json.loads(framenet_model)
            print("Parsed models output normally")
            new_out = {
                "framenet": framenet_model_json,
                "flanagan": flanagan_json
            }
        except:
            print("Parsed models output with strings")
            new_out = {
                "framenet": framenet_model,
                "flanagan": flanagan
            }

        send_data_to_window(4, new_out)

@app.route('/api/test/window1', methods=['GET'])
def test_window1():
    generate_test_data_for_window1()

    # Internally use Flask test client to call existing POST route
    # with app.test_client() as client:
    #     response = client.post('/api/window1', json=test_data)
    #     return response.get_data(), response.status_code, response.headers.items()
    return jsonify({"status": "test data sent"}), 200

@app.route('/api/test/window2', methods=['GET'])
def test_window2():
    generate_test_data_for_window2()
    return jsonify({"status": "test data sent"}), 200

@app.route('/api/test/window3', methods=['GET'])
def test_window3():
    generate_test_data_for_window3()
    return jsonify({"status": "test data sent"}), 200

@app.route('/api/test/window4', methods=['GET'])
def test_window4():
    generate_test_data_for_window4()
    return jsonify({"status": "test data sent"}), 200

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000,  allow_unsafe_werkzeug=True)