from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import json
from datetime import datetime
from src.resources.integration.fparser_python import *
from src.resources.integration.fparser import *
from src.flasking.sending_windows import *
from pathlib import Path

context = ""
with open("./feroz_context.txt", 'r') as f:
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
    global context
    summary = parse_summary_data(context)
    segments = parse_segment_data(context)
    filtered_segments = filter_redundant_actions(segments)
    final_unique_segments = filter_redundant_executed_actions(filtered_segments)
    chain = decomposer_prompt | ollama_llm.with_structured_output(AtomicsModel, method="json_schema")
    final_parse = decompose_segments_with_atomic_actions(chain, final_unique_segments)
    for seg in segments:
        print("📡 Sending test data to 1st window...")
        send_data_to_window(1, seg)

@app.route('/api/test/window1', methods=['GET'])
def test_window1():
    test_data = generate_test_data_for_window1()

    # Internally use Flask test client to call existing POST route
    # with app.test_client() as client:
    #     response = client.post('/api/window1', json=test_data)
    #     return response.get_data(), response.status_code, response.headers.items()
    return ""

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000,  allow_unsafe_werkzeug=True)