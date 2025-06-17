from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Store the last received data globally
last_received_data = None


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/display', methods=['GET', 'POST'])
def display_results():
    global last_received_data

    if request.method == 'POST':
        # Get the JSON data that was sent in the body of the POST request
        received_data = request.get_json()

        if received_data:
            print(f"✅ [Server B] Received data: {received_data}")
            # Store the data for future GET requests
            last_received_data = received_data
            print("Received data type ",type(received_data))
            # Render the template using the data we just received
            return render_template('display.html', data=received_data)
        else:
            return "No data received in POST request.", 400

    # Handle GET requests (when you visit the URL in browser)
    if last_received_data:
        print(f"📄 [Server B] Displaying stored data via GET request")
        return render_template('display.html', data=last_received_data)
    else:
        return render_template('display.html', data={"message": "No data received yet. Waiting for POST request..."})


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8082)