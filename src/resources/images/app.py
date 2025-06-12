import subprocess
import os
from flask import Flask, render_template, request, jsonify

# Initialize the Flask application
app = Flask(__name__)

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main HTML page (the frontend)."""
    return render_template('index.html')

@app.route('/execute', methods=['POST'])
def execute_command():
    """
    API endpoint to execute commands using subprocess.
    *** EXTREMELY DANGEROUS - FOR LOCAL USE ONLY ***
    """
    # Get the JSON data from the request
    data = request.get_json()
    if not data or 'command' not in data:
        return jsonify({'output': 'Error: Invalid request.'}), 400

    command = data['command'].strip()
    if not command:
        return jsonify({'output': ''})

    # --- SECURITY WARNING ---
    # The 'shell=True' argument is powerful and dangerous. It passes the command
    # to your system's shell for interpretation. This is what allows it to find
    # executables in your PATH and understand shell syntax.
    # It also means it can execute ANY command, including malicious ones like 'rm -rf /'.
    try:
        # For 'cd', we need to handle it specially since subprocess
        # runs each command in a new shell.
        if command.startswith('cd '):
            try:
                # Get the target directory
                target_dir = command.split(None, 1)[1]
                # Change the directory for the Flask process
                os.chdir(target_dir)
                # Return the new current directory as output
                output = f"Changed directory to: {os.getcwd()}"
            except FileNotFoundError:
                output = f"Error: Directory not found: {target_dir}"
            except IndexError:
                # Handle 'cd' with no arguments, go to home directory
                home_dir = os.path.expanduser('~')
                os.chdir(home_dir)
                output = f"Changed directory to: {os.getcwd()}"
        else:
            # Execute all other commands using subprocess
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                # The 'cwd' argument ensures the command runs in the
                # current working directory of the Flask app, which 'cd' can change.
                cwd=os.getcwd()
            )
            # Combine stdout and stderr for the output
            if result.stdout:
                output = result.stdout
            else:
                output = result.stderr

        return jsonify({'output': output})

    except Exception as e:
        # Catch any other unexpected errors during execution
        return jsonify({'output': f"Execution error: {str(e)}"}), 500

# --- Main Execution ---

if __name__ == '__main__':
    # Binds to localhost only. Do not change host to '0.0.0.0'
    # as that would expose this dangerous app to your network.
    app.run(debug=True, host='127.0.0.1', port=8082)