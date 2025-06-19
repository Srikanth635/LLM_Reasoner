#!/bin/bash
set -e

# Activate the correct Conda environment first!
# This ensures all subsequent commands use the right tools and libraries.
echo "Activating Conda environment: agentic_ai311"
source /opt/conda/etc/profile.d/conda.sh
conda activate agentic_ai311

# Optional: Verify the Python/pip being used
echo "--- Using pip from: $(which pip)"
echo "--- Using python from: $(which python)"

# Trap to ensure Ollama is cleaned up if the script exits unexpectedly
cleanup() {
    echo "Shutting down Ollama..."
    pkill -f "ollama" || true
}
trap cleanup EXIT

# Run any startup tasks you need
echo "Importing Jupyter Lab workspaces..."
jupyter lab workspaces import ${PWD}/binder/jupyterlab.jupyterlab-workspace

# The final command should run in the foreground using exec.
# This makes 'ollama serve' the main process of the container.
# The container will stay alive as long as 'ollama serve' is running.
echo "Starting Ollama service in the foreground..."
exec ollama serve &

wait -n