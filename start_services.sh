#!/bin/bash

echo "Starting Ollama service..."

jupyter lab workspaces import ${PWD}/binder/jupyterlab.jupyterlab-workspace

python3 -m flask_standalone_windows &

# Start Ollama in the background
ollama serve &

wait -n