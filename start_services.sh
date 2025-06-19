#!/bin/bash

echo "Starting Ollama service..."

# Start Ollama in the background
ollama serve &

#wait -n

jupyter lab workspaces import ${PWD}/binder/jupyterlab.jupyterlab-workspace

#python3 -m flask_standalone_windows &

exec "$@"