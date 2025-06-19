#!/bin/bash

echo "Starting Ollama service..."

python3 -m flask_standalone_windows &

# Start Ollama in the background
ollama serve &

wait -n