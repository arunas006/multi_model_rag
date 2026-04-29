#!/bin/sh

echo "Starting Ollama server..."
ollama serve &

# Wait for server to be ready
sleep 5

echo "Pulling model if not exists..."
ollama list | grep glm-ocr || ollama pull glm-ocr:latest

echo "Ollama ready"
wait