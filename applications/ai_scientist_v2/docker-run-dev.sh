#!/bin/bash
# Development mode: Mount code as volume so changes are reflected without rebuilding

# Stop and remove existing container if it exists
docker stop safe-ai-scientist 2>/dev/null
docker rm safe-ai-scientist 2>/dev/null

# Run with volume mount for development
docker run -d \
  --name safe-ai-scientist \
  -p 8501:8501 \
  -v "$(pwd):/app" \
  -e OPENAI_API_KEY=${OPENAI_API_KEY} \
  -e OPENROUTER_API_KEY=${OPENROUTER_API_KEY} \
  safe-ai-scientist:test

echo "Container running. Code changes will be reflected automatically."
echo "View logs: docker logs -f safe-ai-scientist"
echo "Stop: docker stop safe-ai-scientist"

