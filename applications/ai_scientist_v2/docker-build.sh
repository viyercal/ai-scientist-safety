#!/bin/bash
# Build Docker image without cache to ensure latest changes are included

docker build --platform linux/amd64 --no-cache -t safe-ai-scientist:test .

echo "Build complete. Run with: docker run -d --name safe-ai-scientist -p 8501:8501 safe-ai-scientist:test"

