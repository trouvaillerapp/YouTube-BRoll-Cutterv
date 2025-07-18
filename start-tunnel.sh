#!/bin/bash

echo "🚀 Starting YouTube B-Roll Cutter with Cloudflare Tunnel..."

# Start Docker container
echo "Starting Docker container..."
docker-compose up -d

# Wait for service to be ready
echo "Waiting for service to start..."
sleep 5

# Check if service is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Service is running!"
else
    echo "❌ Service failed to start. Check logs with: docker-compose logs"
    exit 1
fi

# Start Cloudflare tunnel
echo "Starting Cloudflare tunnel..."
echo "Your public URL will appear below:"
echo "-----------------------------------"
cloudflared tunnel --url http://localhost:8000