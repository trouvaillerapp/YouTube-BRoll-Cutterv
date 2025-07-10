#!/bin/bash
# Script to run the B-Roll Cutter server

echo "🎬 Starting YouTube B-Roll Cutter Server..."
echo "📱 Access the interface at: http://localhost:8001"
echo "🎙️ Speech detection features are available!"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""

# Change to the project directory
cd /Users/royantony/YouTube-BRoll-Cutter

# Run the server
python3 -m uvicorn backend.app:app --host 0.0.0.0 --port 8001 --reload