#!/bin/bash

echo "🚀 Testing Railway Deployment"
echo "=============================="

echo "⏳ Waiting for deployment (60 seconds)..."
sleep 60

echo "🏥 Checking health..."
curl -s https://web-production-f517.up.railway.app/health | python3 -m json.tool

echo ""
echo "🧪 Testing extraction..."
curl -X POST "https://web-production-f517.up.railway.app/api/v1/process-sync" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "settings": {
      "clip_duration": 5.0,
      "quality": "720p",
      "max_clips_per_video": 2,
      "scene_threshold": 0.1,
      "start_time": 10,
      "max_duration": 30
    }
  }' \
  --max-time 120 | python3 -m json.tool

echo ""
echo "✅ Test complete!"