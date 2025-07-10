#!/usr/bin/env python3
"""
Simple test to verify web processing works
"""

import requests
import time
import json

# Test URL
test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Start processing
print("🚀 Starting web processing test...")
response = requests.post("http://localhost:8000/process", json={
    "urls": [test_url],
    "settings": {"quality": "480p", "clip_duration": 5.0, "max_clips_per_video": 2}
})

if response.status_code == 200:
    job_data = response.json()
    job_id = job_data["job_id"]
    print(f"✅ Job started: {job_id}")
    
    # Monitor progress
    while True:
        status_response = requests.get(f"http://localhost:8000/status/{job_id}")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"Status: {status['status']} - Progress: {status['progress']:.1f}% - {status['message']}")
            
            if status['status'] in ['completed', 'failed']:
                if status['clips']:
                    print(f"🎬 Clips extracted: {status['clips']}")
                elif status.get('error'):
                    print(f"❌ Error: {status['error']}")
                break
        
        time.sleep(2)
        
else:
    print(f"❌ Failed to start processing: {response.status_code} {response.text}")