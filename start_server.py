#!/usr/bin/env python3
"""
Start the YouTube B-Roll Cutter server with N8N API integration
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Create output directory
Path("./web_output").mkdir(exist_ok=True)

print("🚀 Starting YouTube B-Roll Cutter Server with N8N API Integration...")
print("📱 Web Interface: http://localhost:8000")
print("🔧 API Documentation: http://localhost:8000/api/v1/docs")
print("📊 Health Check: http://localhost:8000/health")
print()
print("N8N API Endpoints:")
print("  POST /api/v1/extract-broll      - Extract B-roll (async)")
print("  POST /api/v1/batch-extract      - Batch processing (async)")
print("  POST /api/v1/process-sync       - Synchronous processing")
print("  GET  /api/v1/status/{job_id}    - Get job status")
print("  GET  /api/v1/download/{job_id}  - Download clips")
print()

if __name__ == "__main__":
    import uvicorn
    from backend.app import app
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )