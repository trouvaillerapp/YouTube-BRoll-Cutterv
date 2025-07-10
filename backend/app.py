#!/usr/bin/env python3
"""
FastAPI backend for YouTube B-Roll Cutter
"""

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import json
import logging
import os
import uuid
from pathlib import Path
import sys
import threading
import time

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from broll_cutter import YouTubeBRollCutter
    BROLL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: B-Roll Cutter not available: {e}")
    BROLL_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="YouTube B-Roll Cutter", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
active_connections: Dict[str, WebSocket] = {}
processing_jobs: Dict[str, Dict] = {}

# Pydantic models
class VideoRequest(BaseModel):
    urls: List[str]
    settings: Optional[Dict[str, Any]] = {}

class ProcessingSettings(BaseModel):
    output_dir: str = "./web_output"
    clip_duration: float = 8.0
    quality: str = "720p"
    scene_threshold: float = 0.3
    min_scene_length: float = 3.0
    max_scene_length: float = 15.0
    remove_watermarks: bool = True
    enhance_video: bool = False
    max_clips_per_video: int = 5
    extraction_mode: str = "scenes"  # scenes, speaking, quotes

class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float
    message: str
    clips: List[str] = []
    error: Optional[str] = None

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main HTML page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🎬 YouTube B-Roll Cutter</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f8f9fa; }
            .container { max-width: 1200px; }
            .card { box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .progress-container { height: 20px; }
            .url-input { min-height: 120px; }
            .results-container { max-height: 400px; overflow-y: auto; }
            .clip-card { margin-bottom: 10px; }
            .status-badge { font-size: 0.8em; }
            .settings-section { background-color: #f8f9fa; }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <div class="row">
                <div class="col-12">
                    <h1 class="text-center mb-4">🎬 YouTube B-Roll Cutter</h1>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>📹 Video URLs</h5>
                        </div>
                        <div class="card-body">
                            <textarea id="urlInput" class="form-control url-input" 
                                placeholder="Enter YouTube URLs here (one per line)&#10;&#10;💡 Tip: Use recent videos for best quality&#10;💡 Add &t=300 to start from 5 minutes (300 seconds)&#10;&#10;Examples:&#10;https://youtube.com/watch?v=ABC123&#10;https://youtube.com/watch?v=ABC123&t=120  (start at 2 min)"></textarea>
                            <div class="mt-3">
                                <button id="previewBtn" class="btn btn-outline-primary me-2">👁️ Preview Scenes</button>
                                <button id="processBtn" class="btn btn-primary">🎬 Extract B-Roll</button>
                                <button id="stopBtn" class="btn btn-danger ms-2" disabled>⏹️ Stop</button>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>⚙️ Settings</h5>
                        </div>
                        <div class="card-body settings-section">
                            <div class="row">
                                <div class="col-md-6">
                                    <label class="form-label">Quality</label>
                                    <select id="qualitySelect" class="form-select">
                                        <option value="480p">480p (Fast)</option>
                                        <option value="720p" selected>720p (Recommended)</option>
                                        <option value="1080p">1080p (High Quality)</option>
                                        <option value="4k">4K (Maximum)</option>
                                    </select>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Clip Duration (seconds)</label>
                                    <input type="number" id="durationInput" class="form-control" value="8" min="3" max="30">
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-md-6">
                                    <label class="form-label">Max Clips per Video</label>
                                    <input type="number" id="maxClipsInput" class="form-control" value="5" min="1" max="20">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Scene Sensitivity</label>
                                    <input type="range" id="sensitivityRange" class="form-range" min="0.1" max="1.0" step="0.1" value="0.3">
                                    <small class="text-muted">0.3</small>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-md-6">
                                    <label class="form-label">Start Time (optional)</label>
                                    <input type="text" id="startTimeInput" class="form-control" placeholder="e.g., 1:30 or 90 (seconds)" title="Format: MM:SS or seconds">
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Max Video Duration</label>
                                    <input type="number" id="maxDurationInput" class="form-control" value="300" min="60" max="3600" title="Maximum seconds to process from start time">
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-12">
                                    <label class="form-label">Extraction Mode</label>
                                    <select id="modeSelect" class="form-select">
                                        <option value="scenes" selected>Scene Detection (General B-Roll)</option>
                                        <option value="speaking">Speaking Segments (Dialogue/Monologue)</option>
                                        <option value="quotes">Quotes & Statements (3-15s clips)</option>
                                    </select>
                                </div>
                            </div>
                            <div class="row mt-3" id="speechSettings" style="display: none;">
                                <div class="col-md-6">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="speakerVisibleCheck">
                                        <label class="form-check-label" for="speakerVisibleCheck">
                                            Speaker must be visible (stricter)
                                        </label>
                                    </div>
                                </div>
                                <div class="col-md-6">
                                    <label class="form-label">Speech Duration Range</label>
                                    <div class="input-group">
                                        <input type="number" id="minSpeechDuration" class="form-control" value="3" min="1" max="30">
                                        <span class="input-group-text">to</span>
                                        <input type="number" id="maxSpeechDuration" class="form-control" value="20" min="5" max="60">
                                        <span class="input-group-text">sec</span>
                                    </div>
                                </div>
                            </div>
                            <div class="row mt-3">
                                <div class="col-12">
                                    <div class="alert alert-info">
                                        <small><strong>🎙️ Speech Detection:</strong> Extract clips where people are speaking or making statements<br>
                                        <strong>📹 Scene Detection:</strong> Extract visually interesting B-roll segments<br>
                                        <strong>💬 Quotes Mode:</strong> Find short impactful statements (3-15 seconds)</small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>📊 Progress</h5>
                        </div>
                        <div class="card-body">
                            <div class="progress progress-container mb-3">
                                <div id="progressBar" class="progress-bar" role="progressbar" style="width: 0%"></div>
                            </div>
                            <p id="statusText" class="mb-0">Ready to process videos</p>
                            <small id="jobId" class="text-muted"></small>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card">
                        <div class="card-header d-flex justify-content-between align-items-center">
                            <h5>📁 Results</h5>
                            <div>
                                <button id="clearResults" class="btn btn-sm btn-outline-secondary">Clear</button>
                                <button id="downloadAll" class="btn btn-sm btn-success" disabled>📥 Download All</button>
                            </div>
                        </div>
                        <div class="card-body">
                            <div id="resultsContainer" class="results-container">
                                <p class="text-muted">No results yet. Process some videos to see extracted clips here.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // WebSocket connection
            let ws = null;
            let currentJobId = null;
            let isProcessing = false;
            
            // DOM elements
            const urlInput = document.getElementById('urlInput');
            const previewBtn = document.getElementById('previewBtn');
            const processBtn = document.getElementById('processBtn');
            const stopBtn = document.getElementById('stopBtn');
            const progressBar = document.getElementById('progressBar');
            const statusText = document.getElementById('statusText');
            const jobIdText = document.getElementById('jobId');
            const resultsContainer = document.getElementById('resultsContainer');
            const clearResults = document.getElementById('clearResults');
            const downloadAll = document.getElementById('downloadAll');
            const sensitivityRange = document.getElementById('sensitivityRange');
            const modeSelect = document.getElementById('modeSelect');
            const speechSettings = document.getElementById('speechSettings');
            
            // Update sensitivity display
            sensitivityRange.addEventListener('input', (e) => {
                e.target.nextElementSibling.textContent = e.target.value;
            });
            
            // Show/hide speech settings based on mode
            modeSelect.addEventListener('change', (e) => {
                if (e.target.value === 'speaking' || e.target.value === 'quotes') {
                    speechSettings.style.display = 'block';
                } else {
                    speechSettings.style.display = 'none';
                }
            });
            
            // Initialize WebSocket
            function initWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${protocol}//${window.location.host}/ws`);
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                };
                
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    handleWebSocketMessage(data);
                };
                
                ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    setTimeout(initWebSocket, 3000); // Reconnect after 3 seconds
                };
                
                ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };
            }
            
            // Handle WebSocket messages
            function handleWebSocketMessage(data) {
                if (data.type === 'progress') {
                    updateProgress(data.progress, data.message);
                } else if (data.type === 'completed') {
                    handleJobCompleted(data);
                } else if (data.type === 'error') {
                    handleJobError(data);
                }
            }
            
            // Poll job status when processing
            function pollJobStatus() {
                if (!currentJobId || !isProcessing) return;
                
                fetch(`/status/${currentJobId}`)
                    .then(response => response.json())
                    .then(status => {
                        updateProgress(status.progress, status.message);
                        
                        if (status.status === 'completed') {
                            handleJobCompleted({clips: status.clips});
                        } else if (status.status === 'failed') {
                            handleJobError({error: status.error || 'Processing failed'});
                        } else if (status.status === 'processing') {
                            // Continue polling
                            setTimeout(pollJobStatus, 2000);
                        }
                    })
                    .catch(error => {
                        console.error('Status poll error:', error);
                        setTimeout(pollJobStatus, 5000); // Retry after 5 seconds
                    });
            }
            
            // Update progress display
            function updateProgress(progress, message) {
                progressBar.style.width = `${progress}%`;
                progressBar.textContent = `${Math.round(progress)}%`;
                statusText.textContent = message;
            }
            
            // Handle job completion
            function handleJobCompleted(data) {
                isProcessing = false;
                processBtn.disabled = false;
                stopBtn.disabled = true;
                
                updateProgress(100, 'Processing completed!');
                
                if (data.clips && data.clips.length > 0) {
                    displayResults(data.clips);
                    downloadAll.disabled = false;
                } else {
                    resultsContainer.innerHTML = '<p class="text-warning">No clips were extracted.</p>';
                }
            }
            
            // Handle job error
            function handleJobError(data) {
                isProcessing = false;
                processBtn.disabled = false;
                stopBtn.disabled = true;
                
                progressBar.classList.add('bg-danger');
                statusText.textContent = `Error: ${data.error}`;
                
                resultsContainer.innerHTML = `<p class="text-danger">Processing failed: ${data.error}</p>`;
            }
            
            // Display results
            function displayResults(clips) {
                let html = '<div class="row">';
                clips.forEach((clip, index) => {
                    const clipName = clip.split('/').pop();
                    html += `
                        <div class="col-md-4 mb-3">
                            <div class="card clip-card">
                                <div class="card-body">
                                    <h6 class="card-title">${clipName}</h6>
                                    <div class="d-flex justify-content-between">
                                        <a href="/download/${encodeURIComponent(clipName)}" 
                                           class="btn btn-sm btn-primary">📥 Download</a>
                                        <button class="btn btn-sm btn-outline-info" 
                                                onclick="previewClip('${clip}')">👁️ Preview</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                });
                html += '</div>';
                resultsContainer.innerHTML = html;
            }
            
            // Parse start time from MM:SS or seconds format
            function parseStartTime(timeStr) {
                if (!timeStr) return 0;
                
                timeStr = timeStr.trim();
                if (timeStr.includes(':')) {
                    const parts = timeStr.split(':');
                    const minutes = parseInt(parts[0]) || 0;
                    const seconds = parseInt(parts[1]) || 0;
                    return minutes * 60 + seconds;
                } else {
                    return parseInt(timeStr) || 0;
                }
            }
            
            // Get current settings
            function getSettings() {
                const mode = document.getElementById('modeSelect').value;
                const settings = {
                    quality: document.getElementById('qualitySelect').value,
                    clip_duration: parseFloat(document.getElementById('durationInput').value),
                    max_clips_per_video: parseInt(document.getElementById('maxClipsInput').value),
                    scene_threshold: parseFloat(document.getElementById('sensitivityRange').value),
                    start_time: parseStartTime(document.getElementById('startTimeInput').value),
                    max_duration: parseInt(document.getElementById('maxDurationInput').value),
                    remove_watermarks: false,  // MVP: Disabled
                    enhance_video: false,      // MVP: Disabled
                    extraction_mode: mode
                };
                
                // Add speech-specific settings
                if (mode === 'speaking' || mode === 'quotes') {
                    settings.speaker_visible = document.getElementById('speakerVisibleCheck').checked;
                    settings.min_speech_duration = parseFloat(document.getElementById('minSpeechDuration').value);
                    settings.max_speech_duration = parseFloat(document.getElementById('maxSpeechDuration').value);
                    
                    if (mode === 'quotes') {
                        settings.min_quote_duration = settings.min_speech_duration;
                        settings.max_quote_duration = Math.min(settings.max_speech_duration, 15);
                    }
                }
                
                return settings;
            }
            
            // Process videos
            async function processVideos() {
                const urls = urlInput.value.trim().split('\\n').filter(url => url.trim());
                
                if (urls.length === 0) {
                    alert('Please enter at least one YouTube URL');
                    return;
                }
                
                isProcessing = true;
                processBtn.disabled = true;
                stopBtn.disabled = false;
                
                const requestData = {
                    urls: urls,
                    settings: getSettings()
                };
                
                try {
                    console.log('Sending request:', requestData);
                    
                    const response = await fetch('/process', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                            'Accept': 'application/json'
                        },
                        body: JSON.stringify(requestData)
                    });
                    
                    console.log('Response status:', response.status);
                    
                    if (!response.ok) {
                        const errorText = await response.text();
                        console.error('Error response:', errorText);
                        throw new Error(`HTTP ${response.status}: ${errorText}`);
                    }
                    
                    const data = await response.json();
                    console.log('Response data:', data);
                    
                    currentJobId = data.job_id;
                    jobIdText.textContent = `Job ID: ${currentJobId}`;
                    updateProgress(0, 'Processing started...');
                    
                    // Start polling for status updates
                    setTimeout(pollJobStatus, 2000);
                    
                } catch (error) {
                    console.error('Fetch error:', error);
                    handleJobError({error: `Failed to start processing: ${error.message}`});
                }
            }
            
            // Preview scenes
            async function previewScenes() {
                const urls = urlInput.value.trim().split('\\n').filter(url => url.trim());
                
                if (urls.length === 0) {
                    alert('Please enter at least one YouTube URL');
                    return;
                }
                
                try {
                    const response = await fetch('/preview', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({url: urls[0]})
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayPreviewResults(data.scenes);
                    } else {
                        throw new Error(data.detail || 'Preview failed');
                    }
                } catch (error) {
                    alert(`Preview failed: ${error.message}`);
                }
            }
            
            // Display preview results
            function displayPreviewResults(scenes) {
                let html = '<h6>Preview Results:</h6>';
                scenes.forEach((scene, index) => {
                    const duration = scene.end_time - scene.start_time;
                    html += `
                        <div class="alert alert-info">
                            <strong>Scene ${index + 1}:</strong> ${duration.toFixed(1)}s - 
                            ${scene.scene_type} (confidence: ${scene.confidence.toFixed(2)})
                        </div>
                    `;
                });
                resultsContainer.innerHTML = html;
            }
            
            // Stop processing
            async function stopProcessing() {
                if (currentJobId) {
                    try {
                        await fetch(`/stop/${currentJobId}`, {method: 'POST'});
                        isProcessing = false;
                        processBtn.disabled = false;
                        stopBtn.disabled = true;
                        updateProgress(0, 'Processing stopped');
                    } catch (error) {
                        console.error('Failed to stop processing:', error);
                    }
                }
            }
            
            // Clear results
            function clearResultsDisplay() {
                resultsContainer.innerHTML = '<p class="text-muted">No results yet. Process some videos to see extracted clips here.</p>';
                downloadAll.disabled = true;
                updateProgress(0, 'Ready to process videos');
                jobIdText.textContent = '';
            }
            
            // Event listeners
            processBtn.addEventListener('click', processVideos);
            previewBtn.addEventListener('click', previewScenes);
            stopBtn.addEventListener('click', stopProcessing);
            clearResults.addEventListener('click', clearResultsDisplay);
            
            // Initialize
            initWebSocket();
        </script>
    </body>
    </html>
    """

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    active_connections[connection_id] = websocket
    
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        del active_connections[connection_id]

async def broadcast_progress(job_id: str, progress: float, message: str, stage: str = "main"):
    """Broadcast progress to all connected clients"""
    data = {
        "type": "progress",
        "job_id": job_id,
        "progress": progress,
        "message": message,
        "stage": stage
    }
    
    for connection in active_connections.values():
        try:
            await connection.send_text(json.dumps(data))
        except:
            pass

async def broadcast_completion(job_id: str, clips: List[str]):
    """Broadcast completion to all connected clients"""
    data = {
        "type": "completed",
        "job_id": job_id,
        "clips": clips
    }
    
    for connection in active_connections.values():
        try:
            await connection.send_text(json.dumps(data))
        except:
            pass

async def broadcast_error(job_id: str, error: str):
    """Broadcast error to all connected clients"""
    data = {
        "type": "error",
        "job_id": job_id,
        "error": error
    }
    
    for connection in active_connections.values():
        try:
            await connection.send_text(json.dumps(data))
        except:
            pass

@app.post("/process")
async def process_videos(request: VideoRequest, background_tasks: BackgroundTasks):
    """Start video processing job"""
    try:
        if not BROLL_AVAILABLE:
            raise HTTPException(status_code=500, detail="B-Roll Cutter module not available")
        
        if not request.urls:
            raise HTTPException(status_code=400, detail="No URLs provided")
        
        job_id = str(uuid.uuid4())
        logger.info(f"Starting new job {job_id} with {len(request.urls)} URLs")
        
        # Store job info
        processing_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Job queued",
            "clips": [],
            "urls": request.urls,
            "settings": request.settings
        }
        
        # Start processing in background
        background_tasks.add_task(process_videos_background, job_id, request.urls, request.settings)
        
        return {"job_id": job_id, "status": "started"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

def process_videos_sync(job_id: str, urls: List[str], settings: Dict[str, Any]):
    """Synchronous video processing function"""
    try:
        if not BROLL_AVAILABLE:
            raise Exception("B-Roll Cutter module not available")
        
        # Update job status
        processing_jobs[job_id]["status"] = "processing"
        
        # Create progress callback
        def sync_progress_callback(info):
            progress = info.get('progress', 0)
            message = info.get('message', '')
            processing_jobs[job_id]["progress"] = progress
            processing_jobs[job_id]["message"] = message
            print(f"Job {job_id}: {progress:.1f}% - {message}")  # Console logging
        
        # Initialize cutter with settings
        output_dir = Path("web_output")
        output_dir.mkdir(exist_ok=True)
        
        cutter_settings = {
            "output_dir": str(output_dir),
            "clip_duration": settings.get("clip_duration", 8.0),
            "quality": settings.get("quality", "720p"),  # Higher quality by default
            "scene_threshold": settings.get("scene_threshold", 0.3),
            "remove_watermarks": False,  # Disabled for MVP
            "enhance_video": False,      # Disabled for MVP
            "max_clips_per_video": settings.get("max_clips_per_video", 5),
            "progress_callback": sync_progress_callback
        }
        
        cutter = YouTubeBRollCutter(**cutter_settings)
        
        # Process each URL
        all_clips = []
        for i, url in enumerate(urls):
            try:
                sync_progress_callback({
                    'progress': (i / len(urls)) * 90,
                    'message': f"Processing video {i+1}/{len(urls)}: {url}"
                })
                
                # Check extraction mode
                extraction_mode = settings.get("extraction_mode", "scenes")
                
                if extraction_mode == "speaking":
                    # Extract speaking clips
                    clips = cutter.extract_speaking_clips(
                        url,
                        speaker_visible=settings.get("speaker_visible", True),
                        quote_mode=False,
                        min_duration=settings.get("min_speech_duration", 3.0),
                        max_duration=settings.get("max_speech_duration", 20.0),
                        custom_settings=settings
                    )
                elif extraction_mode == "quotes":
                    # Extract quote clips
                    clips = cutter.extract_speaking_clips(
                        url,
                        speaker_visible=True,
                        quote_mode=True,
                        min_duration=settings.get("min_quote_duration", 3.0),
                        max_duration=settings.get("max_quote_duration", 15.0),
                        custom_settings=settings
                    )
                else:
                    # Default scene extraction
                    start_time = settings.get("start_time", 0)
                    max_duration = settings.get("max_duration", None)
                    clips = cutter.extract_broll(url, start_time=start_time, max_duration=max_duration)
                
                all_clips.extend(clips)
                
            except Exception as e:
                logger.error(f"Error processing URL {url}: {str(e)}")
                continue
        
        # Update job completion
        processing_jobs[job_id]["status"] = "completed"
        # Just use the filename for download links
        processing_jobs[job_id]["clips"] = [Path(clip).name for clip in all_clips]
        processing_jobs[job_id]["progress"] = 100
        processing_jobs[job_id]["message"] = f"Completed! Extracted {len(all_clips)} clips"
        
        print(f"Job {job_id}: Completed with {len(all_clips)} clips")
        
        # Send webhook notification if configured
        webhook_url = processing_jobs[job_id].get("webhook_url")
        if webhook_url:
            try:
                import requests
                webhook_data = {
                    "job_id": job_id,
                    "status": "completed",
                    "clips_count": len(all_clips),
                    "clips": [Path(clip).name for clip in all_clips],
                    "download_links": [f"/api/v1/download/{job_id}/{Path(clip).name}" for clip in all_clips],
                    "bulk_download_url": f"/api/v1/download/{job_id}",
                    "message": f"Successfully extracted {len(all_clips)} B-roll clips"
                }
                requests.post(webhook_url, json=webhook_data, timeout=10)
                print(f"Webhook sent to {webhook_url}")
            except Exception as e:
                print(f"Failed to send webhook: {e}")
                logger.error(f"Webhook failed for job {job_id}: {str(e)}")
        
    except Exception as e:
        logger.error(f"Background processing failed for job {job_id}: {str(e)}")
        processing_jobs[job_id]["status"] = "failed"
        processing_jobs[job_id]["error"] = str(e)
        
        print(f"Job {job_id}: Failed with error: {str(e)}")
        
        # Send error webhook notification if configured
        webhook_url = processing_jobs[job_id].get("webhook_url")
        if webhook_url:
            try:
                import requests
                webhook_data = {
                    "job_id": job_id,
                    "status": "failed",
                    "error": str(e),
                    "message": f"Processing failed: {str(e)}"
                }
                requests.post(webhook_url, json=webhook_data, timeout=10)
                print(f"Error webhook sent to {webhook_url}")
            except Exception as webhook_error:
                print(f"Failed to send error webhook: {webhook_error}")
                logger.error(f"Error webhook failed for job {job_id}: {str(webhook_error)}")

async def process_videos_background(job_id: str, urls: List[str], settings: Dict[str, Any]):
    """Background task wrapper for processing videos"""
    # Run the synchronous processing in a thread
    thread = threading.Thread(
        target=process_videos_sync,
        args=(job_id, urls, settings)
    )
    thread.start()

@app.post("/preview")
async def preview_scenes(request: dict):
    """Preview scenes for a single video"""
    try:
        if not BROLL_AVAILABLE:
            raise HTTPException(status_code=500, detail="B-Roll Cutter module not available")
        
        url = request.get("url")
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Initialize cutter
        cutter = YouTubeBRollCutter()
        
        # Preview scenes
        scenes = cutter.preview_scenes(url)
        
        return {"scenes": scenes}
        
    except Exception as e:
        logger.error(f"Preview failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        clips=job["clips"],
        error=job.get("error")
    )

@app.post("/stop/{job_id}")
async def stop_job(job_id: str):
    """Stop a processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    processing_jobs[job_id]["status"] = "stopped"
    return {"message": "Job stopped"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download a processed clip"""
    file_path = Path("web_output") / filename
    
    if not file_path.exists():
        # Also check current directory
        file_path = Path(".") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='video/mp4'
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "active_jobs": len(processing_jobs)}

# N8N Integration APIs
class N8NVideoRequest(BaseModel):
    url: str
    settings: Optional[Dict[str, Any]] = {}
    webhook_url: Optional[str] = None  # For async notifications

class N8NBatchRequest(BaseModel):
    urls: List[str]
    settings: Optional[Dict[str, Any]] = {}
    webhook_url: Optional[str] = None

@app.post("/api/v1/extract-broll")
async def n8n_extract_broll(request: N8NVideoRequest, background_tasks: BackgroundTasks):
    """
    N8N API endpoint for extracting B-roll from a single video
    Returns job_id for async processing or can wait for completion
    """
    try:
        if not BROLL_AVAILABLE:
            raise HTTPException(status_code=500, detail="B-Roll Cutter module not available")
        
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        job_id = str(uuid.uuid4())
        logger.info(f"N8N job {job_id} started for URL: {request.url}")
        
        # Store job info with webhook
        processing_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Job queued",
            "clips": [],
            "urls": [request.url],
            "settings": request.settings,
            "webhook_url": request.webhook_url,
            "created_at": time.time()
        }
        
        # Start processing in background
        background_tasks.add_task(process_videos_background, job_id, [request.url], request.settings)
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Processing started",
            "estimated_duration": "2-5 minutes",
            "status_url": f"/api/v1/status/{job_id}",
            "download_url": f"/api/v1/download/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"N8N extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@app.post("/api/v1/batch-extract")
async def n8n_batch_extract(request: N8NBatchRequest, background_tasks: BackgroundTasks):
    """
    N8N API endpoint for batch processing multiple videos
    """
    try:
        if not BROLL_AVAILABLE:
            raise HTTPException(status_code=500, detail="B-Roll Cutter module not available")
        
        if not request.urls:
            raise HTTPException(status_code=400, detail="URLs are required")
        
        job_id = str(uuid.uuid4())
        logger.info(f"N8N batch job {job_id} started with {len(request.urls)} URLs")
        
        # Store job info
        processing_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Batch job queued",
            "clips": [],
            "urls": request.urls,
            "settings": request.settings,
            "webhook_url": request.webhook_url,
            "created_at": time.time()
        }
        
        # Start processing in background
        background_tasks.add_task(process_videos_background, job_id, request.urls, request.settings)
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Batch processing started for {len(request.urls)} videos",
            "estimated_duration": f"{len(request.urls) * 3}-{len(request.urls) * 7} minutes",
            "status_url": f"/api/v1/status/{job_id}",
            "download_url": f"/api/v1/download/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"N8N batch processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start batch processing: {str(e)}")

@app.get("/api/v1/status/{job_id}")
async def n8n_get_status(job_id: str):
    """
    N8N API endpoint to get job status with additional metadata
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    # Calculate elapsed time
    elapsed_time = int(time.time() - job.get("created_at", time.time()))
    
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "clips_count": len(job["clips"]),
        "clips": job["clips"],
        "elapsed_time_seconds": elapsed_time,
        "error": job.get("error"),
        "urls_processed": len(job["urls"]),
        "settings": job.get("settings", {})
    }
    
    # Add download links if completed
    if job["status"] == "completed" and job["clips"]:
        response["download_links"] = [
            f"/api/v1/download/{job_id}/{clip}" for clip in job["clips"]
        ]
        response["bulk_download_url"] = f"/api/v1/download/{job_id}"
    
    return response

@app.get("/api/v1/download/{job_id}")
async def n8n_download_all(job_id: str):
    """
    N8N API endpoint to download all clips as a ZIP file
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    if not job["clips"]:
        raise HTTPException(status_code=404, detail="No clips available")
    
    # Create ZIP file
    import zipfile
    import tempfile
    
    zip_path = Path(tempfile.gettempdir()) / f"broll_clips_{job_id}.zip"
    
    with zipfile.ZipFile(zip_path, 'w') as zip_file:
        for clip_name in job["clips"]:
            clip_path = Path("web_output") / clip_name
            if clip_path.exists():
                zip_file.write(clip_path, clip_name)
    
    if not zip_path.exists():
        raise HTTPException(status_code=500, detail="Failed to create ZIP file")
    
    return FileResponse(
        path=str(zip_path),
        filename=f"broll_clips_{job_id}.zip",
        media_type='application/zip'
    )

@app.get("/api/v1/download/{job_id}/{filename}")
async def n8n_download_single(job_id: str, filename: str):
    """
    N8N API endpoint to download a single clip
    """
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if filename not in job["clips"]:
        raise HTTPException(status_code=404, detail="Clip not found in job")
    
    file_path = Path("web_output") / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='video/mp4'
    )

@app.post("/api/v1/process-sync")
async def n8n_process_sync(request: N8NVideoRequest):
    """
    N8N API endpoint for synchronous processing (waits for completion)
    Use this for smaller jobs where you want to wait for results
    """
    try:
        if not BROLL_AVAILABLE:
            raise HTTPException(status_code=500, detail="B-Roll Cutter module not available")
        
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        # Initialize cutter
        output_dir = Path("web_output")
        output_dir.mkdir(exist_ok=True)
        
        settings = request.settings or {}
        cutter_settings = {
            "output_dir": str(output_dir),
            "clip_duration": settings.get("clip_duration", 8.0),
            "quality": settings.get("quality", "720p"),
            "scene_threshold": settings.get("scene_threshold", 0.3),
            "remove_watermarks": False,
            "enhance_video": False,
            "max_clips_per_video": settings.get("max_clips_per_video", 5)
        }
        
        cutter = YouTubeBRollCutter(**cutter_settings)
        
        # Process synchronously
        start_time = settings.get("start_time", 0)
        max_duration = settings.get("max_duration", None)
        
        clips = cutter.extract_broll(request.url, start_time=start_time, max_duration=max_duration)
        
        # Return results immediately
        clip_names = [Path(clip).name for clip in clips]
        
        return {
            "status": "completed",
            "message": f"Successfully extracted {len(clips)} clips",
            "clips_count": len(clips),
            "clips": clip_names,
            "download_links": [f"/download/{clip}" for clip in clip_names],
            "processing_time": "sync"
        }
        
    except Exception as e:
        logger.error(f"N8N sync processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/api/v1/extract-speaking")
async def n8n_extract_speaking(request: N8NVideoRequest, background_tasks: BackgroundTasks):
    """
    N8N API endpoint for extracting speaking/quote segments
    """
    try:
        if not BROLL_AVAILABLE:
            raise HTTPException(status_code=500, detail="B-Roll Cutter module not available")
        
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        job_id = str(uuid.uuid4())
        logger.info(f"N8N speaking extraction job {job_id} started for URL: {request.url}")
        
        # Configure settings for speech extraction
        settings = request.settings or {}
        settings["extraction_mode"] = "speaking"
        settings["speaker_visible"] = settings.get("speaker_visible", True)
        settings["min_speech_duration"] = settings.get("min_speech_duration", 3.0)
        settings["max_speech_duration"] = settings.get("max_speech_duration", 20.0)
        
        # Store job info
        processing_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Speech extraction queued",
            "clips": [],
            "urls": [request.url],
            "settings": settings,
            "webhook_url": request.webhook_url,
            "created_at": time.time()
        }
        
        # Start processing in background
        background_tasks.add_task(process_videos_background, job_id, [request.url], settings)
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Speech extraction started",
            "extraction_mode": "speaking",
            "estimated_duration": "3-6 minutes",
            "status_url": f"/api/v1/status/{job_id}",
            "download_url": f"/api/v1/download/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"N8N speech extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start speech extraction: {str(e)}")

@app.post("/api/v1/extract-quotes")
async def n8n_extract_quotes(request: N8NVideoRequest, background_tasks: BackgroundTasks):
    """
    N8N API endpoint for extracting quote segments (3-15 seconds)
    """
    try:
        if not BROLL_AVAILABLE:
            raise HTTPException(status_code=500, detail="B-Roll Cutter module not available")
        
        if not request.url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        job_id = str(uuid.uuid4())
        logger.info(f"N8N quote extraction job {job_id} started for URL: {request.url}")
        
        # Configure settings for quote extraction
        settings = request.settings or {}
        settings["extraction_mode"] = "quotes"
        settings["speaker_visible"] = True
        settings["min_speech_duration"] = settings.get("min_quote_duration", 3.0)
        settings["max_speech_duration"] = settings.get("max_quote_duration", 15.0)
        
        # Store job info
        processing_jobs[job_id] = {
            "status": "pending",
            "progress": 0,
            "message": "Quote extraction queued",
            "clips": [],
            "urls": [request.url],
            "settings": settings,
            "webhook_url": request.webhook_url,
            "created_at": time.time()
        }
        
        # Start processing in background
        background_tasks.add_task(process_videos_background, job_id, [request.url], settings)
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": "Quote extraction started",
            "extraction_mode": "quotes",
            "estimated_duration": "3-6 minutes",
            "status_url": f"/api/v1/status/{job_id}",
            "download_url": f"/api/v1/download/{job_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"N8N quote extraction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start quote extraction: {str(e)}")

@app.get("/api/v1/docs")
async def n8n_api_docs():
    """
    N8N Integration documentation
    """
    return {
        "title": "YouTube B-Roll Cutter API for N8N",
        "version": "1.0.0",
        "description": "API endpoints for integrating YouTube B-Roll Cutter with n8n workflows",
        "endpoints": {
            "POST /api/v1/extract-broll": {
                "description": "Extract B-roll from a single video (async)",
                "body": {
                    "url": "https://youtube.com/watch?v=VIDEO_ID",
                    "settings": {
                        "clip_duration": 8.0,
                        "quality": "720p",
                        "max_clips_per_video": 5,
                        "start_time": 0,
                        "max_duration": 300
                    },
                    "webhook_url": "https://your-n8n-webhook-url"
                },
                "response": {
                    "job_id": "uuid",
                    "status": "started",
                    "status_url": "/api/v1/status/{job_id}"
                }
            },
            "POST /api/v1/batch-extract": {
                "description": "Extract B-roll from multiple videos (async)",
                "body": {
                    "urls": ["https://youtube.com/watch?v=VIDEO_ID1", "https://youtube.com/watch?v=VIDEO_ID2"],
                    "settings": "same as above",
                    "webhook_url": "optional"
                }
            },
            "POST /api/v1/process-sync": {
                "description": "Extract B-roll synchronously (waits for completion)",
                "body": "same as extract-broll",
                "response": {
                    "status": "completed",
                    "clips": ["clip1.mp4", "clip2.mp4"],
                    "download_links": ["/download/clip1.mp4"]
                }
            },
            "POST /api/v1/extract-speaking": {
                "description": "Extract clips where people are speaking",
                "body": {
                    "url": "https://youtube.com/watch?v=VIDEO_ID",
                    "settings": {
                        "speaker_visible": True,
                        "min_speech_duration": 3.0,
                        "max_speech_duration": 20.0,
                        "max_clips_per_video": 5
                    },
                    "webhook_url": "optional"
                },
                "response": "same as extract-broll"
            },
            "POST /api/v1/extract-quotes": {
                "description": "Extract short quote/statement clips (3-15 seconds)",
                "body": {
                    "url": "https://youtube.com/watch?v=VIDEO_ID",
                    "settings": {
                        "min_quote_duration": 3.0,
                        "max_quote_duration": 15.0,
                        "max_clips_per_video": 10
                    },
                    "webhook_url": "optional"
                },
                "response": "same as extract-broll"
            },
            "GET /api/v1/status/{job_id}": {
                "description": "Get job status and results",
                "response": {
                    "job_id": "uuid",
                    "status": "completed|processing|failed",
                    "progress": 100,
                    "clips": ["clip1.mp4"],
                    "download_links": ["/api/v1/download/job_id/clip1.mp4"]
                }
            },
            "GET /api/v1/download/{job_id}": {
                "description": "Download all clips as ZIP file"
            },
            "GET /api/v1/download/{job_id}/{filename}": {
                "description": "Download single clip"
            }
        },
        "settings": {
            "clip_duration": "Duration of each clip in seconds (3-30)",
            "quality": "Video quality: 480p, 720p, 1080p, 4k",
            "max_clips_per_video": "Maximum clips to extract (1-20)",
            "start_time": "Start time in seconds (0 = beginning)",
            "max_duration": "Maximum duration to process in seconds",
            "scene_threshold": "Scene detection sensitivity (0.1-1.0)"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    # Create output directory
    Path("./web_output").mkdir(exist_ok=True)
    
    print("🚀 Starting YouTube B-Roll Cutter Web Server...")
    print("📱 Access the web interface at: http://localhost:8000")
    print("🔧 API documentation at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )