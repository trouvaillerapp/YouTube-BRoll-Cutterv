# n8n Integration Examples for YouTube B-Roll Cutter

## 🚀 Quick Start with n8n

### 1. Simple Speech Extraction Workflow

**Nodes:**
1. **Trigger** (Manual/Webhook)
2. **HTTP Request** (Extract Speaking Clips)
3. **Wait** (5 seconds)
4. **HTTP Request** (Check Status) 
5. **IF** (Check if completed)
6. **HTTP Request** (Download ZIP)

---

## 📝 Detailed Node Configurations

### Node 1: HTTP Request - Start Speech Extraction

**Method:** POST  
**URL:** `http://localhost:8001/api/v1/extract-speaking`  
**Headers:**
```json
{
  "Content-Type": "application/json"
}
```

**Body:**
```json
{
  "url": "{{ $json.youtube_url }}",
  "settings": {
    "clip_duration": 8,
    "quality": "720p", 
    "max_clips_per_video": 5,
    "speaker_visible": false,
    "min_speech_duration": 3.0,
    "max_speech_duration": 20.0
  },
  "webhook_url": "{{ $json.webhook_url }}"
}
```

**Output:** `job_id`, `status_url`

---

### Node 2: Loop Until Complete

**HTTP Request - Check Status:**

**Method:** GET  
**URL:** `http://localhost:8001/api/v1/status/{{ $json.job_id }}`

**Wait Node:** 10 seconds between checks

**IF Node:** `{{ $json.status === "completed" }}`

---

### Node 3: Download Results

**HTTP Request - Download ZIP:**

**Method:** GET  
**URL:** `http://localhost:8001/api/v1/download/{{ $json.job_id }}`  
**Response Format:** Binary

---

## 🎯 Advanced Workflows

### Batch Processing Multiple Videos

```json
{
  "urls": [
    "https://youtube.com/watch?v=VIDEO1",
    "https://youtube.com/watch?v=VIDEO2", 
    "https://youtube.com/watch?v=VIDEO3"
  ],
  "settings": {
    "clip_duration": 8,
    "quality": "720p",
    "extraction_mode": "speaking"
  }
}
```

### Quote Extraction for Social Media

```json
{
  "url": "{{ $json.podcast_url }}",
  "settings": {
    "clip_duration": 10,
    "quality": "1080p",
    "min_quote_duration": 5.0,
    "max_quote_duration": 15.0,
    "max_clips_per_video": 10
  }
}
```

---

## 🔄 Webhook Integration

### Async Processing with Webhooks

1. **Your n8n workflow starts the job**
2. **B-Roll Cutter processes in background**
3. **When complete, calls your webhook**
4. **n8n receives completion notification**

**Webhook URL:** `https://your-n8n-instance.com/webhook/broll-complete`

**Webhook Payload:**
```json
{
  "job_id": "uuid",
  "status": "completed", 
  "clips_count": 5,
  "clips": ["clip1.mp4", "clip2.mp4"],
  "download_links": ["/api/v1/download/job_id/clip1.mp4"],
  "bulk_download_url": "/api/v1/download/job_id"
}
```

---

## 🧪 Testing Examples

### Test with curl:

```bash
# Start job
curl -X POST http://localhost:8001/api/v1/extract-speaking \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://youtube.com/watch?v=jKC720uhRDc",
    "settings": {
      "clip_duration": 8,
      "speaker_visible": false,
      "max_clips_per_video": 3
    }
  }'

# Check status (replace JOB_ID)
curl http://localhost:8001/api/v1/status/JOB_ID

# Download ZIP  
curl -o clips.zip http://localhost:8001/api/v1/download/JOB_ID
```

### ngrok Setup for Local Testing:

```bash
# Terminal 1: Start your server
./run_server.sh

# Terminal 2: Expose with ngrok  
ngrok http 8001

# Use the ngrok URL in n8n:
# https://abc123.ngrok.io/api/v1/extract-speaking
```

---

## 🎬 Use Cases

### Content Creator Workflow
1. **Trigger:** New YouTube video published
2. **Extract:** Speaking segments for highlights
3. **Process:** Auto-generate social media clips
4. **Upload:** To TikTok/Instagram/Twitter

### Podcast Processing
1. **Input:** Long-form podcast URL
2. **Extract:** Quote segments (5-15 seconds)
3. **Generate:** Audiogram-ready clips
4. **Distribute:** Across platforms

### News/Interview Processing  
1. **Source:** Interview or news video
2. **Mode:** Quote extraction
3. **Filter:** High-confidence segments only
4. **Output:** Shareable sound bites

---

## 🔧 Error Handling

```javascript
// n8n Function Node - Error Handler
if ($input.first().json.status === 'failed') {
  throw new Error(`B-Roll extraction failed: ${$input.first().json.error}`);
}

if ($input.first().json.clips_count === 0) {
  return [{
    json: { 
      message: "No clips extracted - try different settings",
      suggestion: "Lower speech_duration or disable speaker_visible"
    }
  }];
}
```

---

## 📊 Monitoring & Analytics

Track in n8n:
- Processing times
- Success rates  
- Clip counts per video
- Popular video types
- Quality settings usage