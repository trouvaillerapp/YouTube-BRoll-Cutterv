# N8N Integration Guide for YouTube B-Roll Cutter

## Overview
This guide shows how to integrate the YouTube B-Roll Cutter with n8n workflows for automated video processing.

## API Endpoints

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication
No authentication required for local deployment. For production, consider adding API keys.

## Available Endpoints

### 1. Extract B-Roll (Async)
**POST** `/api/v1/extract-broll`

Process a single video asynchronously.

#### Request Body:
```json
{
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "settings": {
    "clip_duration": 8.0,
    "quality": "720p",
    "max_clips_per_video": 5,
    "start_time": 0,
    "max_duration": 300,
    "scene_threshold": 0.3
  },
  "webhook_url": "https://your-n8n-webhook-url"
}
```

#### Response:
```json
{
  "job_id": "uuid-string",
  "status": "started",
  "message": "Processing started",
  "estimated_duration": "2-5 minutes",
  "status_url": "/api/v1/status/{job_id}",
  "download_url": "/api/v1/download/{job_id}"
}
```

### 2. Batch Extract (Async)
**POST** `/api/v1/batch-extract`

Process multiple videos asynchronously.

#### Request Body:
```json
{
  "urls": [
    "https://youtube.com/watch?v=VIDEO_ID1",
    "https://youtube.com/watch?v=VIDEO_ID2"
  ],
  "settings": {
    "clip_duration": 10.0,
    "quality": "720p",
    "max_clips_per_video": 3
  },
  "webhook_url": "https://your-n8n-webhook-url"
}
```

### 3. Process Sync (Synchronous)
**POST** `/api/v1/process-sync`

Process a single video synchronously (waits for completion).

#### Request Body:
```json
{
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "settings": {
    "clip_duration": 8.0,
    "quality": "720p",
    "max_clips_per_video": 5
  }
}
```

#### Response:
```json
{
  "status": "completed",
  "message": "Successfully extracted 5 clips",
  "clips_count": 5,
  "clips": ["clip1.mp4", "clip2.mp4", "clip3.mp4"],
  "download_links": ["/download/clip1.mp4", "/download/clip2.mp4"],
  "processing_time": "sync"
}
```

### 4. Get Job Status
**GET** `/api/v1/status/{job_id}`

Check the status of an async job.

#### Response:
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 100,
  "message": "Completed! Extracted 5 clips",
  "clips_count": 5,
  "clips": ["clip1.mp4", "clip2.mp4"],
  "elapsed_time_seconds": 180,
  "download_links": ["/api/v1/download/job_id/clip1.mp4"],
  "bulk_download_url": "/api/v1/download/job_id"
}
```

### 5. Download Files
**GET** `/api/v1/download/{job_id}` - Download all clips as ZIP
**GET** `/api/v1/download/{job_id}/{filename}` - Download single clip

## Settings Reference

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `clip_duration` | float | 8.0 | Duration of each clip in seconds (3-30) |
| `quality` | string | "720p" | Video quality: "480p", "720p", "1080p", "4k" |
| `max_clips_per_video` | int | 5 | Maximum clips to extract (1-20) |
| `start_time` | int | 0 | Start time in seconds (0 = beginning) |
| `max_duration` | int | null | Maximum duration to process in seconds |
| `scene_threshold` | float | 0.3 | Scene detection sensitivity (0.1-1.0) |

## N8N Workflow Examples

### Example 1: Simple B-Roll Extraction

```json
{
  "name": "YouTube B-Roll Extractor",
  "nodes": [
    {
      "name": "Trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "parameters": {},
      "position": [240, 300]
    },
    {
      "name": "Extract B-Roll",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://localhost:8000/api/v1/extract-broll",
        "method": "POST",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "url",
              "value": "https://youtube.com/watch?v=dQw4w9WgXcQ"
            },
            {
              "name": "settings",
              "value": {
                "clip_duration": 10,
                "quality": "720p",
                "max_clips_per_video": 3
              }
            }
          ]
        }
      },
      "position": [460, 300]
    },
    {
      "name": "Wait for Completion",
      "type": "n8n-nodes-base.wait",
      "parameters": {
        "amount": 300,
        "unit": "seconds"
      },
      "position": [680, 300]
    },
    {
      "name": "Check Status",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "=http://localhost:8000/api/v1/status/{{$json.job_id}}",
        "method": "GET"
      },
      "position": [900, 300]
    },
    {
      "name": "Download Clips",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "=http://localhost:8000/api/v1/download/{{$json.job_id}}",
        "method": "GET",
        "responseFormat": "file"
      },
      "position": [1120, 300]
    }
  ],
  "connections": {
    "Trigger": {
      "main": [
        [
          {
            "node": "Extract B-Roll",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Extract B-Roll": {
      "main": [
        [
          {
            "node": "Wait for Completion",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Wait for Completion": {
      "main": [
        [
          {
            "node": "Check Status",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Check Status": {
      "main": [
        [
          {
            "node": "Download Clips",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### Example 2: Webhook-Based Processing

```json
{
  "name": "YouTube B-Roll with Webhook",
  "nodes": [
    {
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "broll-completed",
        "httpMethod": "POST"
      },
      "position": [240, 300]
    },
    {
      "name": "Manual Trigger",
      "type": "n8n-nodes-base.manualTrigger",
      "parameters": {},
      "position": [240, 500]
    },
    {
      "name": "Extract B-Roll",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://localhost:8000/api/v1/extract-broll",
        "method": "POST",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "url",
              "value": "https://youtube.com/watch?v=dQw4w9WgXcQ"
            },
            {
              "name": "settings",
              "value": {
                "clip_duration": 15,
                "quality": "1080p",
                "max_clips_per_video": 5
              }
            },
            {
              "name": "webhook_url",
              "value": "={{$node.Webhook Trigger.json.webhookUrl}}"
            }
          ]
        }
      },
      "position": [460, 500]
    },
    {
      "name": "Process Completed",
      "type": "n8n-nodes-base.set",
      "parameters": {
        "values": {
          "string": [
            {
              "name": "status",
              "value": "=Processing completed with {{$json.clips_count}} clips"
            }
          ]
        }
      },
      "position": [460, 300]
    },
    {
      "name": "Download All Clips",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "={{$json.bulk_download_url}}",
        "method": "GET",
        "responseFormat": "file"
      },
      "position": [680, 300]
    }
  ],
  "connections": {
    "Webhook Trigger": {
      "main": [
        [
          {
            "node": "Process Completed",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Manual Trigger": {
      "main": [
        [
          {
            "node": "Extract B-Roll",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Process Completed": {
      "main": [
        [
          {
            "node": "Download All Clips",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```

### Example 3: Batch Processing with Google Sheets

```json
{
  "name": "Batch B-Roll from Google Sheets",
  "nodes": [
    {
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.cron",
      "parameters": {
        "triggerTimes": {
          "item": [
            {
              "hour": 9,
              "minute": 0
            }
          ]
        }
      },
      "position": [240, 300]
    },
    {
      "name": "Read Google Sheet",
      "type": "n8n-nodes-base.googleSheets",
      "parameters": {
        "operation": "read",
        "documentId": "YOUR_SHEET_ID",
        "sheetName": "YouTube URLs",
        "range": "A:B"
      },
      "position": [460, 300]
    },
    {
      "name": "Process Each URL",
      "type": "n8n-nodes-base.splitInBatches",
      "parameters": {
        "batchSize": 1
      },
      "position": [680, 300]
    },
    {
      "name": "Extract B-Roll",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "http://localhost:8000/api/v1/extract-broll",
        "method": "POST",
        "sendHeaders": true,
        "headerParameters": {
          "parameters": [
            {
              "name": "Content-Type",
              "value": "application/json"
            }
          ]
        },
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "url",
              "value": "={{$json.url}}"
            },
            {
              "name": "settings",
              "value": {
                "clip_duration": "={{$json.duration || 8}}",
                "quality": "720p",
                "max_clips_per_video": 5
              }
            }
          ]
        }
      },
      "position": [900, 300]
    },
    {
      "name": "Wait",
      "type": "n8n-nodes-base.wait",
      "parameters": {
        "amount": 300,
        "unit": "seconds"
      },
      "position": [1120, 300]
    },
    {
      "name": "Check Status",
      "type": "n8n-nodes-base.httpRequest",
      "parameters": {
        "url": "=http://localhost:8000/api/v1/status/{{$json.job_id}}",
        "method": "GET"
      },
      "position": [1340, 300]
    },
    {
      "name": "Update Sheet",
      "type": "n8n-nodes-base.googleSheets",
      "parameters": {
        "operation": "update",
        "documentId": "YOUR_SHEET_ID",
        "sheetName": "Results",
        "range": "A:C",
        "values": {
          "values": [
            [
              "={{$json.job_id}}",
              "={{$json.clips_count}}",
              "={{$json.status}}"
            ]
          ]
        }
      },
      "position": [1560, 300]
    }
  ]
}
```

## Setup Instructions

### 1. Start the B-Roll Cutter Server
```bash
cd /path/to/YouTube-BRoll-Cutter
python backend/app.py
```

### 2. Configure N8N
1. Install n8n: `npm install -g n8n`
2. Start n8n: `n8n start`
3. Access n8n at `http://localhost:5678`

### 3. Import Workflow
1. Create a new workflow in n8n
2. Copy one of the example workflows above
3. Paste into the JSON editor
4. Configure your specific URLs and settings

### 4. Test the Integration
1. Use the manual trigger to test
2. Check the B-Roll Cutter logs for processing status
3. Download the generated clips

## Webhook Configuration

When using webhooks, n8n will receive notifications when processing completes:

**Success Webhook Payload:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "clips_count": 5,
  "clips": ["clip1.mp4", "clip2.mp4"],
  "download_links": ["/api/v1/download/job_id/clip1.mp4"],
  "bulk_download_url": "/api/v1/download/job_id",
  "message": "Successfully extracted 5 B-roll clips"
}
```

**Error Webhook Payload:**
```json
{
  "job_id": "uuid-string",
  "status": "failed",
  "error": "Error message",
  "message": "Processing failed: Error message"
}
```

## Best Practices

1. **Use Async Processing**: For videos longer than 5 minutes, use async endpoints
2. **Monitor Job Status**: Always check job status before downloading
3. **Handle Errors**: Implement error handling in your workflows
4. **Rate Limiting**: Don't process too many videos simultaneously
5. **Cleanup**: Delete old clips periodically to save disk space

## Troubleshooting

### Common Issues:

1. **Connection Refused**: Ensure the B-Roll Cutter server is running
2. **Timeout**: Increase wait time for longer videos
3. **No Clips**: Check video availability and scene detection settings
4. **Download Failed**: Verify job completed successfully

### Debug Steps:
1. Check server logs: `python backend/app.py`
2. Test API endpoints manually with curl/Postman
3. Verify n8n webhook URLs are accessible
4. Check disk space for output files

## API Documentation
Visit `http://localhost:8000/api/v1/docs` for complete API documentation.