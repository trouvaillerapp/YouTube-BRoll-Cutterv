# 🎬 YouTube B-Roll Cutter with Speech Recognition

An intelligent AI-powered tool for extracting high-quality B-roll clips from YouTube videos with automatic speech detection, scene detection, and smart video processing.

## 🚀 Features

- **🎙️ Speech Detection**: Extract clips where people are speaking
- **💬 Quote Extraction**: Find 3-15 second quotable moments
- **🎬 Scene Detection**: AI-powered scene boundary detection
- **🔗 n8n Integration**: Full API support for workflow automation
- **⚡ Fast Processing**: Async job processing with status tracking
- **📦 Batch Support**: Process multiple videos simultaneously

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, OpenCV, yt-dlp
- **Speech Detection**: Custom audio analysis with face detection sync
- **API**: RESTful API with comprehensive n8n integration
- **Deployment**: Railway-ready with auto-scaling support

## 📋 Quick Start

### Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/YouTube-BRoll-Cutter.git
cd YouTube-BRoll-Cutter

# Install dependencies
pip install -r requirements.txt

# Start the server
python start_web_server.py
# or
./run_server.sh

# Access at http://localhost:8001
```

### Railway Deployment

1. Fork this repository
2. Connect to Railway.app
3. Deploy directly - Railway will auto-detect Python and use the Procfile

## 🔗 API Endpoints

### Extract Speaking Clips
```bash
POST /api/v1/extract-speaking
{
  "url": "https://youtube.com/watch?v=VIDEO_ID",
  "settings": {
    "clip_duration": 8,
    "speaker_visible": false,
    "max_clips_per_video": 5
  }
}
```

### Check Job Status
```bash
GET /api/v1/status/{job_id}
```

### Download Results
```bash
GET /api/v1/download/{job_id}  # ZIP file with all clips
GET /api/v1/download/{job_id}/{filename}  # Individual clip
```

## 🤖 n8n Integration

This tool is designed for seamless n8n workflow integration:

1. **Start extraction job** via HTTP Request node
2. **Poll status** until completion
3. **Download results** automatically

See [n8n-examples.md](n8n-examples.md) for detailed workflow examples.

## 📊 Settings

- **Clip Duration**: 3-30 seconds per clip
- **Quality**: 480p, 720p, 1080p, 4K
- **Speech Detection**: Configurable sensitivity
- **Output Formats**: MP4 (default), MOV, AVI

## 🔐 Security

- No API keys required for basic functionality
- Respects YouTube ToS
- Processes only publicly available videos

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with FastAPI, OpenCV, yt-dlp, and ❤️ for content creators