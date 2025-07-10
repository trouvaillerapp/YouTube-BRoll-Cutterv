# 🎬 YouTube B-Roll Cutter

An intelligent AI-powered tool for extracting high-quality B-roll clips from YouTube videos with automatic scene detection, watermark removal, and smart video processing.

## 🚀 Features

- **Smart Video Download**: Uses yt-dlp for optimal quality YouTube downloads
- **Intelligent Scene Detection**: AI-powered scene boundary detection with OpenCV
- **Watermark Removal**: Advanced inpainting algorithm to remove logos and watermarks
- **Custom Clip Duration**: Configurable clip lengths (5-30 seconds)
- **Batch Processing**: Process multiple videos simultaneously
- **Quality Enhancement**: Automatic video stabilization and enhancement
- **Format Support**: Export to MP4, MOV, AVI with custom quality settings
- **Progress Tracking**: Real-time progress updates and logging

## 📋 Requirements

- Python 3.8+
- FFmpeg
- OpenCV
- yt-dlp
- NumPy
- Pillow

## 🛠️ Installation

```bash
# Clone or download the project
cd YouTube-BRoll-Cutter

# Install dependencies
pip install -r requirements.txt

# Install FFmpeg (macOS)
brew install ffmpeg

# Install FFmpeg (Ubuntu)
sudo apt install ffmpeg
```

## 🎯 Quick Start

```python
from broll_cutter import YouTubeBRollCutter

# Initialize the cutter
cutter = YouTubeBRollCutter(
    output_dir="./extracted_clips",
    clip_duration=8,
    quality="720p"
)

# Process a single video
clips = cutter.extract_broll("https://youtube.com/watch?v=VIDEO_ID")

# Batch process multiple videos
video_urls = [
    "https://youtube.com/watch?v=VIDEO1",
    "https://youtube.com/watch?v=VIDEO2"
]
all_clips = cutter.batch_process(video_urls)
```

## 📊 Advanced Usage

### Custom Scene Detection
```python
cutter = YouTubeBRollCutter(
    scene_threshold=0.3,        # Sensitivity (0.1-1.0)
    min_scene_length=5,         # Minimum seconds per scene
    max_scene_length=15,        # Maximum seconds per scene
    enable_face_detection=True  # Prioritize scenes with people
)
```

### Watermark Removal
```python
cutter = YouTubeBRollCutter(
    remove_watermarks=True,
    watermark_regions=[          # Define watermark areas
        {"x": 0, "y": 0, "w": 200, "h": 100},      # Top-left logo
        {"x": -150, "y": -50, "w": 150, "h": 50}   # Bottom-right bug
    ]
)
```

### Quality & Enhancement
```python
cutter = YouTubeBRollCutter(
    quality="1080p",
    enhance_video=True,         # Auto-enhance colors/contrast
    stabilize_video=True,       # Remove camera shake
    denoise_audio=True,         # Clean up audio
    export_format="mov"         # MOV, MP4, AVI
)
```

## 🔧 Configuration

Create a `config.yml` file:

```yaml
# Video Processing
quality: "720p"              # 480p, 720p, 1080p, 4k
clip_duration: 8              # Default clip length in seconds
scene_threshold: 0.25         # Scene detection sensitivity

# Enhancement
enable_stabilization: true
enable_color_correction: true
enable_noise_reduction: false

# Watermark Removal
remove_watermarks: true
auto_detect_watermarks: true

# Output
output_format: "mp4"
output_quality: "high"        # low, medium, high, maximum
preserve_audio: true

# Performance
max_concurrent_downloads: 3
use_gpu_acceleration: false
temp_dir: "./temp"
```

## 📁 Project Structure

```
YouTube-BRoll-Cutter/
├── broll_cutter/
│   ├── __init__.py
│   ├── core.py              # Main BRollCutter class
│   ├── downloader.py        # YouTube download logic
│   ├── scene_detector.py    # Scene detection algorithms
│   ├── watermark_remover.py # Watermark removal
│   ├── video_processor.py   # Video processing pipeline
│   └── utils.py             # Helper functions
├── examples/
│   ├── basic_usage.py
│   ├── batch_processing.py
│   └── advanced_config.py
├── tests/
├── config.yml
├── requirements.txt
└── README.md
```

## 🎨 GUI Application

Launch the desktop application:

```bash
python gui_app.py
```

Features:
- Drag & drop YouTube URLs
- Real-time preview of detected scenes
- Visual watermark marking
- One-click batch processing
- Built-in video player for review

## 🔍 API Reference

### YouTubeBRollCutter Class

#### Methods

- `extract_broll(url, custom_settings=None)` - Extract B-roll from single video
- `batch_process(urls, callback=None)` - Process multiple videos
- `detect_scenes(video_path)` - Manual scene detection
- `remove_watermarks(video_path, regions=None)` - Remove watermarks
- `enhance_video(video_path, settings=None)` - Enhance video quality

#### Properties

- `supported_formats` - List of supported output formats
- `processing_stats` - Statistics about processed videos
- `current_progress` - Real-time processing progress

## 🧪 Examples

### News B-Roll Extraction
```python
# Extract clean news footage
news_cutter = YouTubeBRollCutter(
    clip_duration=10,
    remove_watermarks=True,
    enhance_video=True,
    priority_scenes=["talking_heads", "establishing_shots"]
)

clips = news_cutter.extract_broll("https://youtube.com/watch?v=NEWS_VIDEO")
```

### Travel B-Roll Collection
```python
# Extract scenic travel footage
travel_cutter = YouTubeBRollCutter(
    clip_duration=6,
    scene_types=["landscape", "cityscape", "nature"],
    color_enhancement=True,
    stabilization=True
)

clips = travel_cutter.extract_broll("https://youtube.com/watch?v=TRAVEL_VIDEO")
```

## 🚀 Performance

- **Processing Speed**: ~2-4x video length (depending on settings)
- **Memory Usage**: ~1-2GB for 1080p videos
- **Concurrent Processing**: Up to 5 videos simultaneously
- **Formats Supported**: MP4, MOV, AVI, WebM input/output

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Legal Notice

This tool is for educational and fair use purposes only. Users are responsible for complying with YouTube's Terms of Service and copyright laws. Only process videos you have permission to use.

## 🙏 Acknowledgments

- **yt-dlp** - YouTube video downloading
- **OpenCV** - Computer vision and video processing
- **FFmpeg** - Video encoding and processing
- **NumPy** - Numerical computing for video analysis

---

Built with ❤️ for content creators and video editors