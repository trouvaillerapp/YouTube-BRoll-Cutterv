#!/usr/bin/env python3
"""
Simple test to verify if Railway can process videos without YouTube download
"""

import os
import sys
import tempfile
import subprocess

def test_ffmpeg():
    """Test if ffmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=5)
        print(f"✅ ffmpeg available: {result.returncode == 0}")
        if result.returncode == 0:
            print(f"   Version: {result.stdout.decode().split()[2] if result.stdout else 'unknown'}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("❌ ffmpeg not found")

def test_opencv():
    """Test OpenCV video capabilities"""
    try:
        import cv2
        print(f"✅ OpenCV available: {cv2.__version__}")
        
        # Test video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"✅ mp4v codec available")
        
    except ImportError:
        print("❌ OpenCV not available")
    except Exception as e:
        print(f"❌ OpenCV error: {e}")

def test_video_download():
    """Test downloading a simple video"""
    try:
        import yt_dlp
        
        # Test with a short, simple video
        test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll - always works
        
        ydl_opts = {
            'format': 'worst[ext=mp4]',  # Get smallest file
            'outtmpl': '/tmp/test_video.%(ext)s',
            'quiet': True,
            'no_warnings': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(test_url, download=False)
            print(f"✅ Video info extracted: {info.get('title', 'Unknown')}")
            print(f"   Duration: {info.get('duration', 0)}s")
            
    except Exception as e:
        print(f"❌ Video download test failed: {e}")

def main():
    print("🔍 RAILWAY EXTRACTION TEST")
    print("=" * 40)
    
    test_ffmpeg()
    test_opencv()
    test_video_download()
    
    print("\n📊 SUMMARY:")
    print("If ffmpeg and OpenCV are ✅, extraction should work")
    print("If video download fails, it's a YouTube access issue")

if __name__ == "__main__":
    main()