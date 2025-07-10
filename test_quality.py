#!/usr/bin/env python3
"""
Test video quality with different settings
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from broll_cutter import YouTubeBRollCutter

def test_quality(quality, test_name):
    print(f"\n🎬 Testing {test_name} quality...")
    
    def progress_callback(info):
        if info.get('progress', 0) % 20 == 0:  # Only show every 20%
            print(f"  Progress: {info.get('progress', 0):.0f}%")
    
    output_dir = f"./quality_test_{quality}"
    Path(output_dir).mkdir(exist_ok=True)
    
    cutter = YouTubeBRollCutter(
        output_dir=output_dir,
        clip_duration=5.0,
        quality=quality,
        remove_watermarks=False,
        enhance_video=False,
        max_clips_per_video=1,  # Just one clip for testing
        progress_callback=progress_callback
    )
    
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    clips = cutter.extract_broll(test_url)
    
    if clips:
        clip_path = clips[0]
        size_mb = Path(clip_path).stat().st_size / (1024 * 1024)
        print(f"  ✅ Clip created: {Path(clip_path).name}")
        print(f"  📊 File size: {size_mb:.2f} MB")
        
        # Check video properties
        import cv2
        cap = cv2.VideoCapture(clip_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()
        
        print(f"  📺 Resolution: {width}x{height}")
        print(f"  🎞️  Duration: {duration:.1f}s @ {fps:.1f}fps")
        
        return size_mb
    else:
        print(f"  ❌ No clips extracted")
        return 0

# Test different qualities
qualities = [
    ("480p", "Fast"),
    ("720p", "Recommended"),
    ("1080p", "High Quality")
]

print("🔍 Testing video quality at different settings...")

for quality, name in qualities:
    try:
        size = test_quality(quality, name)
        if size > 0:
            print(f"✅ {quality}: {size:.2f} MB")
        else:
            print(f"❌ {quality}: Failed")
    except Exception as e:
        print(f"❌ {quality}: Error - {str(e)}")

print(f"\n🎯 Quality test completed!")