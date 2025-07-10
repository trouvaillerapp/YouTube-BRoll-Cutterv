#!/usr/bin/env python3
"""
Test with a modern HD video
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from broll_cutter import YouTubeBRollCutter

def test_hd_video():
    print("🎬 Testing with a modern HD video...")
    
    def progress_callback(info):
        if info.get('progress', 0) % 25 == 0:  # Show every 25%
            print(f"  Progress: {info.get('progress', 0):.0f}%")
    
    # Use a modern video that definitely has HD versions
    # Replace with any recent YouTube video URL that has good quality
    test_url = "https://www.youtube.com/watch?v=jNQXAC9IVRw"  # Popular video with HD
    
    cutter = YouTubeBRollCutter(
        output_dir="./hd_test_output",
        clip_duration=8.0,
        quality="720p",
        remove_watermarks=False,
        enhance_video=False,
        max_clips_per_video=2,
        progress_callback=progress_callback
    )
    
    try:
        clips = cutter.extract_broll(test_url)
        
        if clips:
            for i, clip_path in enumerate(clips):
                size_mb = Path(clip_path).stat().st_size / (1024 * 1024)
                print(f"  ✅ Clip {i+1}: {Path(clip_path).name} ({size_mb:.2f} MB)")
                
                # Check video properties
                import cv2
                cap = cv2.VideoCapture(clip_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                
                print(f"      📺 Resolution: {width}x{height} @ {fps:.1f}fps")
        else:
            print("  ❌ No clips extracted")
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")

if __name__ == "__main__":
    test_hd_video()