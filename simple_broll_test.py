#!/usr/bin/env python3
"""
Simple test to debug B-roll processing
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from broll_cutter import YouTubeBRollCutter

def simple_progress(info):
    print(f"Progress: {info.get('progress', 0):.1f}% - {info.get('message', '')}")

# Test with a simple setup
print("🔧 Testing B-roll extraction with MVP settings...")

cutter = YouTubeBRollCutter(
    output_dir="./mvp_test_output",
    clip_duration=5.0,
    quality="480p",
    remove_watermarks=False,  # MVP: No watermarks
    enhance_video=False,      # MVP: No enhancement
    max_clips_per_video=3,
    scene_threshold=0.5,      # Less sensitive for MVP
    progress_callback=simple_progress
)

test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

print(f"🎬 Processing: {test_url}")
clips = cutter.extract_broll(test_url)

print(f"\n📊 Results:")
print(f"   Clips extracted: {len(clips)}")
for i, clip in enumerate(clips):
    print(f"   {i+1}. {clip}")

# Check if files actually exist
print(f"\n📁 File verification:")
for clip in clips:
    if Path(clip).exists():
        size = Path(clip).stat().st_size
        print(f"   ✅ {Path(clip).name} ({size} bytes)")
    else:
        print(f"   ❌ {Path(clip).name} - FILE NOT FOUND")