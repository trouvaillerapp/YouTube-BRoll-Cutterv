#!/usr/bin/env python3
"""
Test script to verify B-Roll Cutter is working
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from broll_cutter import YouTubeBRollCutter
    print("✅ B-Roll Cutter module imported successfully")
    
    # Test with a simple video URL
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll (short video)
    
    print(f"🔍 Testing with URL: {test_url}")
    
    # Create a simple progress callback
    def progress_callback(info):
        progress = info.get('progress', 0)
        message = info.get('message', '')
        print(f"Progress: {progress:.1f}% - {message}")
    
    # Initialize with minimal settings
    cutter = YouTubeBRollCutter(
        output_dir="./test_output",
        clip_duration=5.0,
        quality="480p",
        max_clips_per_video=2,
        progress_callback=progress_callback
    )
    
    print("🚀 Starting B-roll extraction...")
    
    # Try to extract clips
    clips = cutter.extract_broll(test_url)
    
    if clips:
        print(f"✅ Successfully extracted {len(clips)} clips:")
        for i, clip in enumerate(clips):
            print(f"   {i+1}. {clip}")
    else:
        print("⚠️  No clips were extracted")
    
    # Show stats
    stats = cutter.get_processing_stats()
    print(f"\n📊 Processing Statistics:")
    print(f"   Videos processed: {stats['videos_processed']}")
    print(f"   Clips extracted: {stats['clips_extracted']}")
    
except ImportError as e:
    print(f"❌ Failed to import B-Roll Cutter: {e}")
    print("Make sure all dependencies are installed")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()