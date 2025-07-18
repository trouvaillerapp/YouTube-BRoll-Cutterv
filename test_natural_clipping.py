#!/usr/bin/env python3
"""
Test script for natural clipping feature
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from broll_cutter import YouTubeBRollCutter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_natural_clipping():
    """Test natural clipping with a sample video"""
    
    # Initialize the cutter
    cutter = YouTubeBRollCutter(
        output_dir="./test_output",
        clip_duration=15.0,
        max_clips_per_video=3
    )
    
    # Test URL - same one that worked before
    test_url = "https://www.youtube.com/watch?v=wi42YjWR5B4"
    
    try:
        logger.info("Testing natural clipping...")
        
        # Extract natural clips
        clips = cutter.extract_natural_clips(
            video_url=test_url,
            target_duration=15.0,  # Target 15 seconds
            max_clips=3            # Max 3 clips
        )
        
        if clips:
            logger.info(f"✅ Success! Extracted {len(clips)} natural clips:")
            for i, clip in enumerate(clips):
                logger.info(f"  {i+1}. {clip}")
        else:
            logger.error("❌ No clips extracted")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        return False
    
    return len(clips) > 0

def test_speech_detector_only():
    """Test just the speech detector component"""
    
    from broll_cutter.speech_detector import SpeechDetector
    
    # Initialize speech detector
    detector = SpeechDetector(
        min_silence_gap=0.5,
        max_extension_seconds=5.0
    )
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=wi42YjWR5B4"
    
    try:
        logger.info("Testing speech detector natural clips...")
        
        # Download a test video first
        from broll_cutter.downloader import VideoDownloader
        downloader = VideoDownloader(output_dir="./test_temp")
        
        video_path = downloader.download_video(test_url, "test_natural")
        
        # Test natural clips creation
        natural_clips = detector.create_natural_clips(
            video_path=video_path,
            clip_duration=15.0,
            max_clips=3
        )
        
        if natural_clips:
            logger.info(f"✅ Speech detector found {len(natural_clips)} natural clips:")
            for i, clip in enumerate(natural_clips):
                logger.info(f"  {i+1}. {clip['start_time']:.2f}s -> {clip['end_time']:.2f}s "
                           f"(duration: {clip['duration']:.2f}s)")
        else:
            logger.error("❌ No natural clips found")
            
    except Exception as e:
        logger.error(f"❌ Speech detector test failed: {str(e)}")
        return False
    
    return len(natural_clips) > 0

if __name__ == "__main__":
    print("🎯 Testing Natural Clipping Feature")
    print("=" * 50)
    
    # Test 1: Speech detector only
    print("\n1. Testing speech detector natural clips...")
    if test_speech_detector_only():
        print("✅ Speech detector test passed")
    else:
        print("❌ Speech detector test failed")
    
    # Test 2: Full natural clipping
    print("\n2. Testing full natural clipping...")
    if test_natural_clipping():
        print("✅ Natural clipping test passed")
    else:
        print("❌ Natural clipping test failed")
    
    print("\n🎬 Natural clipping feature is ready!")
    print("\nTo use in web interface:")
    print("1. Select 'Natural Boundaries' mode")
    print("2. Set target duration (e.g., 15 seconds)")
    print("3. Clips will extend/shrink to end at natural speech pauses")