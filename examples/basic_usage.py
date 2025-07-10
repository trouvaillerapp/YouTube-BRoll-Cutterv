#!/usr/bin/env python3
"""
Basic usage example for YouTube B-Roll Cutter
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from broll_cutter import YouTubeBRollCutter
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def basic_example():
    """Basic B-roll extraction example"""
    
    print("🎬 YouTube B-Roll Cutter - Basic Example")
    print("=" * 50)
    
    # Example YouTube URLs (replace with actual URLs)
    video_urls = [
        "https://youtube.com/watch?v=dQw4w9WgXcQ",  # Example URL
        "https://youtube.com/watch?v=9bZkp7q19f0",  # Another example
    ]
    
    # Initialize the B-roll cutter
    cutter = YouTubeBRollCutter(
        output_dir="./extracted_clips",
        clip_duration=8.0,
        quality="720p",
        remove_watermarks=True,
        max_clips_per_video=5
    )
    
    # Process each video
    for i, url in enumerate(video_urls):
        print(f"\n📹 Processing video {i+1}/{len(video_urls)}")
        print(f"URL: {url}")
        
        try:
            # Extract B-roll clips
            clips = cutter.extract_broll(url)
            
            if clips:
                print(f"✅ Successfully extracted {len(clips)} clips:")
                for j, clip_path in enumerate(clips):
                    print(f"   {j+1}. {clip_path}")
            else:
                print("⚠️  No clips extracted from this video")
                
        except Exception as e:
            print(f"❌ Error processing video: {str(e)}")
    
    # Show statistics
    stats = cutter.get_processing_stats()
    print(f"\n📊 Processing Statistics:")
    print(f"   Videos processed: {stats['videos_processed']}")
    print(f"   Clips extracted: {stats['clips_extracted']}")
    print(f"   Total duration processed: {stats['total_duration_processed']:.1f}s")
    print(f"   Total clips duration: {stats['total_clips_duration']:.1f}s")
    print(f"   Watermarks removed: {stats['watermarks_removed']}")

def custom_settings_example():
    """Example with custom settings"""
    
    print("\n🔧 Custom Settings Example")
    print("=" * 30)
    
    # Custom progress callback
    def progress_callback(info):
        print(f"Progress: {info.get('message', '')} ({info.get('progress', 0):.1f}%)")
    
    # Initialize with custom settings
    cutter = YouTubeBRollCutter(
        output_dir="./custom_clips",
        clip_duration=10.0,
        quality="1080p",
        scene_threshold=0.2,  # More sensitive scene detection
        min_scene_length=5.0,
        max_scene_length=20.0,
        remove_watermarks=True,
        enhance_video=True,
        max_clips_per_video=8,
        progress_callback=progress_callback
    )
    
    url = "https://youtube.com/watch?v=EXAMPLE_URL"
    
    try:
        # Custom settings for this specific video
        custom_settings = {
            'clip_duration': 12.0,  # Override default
            'enhancement': {
                'brightness': 0.1,
                'contrast': 1.2,
                'saturation': 1.1,
                'denoise': True
            }
        }
        
        clips = cutter.extract_broll(url, custom_settings=custom_settings)
        print(f"✅ Extracted {len(clips)} enhanced clips")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def preview_example():
    """Example of previewing scenes before extraction"""
    
    print("\n👁️  Scene Preview Example")
    print("=" * 25)
    
    cutter = YouTubeBRollCutter()
    url = "https://youtube.com/watch?v=EXAMPLE_URL"
    
    try:
        # Preview scenes without extracting
        scenes = cutter.preview_scenes(url, output_dir="./scene_previews")
        
        print(f"📋 Detected {len(scenes)} scenes:")
        for i, scene in enumerate(scenes):
            print(f"   Scene {i+1}: {scene['start_time']:.1f}s - {scene['end_time']:.1f}s")
            print(f"      Type: {scene['scene_type']}, Confidence: {scene['confidence']:.2f}")
            print(f"      Motion: {scene['motion_level']}, Has faces: {scene['has_faces']}")
        
        # Preview watermark detection
        watermarks = cutter.detect_watermarks_preview(url, "watermark_preview.jpg")
        print(f"\n🏷️  Detected {len(watermarks)} potential watermarks")
        
    except Exception as e:
        print(f"❌ Preview failed: {str(e)}")

def batch_processing_example():
    """Example of batch processing multiple videos"""
    
    print("\n🔄 Batch Processing Example")
    print("=" * 30)
    
    # Multiple video URLs
    video_urls = [
        "https://youtube.com/watch?v=URL1",
        "https://youtube.com/watch?v=URL2", 
        "https://youtube.com/watch?v=URL3",
    ]
    
    # Callback for each completed video
    def video_completed_callback(url, clips, error):
        if error:
            print(f"❌ Failed: {url} - {str(error)}")
        else:
            print(f"✅ Completed: {url} - {len(clips)} clips extracted")
    
    cutter = YouTubeBRollCutter(
        output_dir="./batch_clips",
        max_clips_per_video=3  # Limit clips per video for faster processing
    )
    
    try:
        # Process all videos in parallel
        results = cutter.batch_process(
            video_urls, 
            max_workers=2,  # Process 2 videos simultaneously
            callback=video_completed_callback
        )
        
        # Show results
        total_clips = sum(len(clips) for clips in results.values())
        print(f"\n📊 Batch processing completed:")
        print(f"   Videos processed: {len(results)}")
        print(f"   Total clips extracted: {total_clips}")
        
    except Exception as e:
        print(f"❌ Batch processing failed: {str(e)}")

if __name__ == "__main__":
    try:
        # Run examples
        basic_example()
        custom_settings_example()
        preview_example()
        batch_processing_example()
        
        print(f"\n🎉 All examples completed!")
        print(f"📁 Check the output directories for extracted clips:")
        print(f"   - ./extracted_clips")
        print(f"   - ./custom_clips")
        print(f"   - ./batch_clips")
        print(f"   - ./scene_previews")
        
    except KeyboardInterrupt:
        print("\n⏹️  Examples interrupted by user")
    except Exception as e:
        print(f"\n💥 Example failed: {str(e)}")