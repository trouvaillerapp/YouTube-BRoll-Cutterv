#!/usr/bin/env python3
"""
Test script for speech detection functionality
"""

from broll_cutter.speech_detector import SpeechDetector
from broll_cutter.core import YouTubeBRollCutter
import sys

def test_speech_detection():
    """Test speech detection on a sample video"""
    
    print("🎙️ Testing Speech Detection Module...")
    print("-" * 50)
    
    # Initialize speech detector
    detector = SpeechDetector(
        energy_threshold=0.02,
        silence_duration=0.5,
        min_speech_duration=1.0,
        enable_transcription=False,  # Disable transcription for MVP
        enable_face_sync=True
    )
    
    # Test with a downloaded video if available
    test_videos = [
        "test_download.mp4",
        "test_output/39850524_clip_01.mp4",
        "web_output/0a7ea9e3_clip_01.mp4"
    ]
    
    video_found = False
    for video_path in test_videos:
        try:
            print(f"\n📹 Testing with: {video_path}")
            
            # Detect speech segments
            segments = detector.detect_speech_segments(video_path)
            
            if segments:
                video_found = True
                print(f"\n✅ Found {len(segments)} speech segments:")
                for i, segment in enumerate(segments):
                    print(f"\n  Segment {i+1}:")
                    print(f"    Start: {segment.start_time:.2f}s")
                    print(f"    End: {segment.end_time:.2f}s")
                    print(f"    Duration: {segment.duration:.2f}s")
                    print(f"    Confidence: {segment.confidence:.2f}")
                    print(f"    Has Face: {segment.has_face}")
                    print(f"    Type: {segment.speech_type}")
                
                # Test quote extraction
                print("\n\n🎯 Testing Quote Extraction...")
                quotes = detector.extract_quotes(video_path)
                print(f"Found {len(quotes)} potential quotes")
                
                # Test speaking scene detection
                print("\n\n👤 Testing Speaking Scene Detection...")
                speaking_scenes = detector.find_speaking_scenes(video_path, speaker_visible=True)
                print(f"Found {len(speaking_scenes)} speaking scenes with visible speaker")
                
                break
            else:
                print("  No speech segments detected in this video")
                
        except Exception as e:
            print(f"  Error: {str(e)}")
            continue
    
    if not video_found:
        print("\n⚠️  No test videos found. Let's download a sample video first...")
        
        # Download a sample video
        sample_url = input("\nEnter a YouTube URL to test (or press Enter to skip): ").strip()
        
        if sample_url:
            print("\n📥 Downloading sample video...")
            cutter = YouTubeBRollCutter(output_dir="./test_output")
            
            try:
                # Download video
                video_path = cutter.downloader.download(sample_url)
                
                if video_path:
                    print(f"\n✅ Downloaded to: {video_path}")
                    
                    # Test speech detection on downloaded video
                    print("\n🎙️ Detecting speech segments...")
                    segments = detector.detect_speech_segments(video_path)
                    
                    if segments:
                        print(f"\n✅ Found {len(segments)} speech segments")
                        for i, segment in enumerate(segments[:5]):  # Show first 5
                            print(f"\n  Segment {i+1}:")
                            print(f"    Start: {segment.start_time:.2f}s")
                            print(f"    End: {segment.end_time:.2f}s")
                            print(f"    Duration: {segment.duration:.2f}s")
                            print(f"    Confidence: {segment.confidence:.2f}")
                    else:
                        print("❌ No speech segments detected")
                        
            except Exception as e:
                print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_speech_detection()