#!/usr/bin/env python3
"""
Test diverse clip extraction
"""

import sys
import os
from pathlib import Path
import cv2

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def analyze_clip_diversity():
    """Analyze the clips in web_output to show time distribution"""
    
    print("🎬 Analyzing Clip Diversity in Recent Extractions")
    print("=" * 55)
    
    # Look for recent clips
    web_output = Path("web_output")
    if not web_output.exists():
        print("❌ No web_output directory found")
        return
    
    # Get all MP4 files, sorted by modification time
    clips = list(web_output.glob("*.mp4"))
    if not clips:
        print("❌ No clips found in web_output")
        return
    
    clips.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Analyze the most recent batch (assume same job ID prefix)
    if clips:
        recent_job = clips[0].stem.split('_clip_')[0]
        recent_clips = [c for c in clips if c.stem.startswith(recent_job)]
        
        print(f"📊 Most recent job: {recent_job}")
        print(f"📁 Clips found: {len(recent_clips)}")
        
        if len(recent_clips) >= 2:
            print(f"\n🎯 Clip Analysis:")
            
            clip_info = []
            for clip in recent_clips:
                cap = cv2.VideoCapture(str(clip))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                cap.release()
                
                size_mb = clip.stat().st_size / (1024 * 1024)
                clip_info.append({
                    'name': clip.name,
                    'duration': duration,
                    'size_mb': size_mb
                })
                
                print(f"  📹 {clip.name}: {duration:.1f}s ({size_mb:.1f}MB)")
            
            # Check if durations are similar (good) and if clips are likely different
            durations = [c['duration'] for c in clip_info]
            avg_duration = sum(durations) / len(durations)
            duration_variance = sum((d - avg_duration)**2 for d in durations) / len(durations)
            
            print(f"\n📈 Duration Analysis:")
            print(f"  Average duration: {avg_duration:.1f}s")
            print(f"  Duration consistency: {'✅ Good' if duration_variance < 2 else '⚠️ Variable'}")
            
            if len(clip_info) >= 3:
                size_variance = sum((c['size_mb'] - sum(c['size_mb'] for c in clip_info)/len(clip_info))**2 for c in clip_info) / len(clip_info)
                if size_variance > 0.1:
                    print(f"  Content diversity: ✅ Good (different file sizes suggest different content)")
                else:
                    print(f"  Content diversity: ⚠️ May be similar content")
        else:
            print("❌ Need at least 2 clips to analyze diversity")

if __name__ == "__main__":
    analyze_clip_diversity()