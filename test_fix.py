#!/usr/bin/env python3
"""Test the fixed video processor"""

import sys
import os
sys.path.append('.')
from broll_cutter.video_processor import VideoProcessor
import cv2

print('🔧 Testing fixed video processor...')
processor = VideoProcessor(output_dir='./test_fix', enable_enhancement=False)

clip_path = processor.extract_clip(
    video_path='debug_pipeline/temp/debug_source.mp4',
    start_time=0,
    duration=5,
    output_name='fixed_test_clip'
)

if clip_path and os.path.exists(clip_path):
    print(f'✅ Fixed clip created: {clip_path}')
    
    cap = cv2.VideoCapture(clip_path)
    frames_readable = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    for i in range(0, total_frames, max(1, total_frames//10)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames_readable += 1
    cap.release()
    
    print(f'🎞️  Frames readable: {frames_readable}/10')
    if frames_readable > 8:
        print('✅ FIX SUCCESSFUL! Clips are now working properly')
    else:
        print('❌ Fix did not work')
        
    # Check file size
    size_kb = os.path.getsize(clip_path) / 1024
    print(f'📊 File size: {size_kb:.0f} KB')
else:
    print('❌ Clip creation failed')