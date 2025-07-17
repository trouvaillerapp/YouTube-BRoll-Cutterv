"""
Simple fallback extractor that uses basic time-based clipping
"""

import logging
import cv2
import os
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class SimpleExtractor:
    """Simple time-based video clip extractor as fallback"""
    
    def __init__(self, output_dir: str = "./extracted_clips"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_time_based_clips(self, video_path: str, 
                                clip_duration: float = 8.0,
                                max_clips: int = 5,
                                start_time: float = 0,
                                job_id: str = "") -> List[str]:
        """
        Extract clips at regular intervals without scene detection
        
        Args:
            video_path: Path to video file
            clip_duration: Duration of each clip in seconds
            max_clips: Maximum number of clips to extract
            start_time: Start time in seconds
            job_id: Job ID for naming clips
            
        Returns:
            List of extracted clip paths
        """
        try:
            logger.info(f"Starting simple time-based extraction from: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Failed to open video: {video_path}")
                return []
                
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            logger.info(f"Video properties: duration={duration:.1f}s, fps={fps:.1f}, frames={total_frames}")
            
            if duration < clip_duration:
                logger.warning(f"Video too short ({duration:.1f}s) for {clip_duration}s clips")
                return []
            
            # Calculate clip intervals
            available_duration = duration - start_time
            if available_duration < clip_duration:
                logger.warning(f"Not enough video after start_time {start_time}s")
                return []
                
            # Distribute clips evenly across the video
            num_clips = min(max_clips, int(available_duration / (clip_duration + 5)))  # 5s gap between clips
            if num_clips == 0:
                num_clips = 1
                
            interval = available_duration / (num_clips + 1)
            
            clips = []
            for i in range(num_clips):
                clip_start = start_time + (i + 1) * interval - clip_duration / 2
                clip_start = max(start_time, min(clip_start, duration - clip_duration))
                
                output_path = self.output_dir / f"{job_id}_simple_clip_{i+1:02d}.mp4"
                
                logger.info(f"Extracting clip {i+1}/{num_clips} at {clip_start:.1f}s")
                
                if self._extract_clip_ffmpeg(video_path, clip_start, clip_duration, str(output_path)):
                    clips.append(str(output_path))
                    logger.info(f"Successfully extracted: {output_path.name}")
                else:
                    logger.warning(f"Failed to extract clip at {clip_start:.1f}s")
                    
            cap.release()
            
            logger.info(f"Simple extraction completed: {len(clips)}/{num_clips} clips extracted")
            return clips
            
        except Exception as e:
            logger.error(f"Simple extraction failed: {str(e)}")
            return []
            
    def _extract_clip_ffmpeg(self, input_path: str, start_time: float, 
                           duration: float, output_path: str) -> bool:
        """Extract a clip using ffmpeg (more reliable than OpenCV)"""
        try:
            import subprocess
            
            # Check if ffmpeg is available
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.error("ffmpeg not found - falling back to OpenCV")
                return self._extract_clip_opencv(input_path, start_time, duration, output_path)
            
            cmd = [
                'ffmpeg',
                '-i', input_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c', 'copy',  # Fast copy without re-encoding
                '-avoid_negative_ts', 'make_zero',
                '-y',  # Overwrite output
                output_path
            ]
            
            # Run ffmpeg with timeout
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            else:
                logger.debug(f"ffmpeg copy failed: {result.stderr}")
                # Fallback to re-encoding if copy fails
                cmd[6] = 'libx264'  # Replace 'copy' with 'libx264'
                cmd.insert(7, '-preset')
                cmd.insert(8, 'fast')
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                return result.returncode == 0 and os.path.exists(output_path)
                
        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg timeout for clip at {start_time}s")
            return False
        except Exception as e:
            logger.error(f"ffmpeg extraction failed: {e}")
            return self._extract_clip_opencv(input_path, start_time, duration, output_path)
    
    def _extract_clip_opencv(self, input_path: str, start_time: float, 
                           duration: float, output_path: str) -> bool:
        """Fallback clip extraction using OpenCV"""
        try:
            import cv2
            
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                logger.error(f"OpenCV failed to open: {input_path}")
                return False
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                logger.error("Invalid FPS detected")
                return False
                
            start_frame = int(start_time * fps)
            end_frame = int((start_time + duration) * fps)
            
            # Set codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, 
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = 0
            max_frames = end_frame - start_frame
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                out.write(frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            success = os.path.exists(output_path) and os.path.getsize(output_path) > 1000
            if success:
                logger.info(f"OpenCV extraction successful: {output_path}")
            else:
                logger.error(f"OpenCV extraction failed or file too small: {output_path}")
                
            return success
            
        except Exception as e:
            logger.error(f"OpenCV extraction failed: {e}")
            return False