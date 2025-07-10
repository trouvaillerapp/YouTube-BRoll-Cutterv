"""
Video processing module for clip extraction, enhancement, and format conversion
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import subprocess
import tempfile
import json
from tqdm import tqdm

try:
    from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    logging.warning("MoviePy not available - some features will be limited")

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Handles video clip extraction, enhancement, and format conversion"""
    
    def __init__(self,
                 output_dir: str = "./processed_clips",
                 temp_dir: str = None,
                 enable_enhancement: bool = True,
                 output_format: str = "mp4",
                 output_quality: str = "high"):
        """
        Initialize video processor
        
        Args:
            output_dir: Directory for processed video output
            temp_dir: Temporary directory for processing
            enable_enhancement: Enable video enhancement features
            output_format: Output video format (mp4, mov, avi)
            output_quality: Output quality (low, medium, high, maximum)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "video_processor"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_enhancement = enable_enhancement
        self.output_format = output_format
        self.output_quality = output_quality
        
        # Quality settings mapping
        self.quality_settings = {
            "low": {"crf": 28, "bitrate": "500k", "preset": "fast"},
            "medium": {"crf": 23, "bitrate": "1500k", "preset": "medium"},
            "high": {"crf": 18, "bitrate": "3000k", "preset": "medium"},
            "maximum": {"crf": 15, "bitrate": "5000k", "preset": "slow"}
        }
        
        # Check FFmpeg availability
        self.ffmpeg_available = self._check_ffmpeg()
        
        logger.info(f"Video processor initialized (FFmpeg: {self.ffmpeg_available})")
    
    def _check_ffmpeg(self) -> bool:
        """Check if FFmpeg is available"""
        try:
            subprocess.run(['ffmpeg', '-version'], 
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.warning("FFmpeg not available - some features will be limited")
            return False
    
    def extract_clip(self, 
                    video_path: str,
                    start_time: float,
                    duration: float,
                    output_name: str,
                    scene_info: Optional[Dict] = None,
                    custom_settings: Optional[Dict] = None) -> str:
        """
        Extract a clip from video
        
        Args:
            video_path: Path to source video
            start_time: Start time in seconds
            duration: Clip duration in seconds
            output_name: Name for output file (without extension)
            scene_info: Optional scene information for metadata
            custom_settings: Optional custom processing settings
            
        Returns:
            Path to extracted clip
        """
        try:
            logger.info(f"Extracting clip: {output_name} ({start_time:.1f}s - {start_time + duration:.1f}s)")
            
            output_path = self.output_dir / f"{output_name}.{self.output_format}"
            
            # Use FFmpeg if available (faster and more reliable)
            if self.ffmpeg_available:
                clip_path = self._extract_clip_ffmpeg(
                    video_path, start_time, duration, str(output_path), custom_settings
                )
            elif MOVIEPY_AVAILABLE:
                clip_path = self._extract_clip_moviepy(
                    video_path, start_time, duration, str(output_path), custom_settings
                )
            else:
                raise RuntimeError("No video processing backend available (FFmpeg or MoviePy)")
            
            # Add metadata if scene info provided
            if scene_info and clip_path:
                self._add_metadata(clip_path, scene_info)
            
            # Enhance video if enabled
            if self.enable_enhancement and clip_path:
                enhanced_path = self._enhance_video(clip_path, custom_settings)
                if enhanced_path != clip_path:
                    # Replace original with enhanced version
                    Path(clip_path).unlink()
                    Path(enhanced_path).rename(clip_path)
            
            logger.info(f"Clip extracted successfully: {clip_path}")
            return clip_path
            
        except Exception as e:
            logger.error(f"Clip extraction failed: {str(e)}")
            return None
    
    def _extract_clip_ffmpeg(self, 
                            video_path: str,
                            start_time: float,
                            duration: float,
                            output_path: str,
                            custom_settings: Optional[Dict] = None) -> str:
        """Extract clip using FFmpeg"""
        
        settings = self.quality_settings[self.output_quality].copy()
        if custom_settings:
            settings.update(custom_settings)
        
        # Build FFmpeg command
        # Put -ss after -i for frame-accurate seeking
        cmd = [
            'ffmpeg', '-y',  # Overwrite output
            '-i', video_path,
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:v', 'libx264',
            '-crf', str(settings['crf']),
            '-preset', settings['preset'],
            '-c:a', 'aac',
            '-b:a', '128k',
            '-movflags', '+faststart',  # Optimize for streaming
            output_path
        ]
        
        # Run FFmpeg
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg failed: {result.stderr}")
        
        return output_path
    
    def _extract_clip_moviepy(self,
                             video_path: str,
                             start_time: float,
                             duration: float,
                             output_path: str,
                             custom_settings: Optional[Dict] = None) -> str:
        """Extract clip using MoviePy"""
        
        with VideoFileClip(video_path) as video:
            # Extract clip
            clip = video.subclip(start_time, start_time + duration)
            
            # Write clip
            clip.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=str(self.temp_dir / 'temp_audio.m4a'),
                remove_temp=True
            )
        
        return output_path
    
    def _enhance_video(self, video_path: str, custom_settings: Optional[Dict] = None) -> str:
        """Enhance video quality using various techniques"""
        
        if not self.enable_enhancement:
            return video_path
        
        try:
            logger.info(f"Enhancing video: {Path(video_path).name}")
            
            enhanced_path = str(Path(video_path).with_suffix('.enhanced.mp4'))
            
            # Enhancement settings
            enhancement_settings = {
                'brightness': 0.05,
                'contrast': 1.1,
                'saturation': 1.05,
                'denoise': True,
                'sharpen': False,
                'stabilize': False
            }
            
            if custom_settings:
                enhancement_settings.update(custom_settings.get('enhancement', {}))
            
            if self.ffmpeg_available:
                self._enhance_with_ffmpeg(video_path, enhanced_path, enhancement_settings)
            else:
                self._enhance_with_opencv(video_path, enhanced_path, enhancement_settings)
            
            return enhanced_path
            
        except Exception as e:
            logger.warning(f"Video enhancement failed: {str(e)}")
            return video_path
    
    def _enhance_with_ffmpeg(self, input_path: str, output_path: str, settings: Dict):
        """Enhance video using FFmpeg filters"""
        
        filters = []
        
        # Color adjustments
        if settings.get('brightness', 0) != 0 or settings.get('contrast', 1) != 1:
            brightness = settings.get('brightness', 0)
            contrast = settings.get('contrast', 1)
            filters.append(f"eq=brightness={brightness}:contrast={contrast}")
        
        if settings.get('saturation', 1) != 1:
            saturation = settings.get('saturation', 1)
            filters.append(f"eq=saturation={saturation}")
        
        # Denoising
        if settings.get('denoise', False):
            filters.append("hqdn3d=2:1:2:1")
        
        # Sharpening
        if settings.get('sharpen', False):
            filters.append("unsharp=5:5:1.0:5:5:0.0")
        
        # Stabilization
        if settings.get('stabilize', False):
            filters.append("deshake")
        
        # Build command
        cmd = ['ffmpeg', '-y', '-i', input_path]
        
        if filters:
            filter_string = ','.join(filters)
            cmd.extend(['-vf', filter_string])
        
        cmd.extend([
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'medium',
            '-c:a', 'copy',
            output_path
        ])
        
        # Run enhancement
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Enhancement failed: {result.stderr}")
    
    def _enhance_with_opencv(self, input_path: str, output_path: str, settings: Dict):
        """Enhance video using OpenCV (basic enhancement)"""
        
        cap = cv2.VideoCapture(input_path)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        with tqdm(total=total_frames, desc="Enhancing video") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Apply enhancements
                enhanced_frame = self._enhance_frame(frame, settings)
                out.write(enhanced_frame)
                
                pbar.update(1)
        
        cap.release()
        out.release()
    
    def _enhance_frame(self, frame: np.ndarray, settings: Dict) -> np.ndarray:
        """Enhance a single frame"""
        
        enhanced = frame.copy()
        
        # Brightness and contrast
        brightness = settings.get('brightness', 0)
        contrast = settings.get('contrast', 1)
        
        if brightness != 0 or contrast != 1:
            enhanced = cv2.convertScaleAbs(enhanced, alpha=contrast, beta=brightness * 255)
        
        # Saturation adjustment
        saturation = settings.get('saturation', 1)
        if saturation != 1:
            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Denoising
        if settings.get('denoise', False):
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Sharpening
        if settings.get('sharpen', False):
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _add_metadata(self, video_path: str, scene_info: Dict):
        """Add metadata to video file"""
        
        metadata_path = Path(video_path).with_suffix('.json')
        
        metadata = {
            'video_file': Path(video_path).name,
            'extracted_at': str(pd.Timestamp.now()) if 'pd' in globals() else "unknown",
            'scene_info': scene_info,
            'processor_settings': {
                'output_quality': self.output_quality,
                'enhancement_enabled': self.enable_enhancement
            }
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_scene_previews(self, 
                            video_path: str,
                            scenes: List,
                            output_dir: str,
                            job_id: str) -> List[str]:
        """
        Create preview images for detected scenes
        
        Args:
            video_path: Path to source video
            scenes: List of Scene objects
            output_dir: Directory to save preview images
            job_id: Job identifier for naming
            
        Returns:
            List of paths to created preview images
        """
        try:
            logger.info(f"Creating scene previews for {len(scenes)} scenes")
            
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            preview_paths = []
            
            for i, scene in enumerate(scenes):
                # Get frame from middle of scene
                mid_time = (scene.start_time + scene.end_time) / 2
                mid_frame = int(mid_time * fps)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                
                if ret:
                    # Add scene information overlay
                    frame_with_info = self._add_scene_info_overlay(frame, scene, i + 1)
                    
                    # Save preview image
                    preview_path = Path(output_dir) / f"{job_id}_scene_{i+1:02d}_preview.jpg"
                    cv2.imwrite(str(preview_path), frame_with_info)
                    preview_paths.append(str(preview_path))
            
            cap.release()
            
            logger.info(f"Created {len(preview_paths)} scene preview images")
            return preview_paths
            
        except Exception as e:
            logger.error(f"Scene preview creation failed: {str(e)}")
            return []
    
    def _add_scene_info_overlay(self, frame: np.ndarray, scene, scene_number: int) -> np.ndarray:
        """Add scene information overlay to frame"""
        
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, height - 120), (400, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Add text information
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = (255, 255, 255)
        
        texts = [
            f"Scene {scene_number}",
            f"Duration: {scene.duration:.1f}s",
            f"Type: {scene.scene_type}",
            f"Confidence: {scene.confidence:.2f}",
            f"Motion: {scene.motion_level}"
        ]
        
        for i, text in enumerate(texts):
            y_pos = height - 100 + (i * 20)
            cv2.putText(frame, text, (20, y_pos), font, 0.5, color, 1, cv2.LINE_AA)
        
        return frame
    
    def convert_format(self, input_path: str, output_format: str, quality: str = None) -> str:
        """
        Convert video to different format
        
        Args:
            input_path: Path to input video
            output_format: Target format (mp4, mov, avi, webm)
            quality: Output quality override
            
        Returns:
            Path to converted video
        """
        try:
            output_path = Path(input_path).with_suffix(f'.{output_format}')
            quality = quality or self.output_quality
            settings = self.quality_settings[quality]
            
            if self.ffmpeg_available:
                # Use FFmpeg for format conversion
                cmd = [
                    'ffmpeg', '-y',
                    '-i', input_path,
                    '-c:v', 'libx264',
                    '-crf', str(settings['crf']),
                    '-preset', settings['preset'],
                    '-c:a', 'aac',
                    '-b:a', '128k',
                    str(output_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Format conversion failed: {result.stderr}")
                
                logger.info(f"Format converted: {input_path} -> {output_path}")
                return str(output_path)
            
            else:
                raise RuntimeError("FFmpeg required for format conversion")
                
        except Exception as e:
            logger.error(f"Format conversion failed: {str(e)}")
            return input_path
    
    def batch_process_clips(self, 
                           video_paths: List[str],
                           processing_settings: Dict,
                           progress_callback: Optional[callable] = None) -> List[str]:
        """
        Process multiple video clips with the same settings
        
        Args:
            video_paths: List of video file paths
            processing_settings: Processing configuration
            progress_callback: Optional progress callback
            
        Returns:
            List of processed video paths
        """
        processed_clips = []
        
        for i, video_path in enumerate(video_paths):
            try:
                if progress_callback:
                    progress_callback({
                        'step': 'batch_processing',
                        'current': i + 1,
                        'total': len(video_paths),
                        'progress': (i / len(video_paths)) * 100
                    })
                
                # Apply processing based on settings
                processed_path = video_path
                
                if processing_settings.get('enhance', False):
                    processed_path = self._enhance_video(processed_path, processing_settings)
                
                if processing_settings.get('convert_format'):
                    target_format = processing_settings['convert_format']
                    processed_path = self.convert_format(processed_path, target_format)
                
                processed_clips.append(processed_path)
                
            except Exception as e:
                logger.error(f"Batch processing failed for {video_path}: {str(e)}")
                processed_clips.append(video_path)  # Keep original
        
        return processed_clips
    
    def get_video_info(self, video_path: str) -> Dict:
        """Get detailed video file information"""
        
        try:
            if self.ffmpeg_available:
                # Use ffprobe for detailed info
                cmd = [
                    'ffprobe', '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    video_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    return json.loads(result.stdout)
            
            # Fallback to OpenCV
            cap = cv2.VideoCapture(video_path)
            
            info = {
                'format': {
                    'filename': video_path,
                    'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                    'size': Path(video_path).stat().st_size
                },
                'streams': [{
                    'codec_type': 'video',
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'r_frame_rate': f"{cap.get(cv2.CAP_PROP_FPS)}/1"
                }]
            }
            
            cap.release()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {str(e)}")
            return {}