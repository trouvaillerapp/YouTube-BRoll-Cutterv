"""
Main YouTube B-Roll Cutter class - orchestrates all components
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union
import uuid
import json
from concurrent.futures import ThreadPoolExecutor
import tempfile

from .downloader import VideoDownloader
from .scene_detector import SceneDetector, Scene
from .watermark_remover import WatermarkRemover, WatermarkRegion
from .video_processor import VideoProcessor
from .speech_detector import SpeechDetector, SpeechSegment
from .youtube_fallback import YouTubeFallback
from .simple_extractor import SimpleExtractor
from .news_extractor import NewsExtractor

logger = logging.getLogger(__name__)

class YouTubeBRollCutter:
    """
    Main class for extracting B-roll clips from YouTube videos
    """
    
    def __init__(self, 
                 output_dir: str = "./extracted_clips",
                 temp_dir: str = None,
                 clip_duration: float = 8.0,
                 quality: str = "720p",
                 scene_threshold: float = 0.3,
                 min_scene_length: float = 3.0,
                 max_scene_length: float = 15.0,
                 remove_watermarks: bool = True,
                 enhance_video: bool = False,
                 max_clips_per_video: int = 10,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize the YouTube B-Roll Cutter
        
        Args:
            output_dir: Directory to save extracted clips
            temp_dir: Temporary directory for processing
            clip_duration: Default duration for extracted clips (seconds)
            quality: Video quality (480p, 720p, 1080p, 4k)
            scene_threshold: Scene detection sensitivity (0.1-1.0)
            min_scene_length: Minimum scene duration
            max_scene_length: Maximum scene duration
            remove_watermarks: Enable watermark removal
            enhance_video: Enable video enhancement
            max_clips_per_video: Maximum clips to extract per video
            progress_callback: Callback function for progress updates
        """
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "broll_cutter"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.clip_duration = clip_duration
        self.quality = quality
        self.remove_watermarks = remove_watermarks
        self.enhance_video = enhance_video
        self.max_clips_per_video = max_clips_per_video
        self.progress_callback = progress_callback
        
        # Initialize components
        self.downloader = VideoDownloader(
            output_dir=str(self.temp_dir / "downloads"),
            quality=quality,
            temp_dir=str(self.temp_dir)
        )
        
        self.scene_detector = SceneDetector(
            threshold=scene_threshold,
            min_scene_length=min_scene_length,
            max_scene_length=max_scene_length,
            enable_face_detection=True,
            enable_motion_analysis=True,
            enable_color_analysis=True
        )
        
        self.watermark_remover = WatermarkRemover(
            detection_threshold=0.7,
            enable_auto_detection=True,
            temporal_consistency=True
        )
        
        self.video_processor = VideoProcessor(
            output_dir=str(self.output_dir),
            temp_dir=str(self.temp_dir),
            enable_enhancement=enhance_video
        )
        
        self.speech_detector = SpeechDetector(
            energy_threshold=0.01,  # More sensitive to catch quieter speech
            silence_duration=0.3,   # Shorter silence gaps
            min_speech_duration=1.0,  # Shorter minimum duration
            enable_transcription=False,  # Can be enabled with API key
            enable_face_sync=True,
            min_silence_gap=0.5,    # Minimum silence to cut at
            max_extension_seconds=5.0  # Max extension beyond target duration
        )
        
        # Statistics
        self.simple_extractor = SimpleExtractor(
            output_dir=str(self.output_dir)
        )
        
        # Initialize news extractor
        self.news_extractor = NewsExtractor(
            openai_api_key=os.environ.get('OPENAI_API_KEY'),
            enable_transcription=True,
            importance_threshold=0.6
        )
        
        self.stats = {
            'videos_processed': 0,
            'clips_extracted': 0,
            'total_duration_processed': 0.0,
            'total_clips_duration': 0.0,
            'watermarks_removed': 0
        }
        
        logger.info("YouTube B-Roll Cutter initialized")
        self._log_settings()
    
    def _log_settings(self):
        """Log current settings"""
        settings = {
            'quality': self.quality,
            'clip_duration': self.clip_duration,
            'remove_watermarks': self.remove_watermarks,
            'enhance_video': self.enhance_video,
            'max_clips_per_video': self.max_clips_per_video,
            'scene_threshold': self.scene_detector.threshold,
            'min_scene_length': self.scene_detector.min_scene_length,
            'max_scene_length': self.scene_detector.max_scene_length
        }
        
        logger.info(f"Settings: {json.dumps(settings, indent=2)}")
    
    def extract_broll(self, video_url: str, 
                     custom_settings: Optional[Dict] = None,
                     watermark_regions: Optional[List[WatermarkRegion]] = None,
                     start_time: float = 0,
                     max_duration: Optional[float] = None) -> List[str]:
        """
        Extract B-roll clips from a single YouTube video
        
        Args:
            video_url: YouTube video URL
            custom_settings: Optional custom settings for this video
            watermark_regions: Optional predefined watermark regions
            
        Returns:
            List of paths to extracted clip files
        """
        try:
            job_id = str(uuid.uuid4())[:8]
            logger.info(f"Starting B-roll extraction for: {video_url} (Job: {job_id})")
            
            self._update_progress("Validating URL...", 0)
            
            # Validate URL
            if not self.downloader.validate_url(video_url):
                raise ValueError(f"Invalid YouTube URL: {video_url}")
            
            # Get video info
            video_info = self.downloader.get_video_info(video_url)
            logger.info(f"Processing: {video_info['title']} ({video_info['duration']}s)")
            
            self._update_progress("Downloading video...", 10)
            
            # Download video
            video_path = self.downloader.download_video(video_url, f"source_{job_id}")
            
            self._update_progress("Detecting scenes...", 30)
            
            # Detect scenes
            scenes = self.scene_detector.detect_scenes(video_path, self._scene_progress_callback)
            logger.info(f"Detected {len(scenes)} scenes")
            
            if not scenes:
                logger.warning("No scenes detected in video")
                logger.warning(f"Scene detector settings: threshold={self.scene_detector.threshold}, "
                            f"min_length={self.scene_detector.min_scene_length}, "
                            f"max_length={self.scene_detector.max_scene_length}")
                
                # Try simple time-based extraction as fallback
                logger.info("Falling back to simple time-based extraction")
                clips = self.simple_extractor.extract_time_based_clips(
                    video_path=video_path,
                    clip_duration=custom_settings.get('clip_duration', self.clip_duration) if custom_settings else self.clip_duration,
                    max_clips=self.max_clips_per_video,
                    start_time=start_time,
                    job_id=job_id
                )
                
                if clips:
                    logger.info(f"Simple extraction succeeded: {len(clips)} clips")
                    self.stats['clips_extracted'] += len(clips)
                    self.stats['videos_processed'] += 1
                    self.stats['total_duration_processed'] += video_info.get('duration', 0)
                    return clips
                else:
                    logger.error("Both scene detection and simple extraction failed")
                    return []
            
            # Log scene details for debugging
            logger.info(f"Scene detection details:")
            for i, scene in enumerate(scenes[:5]):  # Log first 5 scenes
                logger.info(f"  Scene {i+1}: {scene.start_time:.1f}s-{scene.end_time:.1f}s, "
                          f"duration={scene.duration:.1f}s, confidence={scene.confidence:.2f}, "
                          f"type={scene.scene_type}, has_faces={scene.has_faces}")
            
            # Filter scenes by start time and max duration if specified
            if start_time > 0 or max_duration:
                end_time = start_time + max_duration if max_duration else float('inf')
                filtered_scenes = []
                for scene in scenes:
                    # Include scenes that overlap with our time range
                    if scene.end_time > start_time and scene.start_time < end_time:
                        # Adjust scene times to respect start_time
                        adjusted_scene = Scene(
                            start_time=max(scene.start_time, start_time),
                            end_time=min(scene.end_time, end_time),
                            start_frame=scene.start_frame,
                            end_frame=scene.end_frame,
                            confidence=scene.confidence,
                            scene_type=scene.scene_type,
                            has_faces=scene.has_faces,
                            motion_level=scene.motion_level
                        )
                        if adjusted_scene.duration >= 3.0:  # Minimum scene length
                            filtered_scenes.append(adjusted_scene)
                scenes = filtered_scenes
                logger.info(f"Filtered to {len(scenes)} scenes in time range {start_time}s-{end_time}s")
            
            # Filter and rank scenes
            top_scenes = self._select_best_scenes(scenes, video_info)
            
            self._update_progress("Processing watermarks...", 60)
            
            # Handle watermarks
            clean_video_path = video_path
            if self.remove_watermarks:
                if watermark_regions is None:
                    watermark_regions = self.watermark_remover.detect_watermarks(video_path)
                
                if watermark_regions:
                    clean_video_path = str(self.temp_dir / f"clean_{job_id}.mp4")
                    self.watermark_remover.remove_watermarks(
                        video_path, clean_video_path, watermark_regions, self._watermark_progress_callback
                    )
                    self.stats['watermarks_removed'] += len(watermark_regions)
            
            self._update_progress("Extracting clips...", 80)
            
            # Extract clips from best scenes with proper spacing to avoid overlap
            extracted_clips = []
            used_time_ranges = []  # Track used time ranges to avoid overlap
            clip_settings = custom_settings or {}
            clip_duration = clip_settings.get('clip_duration', self.clip_duration)
            
            for i, scene in enumerate(top_scenes):
                # Check if this scene would overlap with existing clips
                scene_end_time = scene.start_time + clip_duration
                overlaps = False
                
                for used_start, used_end in used_time_ranges:
                    # Check for overlap (with 5 second buffer between clips)
                    if (scene.start_time < used_end + 5 and scene_end_time > used_start - 5):
                        overlaps = True
                        break
                
                if not overlaps:
                    clip_path = self.video_processor.extract_clip(
                        clean_video_path,
                        scene.start_time,
                        clip_duration,
                        f"{job_id}_clip_{i+1:02d}",
                        scene_info=scene.to_dict()
                    )
                    
                    if clip_path:
                        extracted_clips.append(clip_path)
                        used_time_ranges.append((scene.start_time, scene_end_time))
                        self.stats['clips_extracted'] += 1
                        self.stats['total_clips_duration'] += clip_duration
                        
                        logger.info(f"Extracted clip {len(extracted_clips)}: {scene.start_time:.1f}s-{scene_end_time:.1f}s")
                else:
                    logger.info(f"Skipped overlapping scene at {scene.start_time:.1f}s")
                
                # Stop when we have enough clips
                if len(extracted_clips) >= self.max_clips_per_video:
                    break
            
            # Update statistics
            self.stats['videos_processed'] += 1
            self.stats['total_duration_processed'] += video_info['duration']
            
            self._update_progress("Completed!", 100)
            
            logger.info(f"B-roll extraction completed: {len(extracted_clips)} clips extracted")
            return extracted_clips
            
        except Exception as e:
            logger.error(f"B-roll extraction failed for {video_url}: {str(e)}")
            raise
        finally:
            # Cleanup temporary files
            self._cleanup_temp_files(job_id)
    
    def batch_process(self, video_urls: List[str], 
                     max_workers: int = 3,
                     callback: Optional[Callable] = None) -> Dict[str, List[str]]:
        """
        Process multiple videos in parallel
        
        Args:
            video_urls: List of YouTube video URLs
            max_workers: Maximum number of concurrent workers
            callback: Optional callback for each completed video
            
        Returns:
            Dictionary mapping URLs to extracted clip paths
        """
        logger.info(f"Starting batch processing of {len(video_urls)} videos")
        
        results = {}
        
        def process_single_video(url: str) -> tuple:
            try:
                clips = self.extract_broll(url)
                if callback:
                    callback(url, clips, None)
                return url, clips
            except Exception as e:
                logger.error(f"Failed to process {url}: {str(e)}")
                if callback:
                    callback(url, [], e)
                return url, []
        
        # Process videos in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_single_video, url) for url in video_urls]
            
            for future in futures:
                try:
                    url, clips = future.result()
                    results[url] = clips
                except Exception as e:
                    logger.error(f"Batch processing error: {str(e)}")
        
        logger.info(f"Batch processing completed: {len(results)} videos processed")
        return results
    
    def _select_best_scenes(self, scenes: List[Scene], video_info: Dict) -> List[Scene]:
        """Select the best scenes for B-roll extraction with good distribution"""
        
        if not scenes:
            return []
        
        # Score scenes based on various factors
        scored_scenes = []
        video_duration = video_info.get('duration', scenes[-1].end_time)
        
        for scene in scenes:
            score = scene.confidence
            
            # Boost score for good characteristics
            if scene.has_faces:
                score += 0.2
            
            if scene.motion_level == "medium":
                score += 0.1
            elif scene.motion_level == "low" and scene.scene_type == "talking_head":
                score += 0.15  # Good for talking head clips
            
            if 5.0 <= scene.duration <= 12.0:  # Ideal duration
                score += 0.1
            
            if hasattr(scene, 'avg_brightness') and 100 < scene.avg_brightness < 200:  # Good exposure
                score += 0.05
            
            # Boost scenes that are well-distributed throughout the video
            position_in_video = scene.start_time / video_duration
            if 0.1 < position_in_video < 0.9:  # Avoid very beginning/end
                score += 0.05
            
            scored_scenes.append((score, scene))
        
        # Sort by score
        scored_scenes.sort(key=lambda x: x[0], reverse=True)
        
        # Select scenes with good temporal distribution
        selected_scenes = []
        target_clips = min(self.max_clips_per_video * 3, len(scored_scenes))  # Get more candidates
        candidates = [scene for _, scene in scored_scenes[:target_clips]]
        
        # Sort candidates by start time for distribution
        candidates.sort(key=lambda s: s.start_time)
        
        # Divide video into segments and pick best from each segment
        if len(candidates) >= self.max_clips_per_video:
            segments = self.max_clips_per_video
            segment_duration = video_duration / segments
            
            for i in range(segments):
                segment_start = i * segment_duration
                segment_end = (i + 1) * segment_duration
                
                # Find best scene in this segment
                segment_scenes = [s for s in candidates 
                                if segment_start <= s.start_time < segment_end]
                
                if segment_scenes:
                    # Pick the highest scoring scene from this segment
                    best_in_segment = max(segment_scenes, 
                                        key=lambda s: next(score for score, scene in scored_scenes if scene == s))
                    selected_scenes.append(best_in_segment)
        
        # If we don't have enough distributed scenes, fill with remaining best scenes
        while len(selected_scenes) < self.max_clips_per_video and len(selected_scenes) < len(candidates):
            for _, scene in scored_scenes:
                if scene not in selected_scenes:
                    selected_scenes.append(scene)
                    break
        
        # Sort final selection by start time
        selected_scenes.sort(key=lambda s: s.start_time)
        
        logger.info(f"Selected {len(selected_scenes)} distributed scenes from {len(scenes)} total")
        for i, scene in enumerate(selected_scenes):
            logger.info(f"  Scene {i+1}: {scene.start_time:.1f}s-{scene.end_time:.1f}s ({scene.scene_type})")
        
        return selected_scenes
    
    def _update_progress(self, message: str, progress: float):
        """Update progress via callback"""
        if self.progress_callback:
            self.progress_callback({
                'message': message,
                'progress': progress,
                'stage': 'main'
            })
    
    def _scene_progress_callback(self, progress_info: Dict):
        """Progress callback for scene detection"""
        if self.progress_callback:
            progress_info['stage'] = 'scene_detection'
            self.progress_callback(progress_info)
    
    def _watermark_progress_callback(self, progress_info: Dict):
        """Progress callback for watermark removal"""
        if self.progress_callback:
            progress_info['stage'] = 'watermark_removal'
            self.progress_callback(progress_info)
    
    def _cleanup_temp_files(self, job_id: str):
        """Clean up temporary files for a specific job"""
        try:
            temp_files = list(self.temp_dir.glob(f"*{job_id}*"))
            for file in temp_files:
                if file.is_file():
                    file.unlink()
                    logger.debug(f"Cleaned up: {file.name}")
        except Exception as e:
            logger.warning(f"Cleanup failed for job {job_id}: {str(e)}")
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        return self.stats.copy()
    
    def preview_scenes(self, video_url: str, output_dir: str = None) -> List[Dict]:
        """
        Preview detected scenes without extracting clips
        
        Args:
            video_url: YouTube video URL
            output_dir: Optional directory to save preview images
            
        Returns:
            List of scene information dictionaries
        """
        try:
            logger.info(f"Previewing scenes for: {video_url}")
            
            # Download video
            job_id = str(uuid.uuid4())[:8]
            video_path = self.downloader.download_video(video_url, f"preview_{job_id}")
            
            # Detect scenes
            scenes = self.scene_detector.detect_scenes(video_path)
            
            # Create preview images if output directory specified
            if output_dir and scenes:
                preview_dir = Path(output_dir)
                preview_dir.mkdir(parents=True, exist_ok=True)
                
                self.video_processor.create_scene_previews(
                    video_path, scenes, str(preview_dir), job_id
                )
            
            return [scene.to_dict() for scene in scenes]
            
        except Exception as e:
            logger.error(f"Scene preview failed: {str(e)}")
            raise
        finally:
            self._cleanup_temp_files(job_id)
    
    def detect_watermarks_preview(self, video_url: str, output_image: str = None) -> List[Dict]:
        """
        Preview watermark detection without processing video
        
        Args:
            video_url: YouTube video URL
            output_image: Optional path to save preview image
            
        Returns:
            List of detected watermark regions
        """
        try:
            logger.info(f"Previewing watermark detection for: {video_url}")
            
            # Download video
            job_id = str(uuid.uuid4())[:8]
            video_path = self.downloader.download_video(video_url, f"watermark_preview_{job_id}")
            
            # Detect watermarks
            regions = self.watermark_remover.detect_watermarks(video_path)
            
            # Create preview image if path specified
            if output_image and regions:
                self.watermark_remover.preview_watermark_detection(video_path, output_image)
            
            return [region.to_dict() for region in regions]
            
        except Exception as e:
            logger.error(f"Watermark detection preview failed: {str(e)}")
            raise
        finally:
            self._cleanup_temp_files(job_id)
    
    def export_project_settings(self, output_path: str):
        """Export current settings to JSON file"""
        settings = {
            'version': '1.0.0',
            'settings': {
                'clip_duration': self.clip_duration,
                'quality': self.quality,
                'remove_watermarks': self.remove_watermarks,
                'enhance_video': self.enhance_video,
                'max_clips_per_video': self.max_clips_per_video,
                'scene_detection': {
                    'threshold': self.scene_detector.threshold,
                    'min_scene_length': self.scene_detector.min_scene_length,
                    'max_scene_length': self.scene_detector.max_scene_length,
                    'enable_face_detection': self.scene_detector.enable_face_detection,
                    'enable_motion_analysis': self.scene_detector.enable_motion_analysis,
                    'enable_color_analysis': self.scene_detector.enable_color_analysis
                },
                'watermark_removal': {
                    'detection_threshold': self.watermark_remover.detection_threshold,
                    'enable_auto_detection': self.watermark_remover.enable_auto_detection,
                    'temporal_consistency': self.watermark_remover.temporal_consistency
                }
            },
            'statistics': self.stats
        }
        
        with open(output_path, 'w') as f:
            json.dump(settings, f, indent=2)
        
        logger.info(f"Project settings exported to: {output_path}")
    
    def import_project_settings(self, input_path: str):
        """Import settings from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        settings = data.get('settings', {})
        
        # Update main settings
        self.clip_duration = settings.get('clip_duration', self.clip_duration)
        self.quality = settings.get('quality', self.quality)
        self.remove_watermarks = settings.get('remove_watermarks', self.remove_watermarks)
        self.enhance_video = settings.get('enhance_video', self.enhance_video)
        self.max_clips_per_video = settings.get('max_clips_per_video', self.max_clips_per_video)
        
        # Update scene detection settings
        scene_settings = settings.get('scene_detection', {})
        self.scene_detector.threshold = scene_settings.get('threshold', self.scene_detector.threshold)
        self.scene_detector.min_scene_length = scene_settings.get('min_scene_length', self.scene_detector.min_scene_length)
        self.scene_detector.max_scene_length = scene_settings.get('max_scene_length', self.scene_detector.max_scene_length)
        
        # Update watermark removal settings
        watermark_settings = settings.get('watermark_removal', {})
        self.watermark_remover.detection_threshold = watermark_settings.get('detection_threshold', self.watermark_remover.detection_threshold)
        
        logger.info(f"Project settings imported from: {input_path}")
        self._log_settings()
    
    def extract_speaking_clips(self, video_url: str,
                              speaker_visible: bool = True,
                              quote_mode: bool = False,
                              min_duration: float = 3.0,
                              max_duration: float = 20.0,
                              custom_settings: Optional[Dict] = None) -> List[str]:
        """
        Extract clips where people are speaking or quoting
        
        Args:
            video_url: YouTube video URL
            speaker_visible: Only extract when speaker's face is visible
            quote_mode: Focus on quote-like segments (3-20 seconds)
            min_duration: Minimum clip duration
            max_duration: Maximum clip duration
            custom_settings: Optional custom settings
            
        Returns:
            List of paths to extracted speaking clips
        """
        try:
            job_id = str(uuid.uuid4())[:8]
            logger.info(f"Starting speaking clip extraction for: {video_url} (Job: {job_id})")
            
            self._update_progress("Downloading video for speech analysis...", 0)
            
            # Download video
            video_path = self.downloader.download_video(video_url, f"speech_{job_id}")
            video_info = self.downloader.get_video_info(video_url)
            
            self._update_progress("Detecting speech segments...", 20)
            
            # Detect speech segments
            if quote_mode:
                speech_segments = self.speech_detector.extract_quotes(
                    video_path, 
                    min_quote_duration=min_duration,
                    max_quote_duration=max_duration
                )
            else:
                all_segments = self.speech_detector.detect_speech_segments(video_path)
                # Filter based on criteria
                speech_segments = []
                for segment in all_segments:
                    # Skip segments with invalid duration
                    if segment.duration is None or segment.duration <= 0:
                        continue
                    if speaker_visible and not segment.has_face:
                        continue
                    if min_duration <= segment.duration <= max_duration:
                        speech_segments.append(segment)
            
            logger.info(f"Found {len(speech_segments)} speaking segments")
            
            if not speech_segments:
                logger.warning("No speaking segments found in video")
                return []
            
            self._update_progress("Extracting speaking clips...", 50)
            
            # Convert speech segments to scenes for extraction
            scenes = []
            for segment in speech_segments:
                scene = Scene(
                    start_time=segment.start_time,
                    end_time=segment.end_time,
                    start_frame=0,  # Will be calculated if needed
                    end_frame=0,
                    confidence=segment.confidence,
                    scene_type="speaking" if not quote_mode else "quote",
                    has_faces=segment.has_face,
                    has_speech=True,
                    speech_confidence=segment.confidence,
                    is_quote=quote_mode or segment.speech_type == "quote"
                )
                scenes.append(scene)
            
            # Sort by confidence and duration
            scenes.sort(key=lambda s: (s.confidence, s.duration), reverse=True)
            
            # Extract clips
            extracted_clips = []
            max_clips = custom_settings.get('max_clips_per_video', self.max_clips_per_video) if custom_settings else self.max_clips_per_video
            
            for i, scene in enumerate(scenes[:max_clips]):
                self._update_progress(f"Extracting clip {i+1}/{min(len(scenes), max_clips)}...", 50 + (i / max_clips) * 40)
                
                # Use user's clip_duration setting or scene duration, whichever is smaller
                user_clip_duration = custom_settings.get('clip_duration', self.clip_duration) if custom_settings else self.clip_duration
                clip_duration = min(user_clip_duration, scene.duration, max_duration)
                
                clip_path = self.video_processor.extract_clip(
                    video_path,
                    scene.start_time,
                    clip_duration,
                    f"{job_id}_speech_{i+1:02d}",
                    scene_info=scene.to_dict()
                )
                
                if clip_path:
                    extracted_clips.append(clip_path)
                    logger.info(f"Extracted speaking clip {i+1}: {scene.start_time:.1f}s-{scene.end_time:.1f}s ({scene.scene_type})")
            
            self._update_progress("Speaking clip extraction completed!", 100)
            
            logger.info(f"Speaking clip extraction completed: {len(extracted_clips)} clips extracted")
            return extracted_clips
            
        except Exception as e:
            logger.error(f"Speaking clip extraction failed for {video_url}: {str(e)}")
            raise
        finally:
            self._cleanup_temp_files(job_id)
    
    def extract_natural_clips(self, video_url: str,
                             target_duration: float = 15.0,
                             max_clips: int = 5,
                             custom_settings: Optional[Dict] = None) -> List[str]:
        """
        Extract clips that end at natural speech boundaries (silence gaps)
        
        Args:
            video_url: YouTube video URL
            target_duration: Target duration for each clip (can be extended)
            max_clips: Maximum number of clips to extract
            custom_settings: Optional custom settings
            
        Returns:
            List of paths to extracted natural clips
        """
        try:
            job_id = str(uuid.uuid4())[:8]
            logger.info(f"Starting natural clip extraction for: {video_url} (Job: {job_id})")
            
            self._update_progress("Downloading video for natural boundary analysis...", 0)
            
            # Download video
            video_path = self.downloader.download_video(video_url, f"natural_{job_id}")
            video_info = self.downloader.get_video_info(video_url)
            
            self._update_progress("Analyzing speech patterns and silence gaps...", 20)
            
            # Create clips with natural boundaries
            natural_clips = self.speech_detector.create_natural_clips(
                video_path=video_path,
                clip_duration=target_duration,
                max_clips=max_clips
            )
            
            if not natural_clips:
                logger.warning("No natural clips could be created")
                return []
            
            self._update_progress("Extracting clips at natural boundaries...", 50)
            
            # Extract the clips
            clip_paths = []
            for i, clip in enumerate(natural_clips):
                self._update_progress(f"Extracting clip {i+1}/{len(natural_clips)}...", 
                                    50 + (i / len(natural_clips)) * 40)
                
                # Create unique output filename
                duration_str = f"{clip['duration']:.1f}s"
                output_filename = f"{job_id}_natural_{i+1:02d}_{duration_str}.mp4"
                output_path = self.output_dir / output_filename
                
                # Extract the clip
                success = self.video_processor.extract_clip(
                    video_path=video_path,
                    start_time=clip['start_time'],
                    end_time=clip['end_time'],
                    output_path=str(output_path),
                    enhance=self.enhance_video
                )
                
                if success:
                    clip_paths.append(str(output_path))
                    logger.info(f"Extracted natural clip: {clip['start_time']:.2f}s -> {clip['end_time']:.2f}s "
                               f"(duration: {clip['duration']:.2f}s, target: {target_duration}s)")
                else:
                    logger.warning(f"Failed to extract natural clip {i+1}")
            
            self._update_progress("Extraction complete!", 100)
            
            # Update statistics
            self.stats['clips_extracted'] += len(clip_paths)
            self.stats['videos_processed'] += 1
            
            logger.info(f"Natural clip extraction completed: {len(clip_paths)} clips extracted")
            return clip_paths
            
        except Exception as e:
            logger.error(f"Natural clip extraction failed: {str(e)}")
            if self.progress_callback:
                self.progress_callback({
                    'status': 'error',
                    'progress': 0,
                    'message': f"Natural clip extraction error: {str(e)}"
                })
            return []
        finally:
            self._cleanup_temp_files(job_id)
    
    def extract_newsworthy_clips(self, video_url: str,
                                max_clips: int = 5,
                                importance_threshold: float = 0.6,
                                custom_settings: Optional[Dict] = None) -> List[str]:
        """
        Extract clips containing the most newsworthy/important content using AI analysis
        
        Args:
            video_url: YouTube video URL
            max_clips: Maximum number of clips to extract
            importance_threshold: Minimum importance score (0.0-1.0)
            custom_settings: Optional custom settings
            
        Returns:
            List of paths to extracted newsworthy clips
        """
        try:
            job_id = str(uuid.uuid4())[:8]
            logger.info(f"Starting newsworthy clip extraction for: {video_url} (Job: {job_id})")
            
            self._update_progress("Downloading video for news analysis...", 0)
            
            # Download video
            video_path = self.downloader.download_video(video_url, f"news_{job_id}")
            video_info = self.downloader.get_video_info(video_url)
            
            self._update_progress("Analyzing video for newsworthy content...", 20)
            
            # Set importance threshold
            self.news_extractor.importance_threshold = importance_threshold
            
            # Extract newsworthy clips
            newsworthy_clips = self.news_extractor.extract_newsworthy_clips(
                video_path=video_path,
                max_clips=max_clips
            )
            
            if not newsworthy_clips:
                logger.warning("No newsworthy clips found")
                return []
            
            self._update_progress("Extracting most important segments...", 50)
            
            # Extract the clips
            clip_paths = []
            for i, clip in enumerate(newsworthy_clips):
                self._update_progress(f"Extracting newsworthy clip {i+1}/{len(newsworthy_clips)}...", 
                                    50 + (i / len(newsworthy_clips)) * 40)
                
                # Create unique output filename with metadata
                category = clip['news_category']
                score = clip['importance_score']
                duration_str = f"{clip['duration']:.1f}s"
                output_filename = f"{job_id}_news_{i+1:02d}_{category}_{score:.2f}_{duration_str}.mp4"
                output_path = self.output_dir / output_filename
                
                # Extract the clip
                success = self.video_processor.extract_clip(
                    video_path=video_path,
                    start_time=clip['start_time'],
                    end_time=clip['end_time'],
                    output_path=str(output_path),
                    enhance=self.enhance_video
                )
                
                if success:
                    clip_paths.append(str(output_path))
                    logger.info(f"Extracted newsworthy clip: {clip['start_time']:.2f}s-{clip['end_time']:.2f}s "
                               f"({category}, score: {score:.2f}, summary: {clip['summary'][:50]}...)")
                else:
                    logger.warning(f"Failed to extract newsworthy clip {i+1}")
            
            self._update_progress("News extraction complete!", 100)
            
            # Update statistics
            self.stats['clips_extracted'] += len(clip_paths)
            self.stats['videos_processed'] += 1
            
            logger.info(f"Newsworthy clip extraction completed: {len(clip_paths)} clips extracted")
            
            # Log analysis summary
            if newsworthy_clips:
                categories = [clip['news_category'] for clip in newsworthy_clips]
                avg_score = sum(clip['importance_score'] for clip in newsworthy_clips) / len(newsworthy_clips)
                logger.info(f"News analysis: Categories: {set(categories)}, Avg importance: {avg_score:.2f}")
            
            return clip_paths
            
        except Exception as e:
            logger.error(f"Newsworthy clip extraction failed: {str(e)}")
            if self.progress_callback:
                self.progress_callback({
                    'status': 'error',
                    'progress': 0,
                    'message': f"News extraction error: {str(e)}"
                })
            return []
        finally:
            self._cleanup_temp_files(job_id)
    
    def analyze_video_news_content(self, video_url: str) -> Dict[str, Any]:
        """
        Analyze video for news content without extracting clips
        
        Args:
            video_url: YouTube video URL
            
        Returns:
            Comprehensive news analysis report
        """
        try:
            job_id = str(uuid.uuid4())[:8]
            logger.info(f"Analyzing news content for: {video_url}")
            
            # Download video
            video_path = self.downloader.download_video(video_url, f"analysis_{job_id}")
            
            # Analyze for news content
            analysis = self.news_extractor.analyze_video_for_news(video_path)
            
            # Add video metadata
            video_info = self.downloader.get_video_info(video_url)
            analysis['video_metadata'] = {
                'title': video_info.get('title', 'Unknown'),
                'duration': video_info.get('duration', 0),
                'uploader': video_info.get('uploader', 'Unknown')
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"News content analysis failed: {str(e)}")
            return {'error': str(e)}
        finally:
            self._cleanup_temp_files(job_id)
    
    def cleanup_all_temp_files(self):
        """Clean up all temporary files"""
        try:
            temp_files = list(self.temp_dir.glob("*"))
            for file in temp_files:
                if file.is_file():
                    file.unlink()
            
            logger.info(f"Cleaned up {len(temp_files)} temporary files")
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        self.cleanup_all_temp_files()