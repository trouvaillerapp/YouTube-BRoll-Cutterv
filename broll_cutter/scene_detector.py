"""
Intelligent scene detection module using computer vision and AI techniques
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)

@dataclass
class Scene:
    """Represents a detected scene in a video"""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    confidence: float
    scene_type: str = "general"
    has_faces: bool = False
    motion_level: str = "medium"  # low, medium, high
    avg_brightness: float = 0.0
    dominant_colors: List[Tuple[int, int, int]] = None
    has_speech: bool = False
    speech_confidence: float = 0.0
    is_quote: bool = False
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'confidence': self.confidence,
            'scene_type': self.scene_type,
            'has_faces': self.has_faces,
            'motion_level': self.motion_level,
            'avg_brightness': self.avg_brightness,
            'dominant_colors': self.dominant_colors or [],
            'has_speech': self.has_speech,
            'speech_confidence': self.speech_confidence,
            'is_quote': self.is_quote
        }

class SceneDetector:
    """Advanced scene detection using multiple computer vision techniques"""
    
    def __init__(self, 
                 threshold: float = 0.3,
                 min_scene_length: float = 3.0,
                 max_scene_length: float = 15.0,
                 enable_face_detection: bool = True,
                 enable_motion_analysis: bool = True,
                 enable_color_analysis: bool = True):
        """
        Initialize the scene detector
        
        Args:
            threshold: Scene change sensitivity (0.1-1.0, lower = more sensitive)
            min_scene_length: Minimum scene duration in seconds
            max_scene_length: Maximum scene duration in seconds  
            enable_face_detection: Enable face detection for scene prioritization
            enable_motion_analysis: Enable motion analysis for scene classification
            enable_color_analysis: Enable color analysis for scene characteristics
        """
        self.threshold = threshold
        self.min_scene_length = min_scene_length
        self.max_scene_length = max_scene_length
        self.enable_face_detection = enable_face_detection
        self.enable_motion_analysis = enable_motion_analysis
        self.enable_color_analysis = enable_color_analysis
        
        # Initialize face detector if enabled
        self.face_cascade = None
        if enable_face_detection:
            try:
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                logger.info("Face detection enabled")
            except Exception as e:
                logger.warning(f"Face detection initialization failed: {e}")
                self.enable_face_detection = False
        
        # Motion detection parameters
        self.motion_threshold = 2000  # Minimum motion for scene classification
        
        # Color analysis parameters
        self.color_clusters = 3  # Number of dominant colors to extract
        
    def detect_scenes(self, video_path: str, progress_callback: Optional[callable] = None) -> List[Scene]:
        """
        Detect scenes in a video using multiple techniques
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of detected scenes
        """
        try:
            logger.info(f"Starting scene detection for: {video_path}")
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            logger.info(f"Video properties: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s")
            
            # Detect scene boundaries
            scene_boundaries = self._detect_scene_boundaries(cap, progress_callback)
            
            # Convert boundaries to scenes
            scenes = self._boundaries_to_scenes(scene_boundaries, fps, total_frames)
            
            # Analyze each scene for additional properties
            scenes = self._analyze_scenes(video_path, scenes, progress_callback)
            
            # Filter scenes by duration
            filtered_scenes = self._filter_scenes_by_duration(scenes)
            
            cap.release()
            
            logger.info(f"Scene detection completed: {len(filtered_scenes)} scenes found")
            return filtered_scenes
            
        except Exception as e:
            logger.error(f"Scene detection failed: {str(e)}")
            raise
    
    def _detect_scene_boundaries(self, cap: cv2.VideoCapture, progress_callback: Optional[callable] = None) -> List[int]:
        """Detect scene boundary frames using histogram comparison"""
        
        boundaries = [0]  # Always start with frame 0
        prev_hist = None
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample every nth frame for performance
        sample_rate = max(1, total_frames // 1000)  # Sample ~1000 frames max
        
        with tqdm(total=total_frames, desc="Detecting scenes") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Sample frames for analysis
                if frame_count % sample_rate == 0:
                    # Resize frame for faster processing
                    small_frame = cv2.resize(frame, (320, 240))
                    
                    # Convert to HSV for better color analysis
                    hsv_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
                    
                    # Calculate histogram
                    hist = cv2.calcHist([hsv_frame], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
                    
                    if prev_hist is not None:
                        # Compare histograms using correlation
                        correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                        
                        # Scene change detected if correlation is below threshold
                        if correlation < (1.0 - self.threshold):
                            boundaries.append(frame_count)
                            logger.debug(f"Scene boundary detected at frame {frame_count} (correlation: {correlation:.3f})")
                    
                    prev_hist = hist
                
                # Update progress
                if progress_callback and frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 50  # 50% of total progress
                    progress_callback({'step': 'scene_detection', 'progress': progress})
                
                pbar.update(1)
        
        # Always end with the last frame
        boundaries.append(frame_count)
        
        logger.info(f"Detected {len(boundaries)-1} potential scene boundaries")
        return boundaries
    
    def _boundaries_to_scenes(self, boundaries: List[int], fps: float, total_frames: int) -> List[Scene]:
        """Convert frame boundaries to scene objects"""
        
        scenes = []
        
        for i in range(len(boundaries) - 1):
            start_frame = boundaries[i]
            end_frame = boundaries[i + 1]
            
            start_time = start_frame / fps
            end_time = end_frame / fps
            duration = end_time - start_time
            
            # Skip very short scenes
            if duration < 1.0:
                continue
            
            scene = Scene(
                start_time=start_time,
                end_time=end_time,
                start_frame=start_frame,
                end_frame=end_frame,
                confidence=0.8,  # Will be updated during analysis
                scene_type="general"
            )
            
            scenes.append(scene)
        
        return scenes
    
    def _analyze_scenes(self, video_path: str, scenes: List[Scene], progress_callback: Optional[callable] = None) -> List[Scene]:
        """Analyze scenes for additional properties like faces, motion, colors"""
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        analyzed_scenes = []
        
        for i, scene in enumerate(scenes):
            try:
                # Sample frames from the middle of the scene for analysis
                mid_frame = int((scene.start_frame + scene.end_frame) / 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                
                ret, frame = cap.read()
                if not ret:
                    analyzed_scenes.append(scene)
                    continue
                
                # Analyze this scene
                scene = self._analyze_single_scene(cap, scene, frame)
                analyzed_scenes.append(scene)
                
                # Update progress
                if progress_callback:
                    progress = 50 + (i / len(scenes)) * 30  # 30% of total progress
                    progress_callback({'step': 'scene_analysis', 'progress': progress})
                
            except Exception as e:
                logger.warning(f"Scene analysis failed for scene {i}: {e}")
                analyzed_scenes.append(scene)
        
        cap.release()
        return analyzed_scenes
    
    def _analyze_single_scene(self, cap: cv2.VideoCapture, scene: Scene, sample_frame: np.ndarray) -> Scene:
        """Analyze a single scene for various properties"""
        
        # Face detection
        if self.enable_face_detection and self.face_cascade is not None:
            scene.has_faces = self._detect_faces_in_frame(sample_frame)
        
        # Motion analysis
        if self.enable_motion_analysis:
            scene.motion_level = self._analyze_motion(cap, scene)
        
        # Color analysis
        if self.enable_color_analysis:
            scene.avg_brightness = self._calculate_brightness(sample_frame)
            scene.dominant_colors = self._extract_dominant_colors(sample_frame)
        
        # Scene type classification
        scene.scene_type = self._classify_scene_type(scene, sample_frame)
        
        # Update confidence based on analysis
        scene.confidence = self._calculate_scene_confidence(scene)
        
        return scene
    
    def _detect_faces_in_frame(self, frame: np.ndarray) -> bool:
        """Detect if frame contains faces"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            return len(faces) > 0
        except Exception:
            return False
    
    def _analyze_motion(self, cap: cv2.VideoCapture, scene: Scene) -> str:
        """Analyze motion level in a scene"""
        try:
            # Sample a few frames from the scene
            fps = cap.get(cv2.CAP_PROP_FPS)
            sample_frames = []
            
            # Get 3 sample frames from start, middle, end
            sample_positions = [
                scene.start_frame,
                int((scene.start_frame + scene.end_frame) / 2),
                scene.end_frame - 1
            ]
            
            for pos in sample_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    sample_frames.append(gray)
            
            if len(sample_frames) < 2:
                return "medium"
            
            # Calculate motion between frames
            total_motion = 0
            for i in range(len(sample_frames) - 1):
                diff = cv2.absdiff(sample_frames[i], sample_frames[i + 1])
                motion = np.sum(diff)
                total_motion += motion
            
            avg_motion = total_motion / (len(sample_frames) - 1)
            
            # Classify motion level
            if avg_motion < self.motion_threshold * 0.5:
                return "low"
            elif avg_motion > self.motion_threshold * 2:
                return "high"
            else:
                return "medium"
                
        except Exception:
            return "medium"
    
    def _calculate_brightness(self, frame: np.ndarray) -> float:
        """Calculate average brightness of frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return float(np.mean(gray))
        except Exception:
            return 128.0
    
    def _extract_dominant_colors(self, frame: np.ndarray) -> List[Tuple[int, int, int]]:
        """Extract dominant colors from frame using k-means clustering"""
        try:
            # Resize for faster processing
            small_frame = cv2.resize(frame, (150, 150))
            data = small_frame.reshape((-1, 3))
            data = np.float32(data)
            
            # K-means clustering
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(data, self.color_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convert centers to int and return as list of tuples
            centers = np.uint8(centers)
            dominant_colors = [tuple(map(int, center)) for center in centers]
            
            return dominant_colors
            
        except Exception:
            return [(128, 128, 128)]  # Default gray
    
    def _classify_scene_type(self, scene: Scene, frame: np.ndarray) -> str:
        """Classify scene type based on analysis"""
        
        # Simple classification based on properties
        if scene.has_faces:
            if scene.motion_level == "low":
                return "talking_head"
            else:
                return "person_activity"
        
        if scene.motion_level == "low" and scene.avg_brightness > 150:
            return "static_bright"
        elif scene.motion_level == "high":
            return "action"
        elif scene.avg_brightness < 80:
            return "dark_scene"
        else:
            return "general"
    
    def _calculate_scene_confidence(self, scene: Scene) -> float:
        """Calculate confidence score for scene quality"""
        
        confidence = 0.5  # Base confidence
        
        # Boost confidence for good characteristics
        if scene.duration >= 5.0:  # Good length
            confidence += 0.2
        
        if scene.has_faces:  # People are interesting
            confidence += 0.2
        
        if scene.motion_level == "medium":  # Not too static, not too chaotic
            confidence += 0.1
        
        if 100 < scene.avg_brightness < 200:  # Good exposure
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _filter_scenes_by_duration(self, scenes: List[Scene]) -> List[Scene]:
        """Filter scenes by minimum and maximum duration"""
        
        filtered = []
        
        for scene in scenes:
            if self.min_scene_length <= scene.duration <= self.max_scene_length:
                filtered.append(scene)
            elif scene.duration > self.max_scene_length:
                # Split long scenes into multiple scenes
                split_scenes = self._split_long_scene(scene)
                filtered.extend(split_scenes)
        
        return filtered
    
    def _split_long_scene(self, scene: Scene) -> List[Scene]:
        """Split a long scene into multiple shorter scenes"""
        
        scenes = []
        current_start = scene.start_time
        segment_duration = self.max_scene_length
        
        while current_start < scene.end_time:
            segment_end = min(current_start + segment_duration, scene.end_time)
            
            if segment_end - current_start >= self.min_scene_length:
                # Calculate frame numbers
                fps = (scene.end_frame - scene.start_frame) / (scene.end_time - scene.start_time)
                start_frame = int(scene.start_frame + (current_start - scene.start_time) * fps)
                end_frame = int(scene.start_frame + (segment_end - scene.start_time) * fps)
                
                split_scene = Scene(
                    start_time=current_start,
                    end_time=segment_end,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    confidence=scene.confidence * 0.9,  # Slightly lower confidence for split scenes
                    scene_type=scene.scene_type,
                    has_faces=scene.has_faces,
                    motion_level=scene.motion_level,
                    avg_brightness=scene.avg_brightness,
                    dominant_colors=scene.dominant_colors
                )
                
                scenes.append(split_scene)
            
            current_start = segment_end
        
        return scenes
    
    def save_scene_data(self, scenes: List[Scene], output_path: str):
        """Save scene detection results to JSON file"""
        
        scene_data = {
            'total_scenes': len(scenes),
            'detection_params': {
                'threshold': self.threshold,
                'min_scene_length': self.min_scene_length,
                'max_scene_length': self.max_scene_length
            },
            'scenes': [scene.to_dict() for scene in scenes]
        }
        
        with open(output_path, 'w') as f:
            json.dump(scene_data, f, indent=2)
        
        logger.info(f"Scene data saved to: {output_path}")
    
    def load_scene_data(self, input_path: str) -> List[Scene]:
        """Load scene detection results from JSON file"""
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        scenes = []
        for scene_dict in data['scenes']:
            scene = Scene(
                start_time=scene_dict['start_time'],
                end_time=scene_dict['end_time'],
                start_frame=scene_dict['start_frame'],
                end_frame=scene_dict['end_frame'],
                confidence=scene_dict['confidence'],
                scene_type=scene_dict.get('scene_type', 'general'),
                has_faces=scene_dict.get('has_faces', False),
                motion_level=scene_dict.get('motion_level', 'medium'),
                avg_brightness=scene_dict.get('avg_brightness', 128.0),
                dominant_colors=scene_dict.get('dominant_colors', [])
            )
            scenes.append(scene)
        
        logger.info(f"Loaded {len(scenes)} scenes from: {input_path}")
        return scenes