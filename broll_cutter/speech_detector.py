"""
Speech detection and analysis module for extracting speaking segments
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import subprocess
import json
import tempfile
import wave
import struct

logger = logging.getLogger(__name__)

@dataclass
class SpeechSegment:
    """Represents a speech segment in video"""
    start_time: float
    end_time: float
    confidence: float
    text: Optional[str] = None
    speaker_id: Optional[str] = None
    has_face: bool = False
    speech_type: str = "dialogue"  # dialogue, monologue, quote, narration
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'confidence': self.confidence,
            'text': self.text,
            'speaker_id': self.speaker_id,
            'has_face': self.has_face,
            'speech_type': self.speech_type
        }

class SpeechDetector:
    """Detect and analyze speech in videos"""
    
    def __init__(self,
                 energy_threshold: float = 0.02,
                 silence_duration: float = 0.5,
                 min_speech_duration: float = 1.0,
                 enable_transcription: bool = False,
                 enable_face_sync: bool = True):
        """
        Initialize speech detector
        
        Args:
            energy_threshold: Audio energy threshold for speech detection
            silence_duration: Duration of silence to split segments
            min_speech_duration: Minimum duration for valid speech segment
            enable_transcription: Enable speech-to-text transcription
            enable_face_sync: Sync speech with face detection
        """
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.enable_transcription = enable_transcription
        self.enable_face_sync = enable_face_sync
        
        # Initialize face detector if needed
        if enable_face_sync:
            try:
                self.face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            except:
                logger.warning("Face detection not available")
                self.face_cascade = None
    
    def detect_speech_segments(self, video_path: str, 
                             progress_callback: Optional[callable] = None) -> List[SpeechSegment]:
        """
        Detect speech segments in video
        
        Args:
            video_path: Path to video file
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of detected speech segments
        """
        try:
            logger.info(f"Detecting speech segments in: {video_path}")
            
            # Extract audio from video
            audio_path = self._extract_audio(video_path)
            if not audio_path:
                logger.error("Failed to extract audio")
                return []
            
            # Detect speech using audio energy
            speech_segments = self._detect_speech_from_audio(audio_path)
            
            # Enhance with face detection if enabled
            if self.enable_face_sync and self.face_cascade is not None:
                speech_segments = self._sync_with_faces(video_path, speech_segments, progress_callback)
            
            # Transcribe if enabled
            if self.enable_transcription:
                speech_segments = self._transcribe_segments(audio_path, speech_segments, progress_callback)
            
            # Filter valid segments
            valid_segments = [s for s in speech_segments if s.duration >= self.min_speech_duration]
            
            logger.info(f"Detected {len(valid_segments)} speech segments")
            return valid_segments
            
        except Exception as e:
            logger.error(f"Speech detection failed: {str(e)}")
            return []
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio track from video"""
        try:
            audio_path = Path(tempfile.gettempdir()) / f"audio_{Path(video_path).stem}.wav"
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', '16000',  # 16kHz sample rate
                '-ac', '1',  # Mono
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0 and audio_path.exists():
                return str(audio_path)
            else:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Audio extraction error: {str(e)}")
            return None
    
    def _detect_speech_from_audio(self, audio_path: str) -> List[SpeechSegment]:
        """Detect speech segments from audio energy"""
        try:
            # Read WAV file
            with wave.open(audio_path, 'rb') as wav_file:
                framerate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_data = wav_file.readframes(n_frames)
            
            # Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
            audio_array = audio_array / 32768.0  # Normalize
            
            # Calculate energy in windows
            window_size = int(framerate * 0.1)  # 100ms windows
            hop_size = int(framerate * 0.05)  # 50ms hop
            
            energies = []
            for i in range(0, len(audio_array) - window_size, hop_size):
                window = audio_array[i:i + window_size]
                energy = np.sqrt(np.mean(window ** 2))
                energies.append(energy)
            
            # Find speech segments
            segments = []
            in_speech = False
            start_idx = 0
            silence_count = 0
            silence_threshold = int(self.silence_duration / 0.05)  # Convert to hop counts
            
            for i, energy in enumerate(energies):
                if energy > self.energy_threshold:
                    if not in_speech:
                        in_speech = True
                        start_idx = i
                    silence_count = 0
                else:
                    if in_speech:
                        silence_count += 1
                        if silence_count >= silence_threshold:
                            # End of speech segment
                            start_time = start_idx * 0.05
                            end_time = (i - silence_count) * 0.05
                            
                            # Validate segment times
                            if start_time is not None and end_time is not None and end_time > start_time:
                                segment = SpeechSegment(
                                    start_time=start_time,
                                    end_time=end_time,
                                    confidence=0.8
                                )
                                segments.append(segment)
                            
                            in_speech = False
                            silence_count = 0
            
            # Handle final segment
            if in_speech:
                start_time = start_idx * 0.05
                end_time = len(energies) * 0.05
                # Validate segment times
                if start_time is not None and end_time is not None and end_time > start_time:
                    segment = SpeechSegment(
                        start_time=start_time,
                        end_time=end_time,
                        confidence=0.8
                    )
                    segments.append(segment)
            
            return segments
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {str(e)}")
            return []
    
    def _sync_with_faces(self, video_path: str, 
                        speech_segments: List[SpeechSegment],
                        progress_callback: Optional[callable] = None) -> List[SpeechSegment]:
        """Sync speech segments with face detection"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Check each speech segment for faces
            for i, segment in enumerate(speech_segments):
                if progress_callback:
                    progress = (i / len(speech_segments)) * 50 + 25
                    progress_callback({
                        'progress': progress,
                        'message': f"Analyzing faces in segment {i+1}/{len(speech_segments)}"
                    })
                
                # Sample frames from the segment
                start_frame = int(segment.start_time * fps)
                end_frame = int(segment.end_time * fps)
                sample_frames = np.linspace(start_frame, end_frame, min(10, end_frame - start_frame), dtype=int)
                
                face_count = 0
                for frame_idx in sample_frames:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                        if len(faces) > 0:
                            face_count += 1
                
                # Mark segment as having face if detected in majority of samples
                if face_count > len(sample_frames) / 2:
                    segment.has_face = True
                    segment.confidence = min(1.0, segment.confidence + 0.1)
                    
                    # Determine speech type based on face presence
                    if segment.duration > 5.0:
                        segment.speech_type = "monologue"
                    else:
                        segment.speech_type = "dialogue"
            
            cap.release()
            return speech_segments
            
        except Exception as e:
            logger.error(f"Face sync failed: {str(e)}")
            return speech_segments
    
    def _transcribe_segments(self, audio_path: str,
                           speech_segments: List[SpeechSegment],
                           progress_callback: Optional[callable] = None) -> List[SpeechSegment]:
        """Transcribe speech segments (placeholder for actual implementation)"""
        # This is a placeholder - in production you would use:
        # - OpenAI Whisper
        # - Google Speech-to-Text
        # - Azure Speech Services
        # - AWS Transcribe
        
        logger.info("Transcription not implemented in MVP - would use Whisper/Google Speech API")
        
        # For now, just mark segments that might contain quotes based on duration
        for segment in speech_segments:
            if 3.0 <= segment.duration <= 15.0:
                segment.speech_type = "quote"
                segment.text = "[Speech segment - transcription available with API key]"
        
        return speech_segments
    
    def extract_quotes(self, video_path: str,
                      min_quote_duration: float = 3.0,
                      max_quote_duration: float = 20.0) -> List[SpeechSegment]:
        """
        Extract potential quote segments from video
        
        Args:
            video_path: Path to video file
            min_quote_duration: Minimum duration for a quote
            max_quote_duration: Maximum duration for a quote
            
        Returns:
            List of speech segments that could be quotes
        """
        # Detect all speech segments
        all_segments = self.detect_speech_segments(video_path)
        
        # Filter for quote-like segments
        quote_segments = []
        for segment in all_segments:
            # Skip segments with invalid duration
            if segment.duration is None or segment.duration <= 0:
                continue
            
            # Quotes are typically:
            # - Between 3-20 seconds
            # - Have clear speech (high confidence)
            # - Often have a face visible
            if (min_quote_duration <= segment.duration <= max_quote_duration and
                segment.confidence > 0.7):
                
                segment.speech_type = "quote"
                quote_segments.append(segment)
        
        logger.info(f"Found {len(quote_segments)} potential quote segments")
        return quote_segments
    
    def find_speaking_scenes(self, video_path: str,
                           speaker_visible: bool = True) -> List[Dict[str, Any]]:
        """
        Find scenes where main character is speaking
        
        Args:
            video_path: Path to video file
            speaker_visible: Only include segments where speaker's face is visible
            
        Returns:
            List of scene dictionaries with speech information
        """
        segments = self.detect_speech_segments(video_path)
        
        speaking_scenes = []
        for segment in segments:
            if speaker_visible and not segment.has_face:
                continue
            
            scene = {
                'start_time': segment.start_time,
                'end_time': segment.end_time,
                'duration': segment.duration,
                'scene_type': 'speaking',
                'confidence': segment.confidence,
                'has_face': segment.has_face,
                'speech_type': segment.speech_type,
                'suitable_for_broll': True
            }
            
            # Add metadata for B-roll extraction
            if segment.speech_type == "monologue":
                scene['description'] = "Extended speaking segment - good for testimonials"
            elif segment.speech_type == "quote":
                scene['description'] = "Quote or statement - good for highlights"
            else:
                scene['description'] = "Dialogue or conversation"
            
            speaking_scenes.append(scene)
        
        return speaking_scenes