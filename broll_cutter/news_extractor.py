"""
AI-powered news extraction module for identifying important/newsworthy content
"""

import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import subprocess
import tempfile
import requests
import time

logger = logging.getLogger(__name__)

@dataclass
class NewsSegment:
    """Represents a newsworthy segment in video"""
    start_time: float
    end_time: float
    transcript: str
    importance_score: float
    news_category: str
    key_phrases: List[str]
    summary: str
    confidence: float
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time
    
    def to_dict(self) -> Dict:
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'transcript': self.transcript,
            'importance_score': self.importance_score,
            'news_category': self.news_category,
            'key_phrases': self.key_phrases,
            'summary': self.summary,
            'confidence': self.confidence
        }

class NewsExtractor:
    """AI-powered news extraction for finding important video segments"""
    
    def __init__(self, 
                 openai_api_key: Optional[str] = None,
                 enable_transcription: bool = True,
                 min_segment_duration: float = 5.0,
                 max_segment_duration: float = 60.0,
                 importance_threshold: float = 0.6):
        """
        Initialize news extractor
        
        Args:
            openai_api_key: OpenAI API key for AI analysis
            enable_transcription: Enable video transcription
            min_segment_duration: Minimum segment duration
            max_segment_duration: Maximum segment duration
            importance_threshold: Minimum importance score to include
        """
        self.openai_api_key = openai_api_key
        self.enable_transcription = enable_transcription
        self.min_segment_duration = min_segment_duration
        self.max_segment_duration = max_segment_duration
        self.importance_threshold = importance_threshold
        
        # News keywords and patterns
        self.news_keywords = {
            'breaking': ['breaking', 'urgent', 'alert', 'developing', 'just in'],
            'politics': ['government', 'election', 'political', 'policy', 'legislation', 'congress', 'senate'],
            'business': ['earnings', 'stock', 'market', 'economy', 'company', 'ceo', 'financial'],
            'technology': ['ai', 'artificial intelligence', 'tech', 'innovation', 'startup', 'digital'],
            'health': ['health', 'medical', 'vaccine', 'disease', 'treatment', 'doctor'],
            'crime': ['police', 'arrest', 'investigation', 'crime', 'criminal', 'court'],
            'weather': ['storm', 'hurricane', 'weather', 'forecast', 'temperature', 'climate'],
            'international': ['international', 'global', 'world', 'country', 'nation', 'foreign'],
            'sports': ['sports', 'game', 'team', 'player', 'championship', 'win', 'loss'],
            'entertainment': ['celebrity', 'movie', 'music', 'entertainment', 'actor', 'singer']
        }
        
        # Importance indicators
        self.importance_indicators = {
            'high': ['exclusive', 'first time', 'never before', 'unprecedented', 'shocking', 'major', 'significant'],
            'medium': ['important', 'notable', 'interesting', 'reveals', 'confirms', 'announces'],
            'low': ['routine', 'regular', 'standard', 'typical', 'usual', 'normal']
        }
        
        logger.info("News extractor initialized")
    
    def extract_transcript(self, video_path: str) -> List[Dict[str, Any]]:
        """
        Extract transcript from video with timestamps
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of transcript segments with timestamps
        """
        try:
            # Extract audio first
            audio_path = self._extract_audio(video_path)
            if not audio_path:
                return []
            
            # Try multiple transcription methods
            transcript_segments = []
            
            # Method 1: OpenAI Whisper API (most accurate)
            if self.openai_api_key:
                transcript_segments = self._transcribe_with_openai(audio_path)
            
            # Method 2: Fallback to simple speech detection with mock transcription
            if not transcript_segments:
                transcript_segments = self._mock_transcribe_with_speech_detection(video_path)
            
            logger.info(f"Extracted {len(transcript_segments)} transcript segments")
            return transcript_segments
            
        except Exception as e:
            logger.error(f"Transcript extraction failed: {str(e)}")
            return []
    
    def _extract_audio(self, video_path: str) -> Optional[str]:
        """Extract audio from video"""
        try:
            audio_path = Path(tempfile.gettempdir()) / f"audio_{Path(video_path).stem}.wav"
            
            cmd = [
                'ffmpeg', '-y',
                '-i', video_path,
                '-vn',  # No video
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # 16kHz
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
    
    def _transcribe_with_openai(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio using OpenAI Whisper API"""
        try:
            # This would use OpenAI's Whisper API
            # For now, return mock data structure
            logger.info("OpenAI Whisper transcription would be used here")
            
            # Mock implementation - in production, use actual OpenAI API
            return [
                {
                    'start_time': 0.0,
                    'end_time': 30.0,
                    'text': '[OpenAI Whisper transcription would appear here]',
                    'confidence': 0.95
                }
            ]
            
        except Exception as e:
            logger.error(f"OpenAI transcription failed: {str(e)}")
            return []
    
    def _mock_transcribe_with_speech_detection(self, video_path: str) -> List[Dict[str, Any]]:
        """Mock transcription using speech detection for demo purposes"""
        try:
            from .speech_detector import SpeechDetector
            
            # Use existing speech detector
            detector = SpeechDetector()
            segments = detector.detect_speech_segments(video_path)
            
            # Create mock transcript segments
            mock_texts = [
                "In breaking news today, major developments in the technology sector",
                "Government officials announce new policy changes affecting millions",
                "Market analysts reveal unprecedented trends in the financial sector",
                "Exclusive investigation uncovers significant findings",
                "Expert commentary on the latest political developments"
            ]
            
            transcript_segments = []
            for i, segment in enumerate(segments):
                transcript_segments.append({
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'text': mock_texts[i % len(mock_texts)],
                    'confidence': 0.8
                })
            
            return transcript_segments
            
        except Exception as e:
            logger.error(f"Mock transcription failed: {str(e)}")
            return []
    
    def analyze_news_importance(self, transcript_segments: List[Dict[str, Any]]) -> List[NewsSegment]:
        """
        Analyze transcript segments for news importance
        
        Args:
            transcript_segments: List of transcript segments
            
        Returns:
            List of NewsSegment objects with importance scores
        """
        try:
            news_segments = []
            
            for segment in transcript_segments:
                text = segment['text'].lower()
                
                # Calculate importance score
                importance_score = self._calculate_importance_score(text)
                
                # Determine news category
                news_category = self._classify_news_category(text)
                
                # Extract key phrases
                key_phrases = self._extract_key_phrases(text)
                
                # Generate summary
                summary = self._generate_summary(segment['text'])
                
                # Only include segments above threshold
                if importance_score >= self.importance_threshold:
                    news_segment = NewsSegment(
                        start_time=segment['start_time'],
                        end_time=segment['end_time'],
                        transcript=segment['text'],
                        importance_score=importance_score,
                        news_category=news_category,
                        key_phrases=key_phrases,
                        summary=summary,
                        confidence=segment.get('confidence', 0.8)
                    )
                    news_segments.append(news_segment)
            
            # Sort by importance score
            news_segments.sort(key=lambda x: x.importance_score, reverse=True)
            
            logger.info(f"Found {len(news_segments)} newsworthy segments")
            return news_segments
            
        except Exception as e:
            logger.error(f"News importance analysis failed: {str(e)}")
            return []
    
    def _calculate_importance_score(self, text: str) -> float:
        """Calculate importance score for text"""
        score = 0.0
        word_count = len(text.split())
        
        # Check for breaking news indicators
        for keyword in self.importance_indicators['high']:
            if keyword in text:
                score += 0.3
        
        for keyword in self.importance_indicators['medium']:
            if keyword in text:
                score += 0.2
        
        # Penalty for low importance indicators
        for keyword in self.importance_indicators['low']:
            if keyword in text:
                score -= 0.1
        
        # Check for news categories
        category_score = 0.0
        for category, keywords in self.news_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    category_score += 0.1
        
        score += min(category_score, 0.4)  # Cap category score
        
        # Length factor (prefer substantial content)
        if word_count > 20:
            score += 0.1
        if word_count > 50:
            score += 0.1
        
        # Normalize score
        return min(max(score, 0.0), 1.0)
    
    def _classify_news_category(self, text: str) -> str:
        """Classify text into news category"""
        category_scores = {}
        
        for category, keywords in self.news_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return 'general'
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction
        phrases = []
        
        # Look for capitalized words/phrases
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        phrases.extend(capitalized)
        
        # Look for quoted text
        quoted = re.findall(r'"([^"]*)"', text)
        phrases.extend(quoted)
        
        # Look for numbers with context
        numbers = re.findall(r'\b\d+(?:\.\d+)?(?:\s*(?:percent|million|billion|thousand))\b', text)
        phrases.extend(numbers)
        
        return list(set(phrases))[:5]  # Top 5 unique phrases
    
    def _generate_summary(self, text: str) -> str:
        """Generate a summary of the text"""
        # Simple summarization - first sentence or truncated version
        sentences = text.split('. ')
        if sentences:
            summary = sentences[0]
            if len(summary) > 100:
                summary = summary[:97] + "..."
            return summary
        return text[:100] + "..." if len(text) > 100 else text
    
    def extract_newsworthy_clips(self, video_path: str, max_clips: int = 5) -> List[Dict[str, Any]]:
        """
        Extract the most newsworthy clips from video
        
        Args:
            video_path: Path to video file
            max_clips: Maximum number of clips to extract
            
        Returns:
            List of newsworthy clip dictionaries
        """
        try:
            logger.info(f"Extracting newsworthy clips from: {video_path}")
            
            # Step 1: Extract transcript
            transcript_segments = self.extract_transcript(video_path)
            if not transcript_segments:
                logger.warning("No transcript available")
                return []
            
            # Step 2: Analyze for news importance
            news_segments = self.analyze_news_importance(transcript_segments)
            
            # Step 3: Create clips from newsworthy segments
            clips = []
            for i, segment in enumerate(news_segments[:max_clips]):
                clip = {
                    'id': f'news_clip_{i+1}',
                    'start_time': segment.start_time,
                    'end_time': segment.end_time,
                    'duration': segment.duration,
                    'importance_score': segment.importance_score,
                    'news_category': segment.news_category,
                    'summary': segment.summary,
                    'key_phrases': segment.key_phrases,
                    'transcript': segment.transcript,
                    'confidence': segment.confidence,
                    'description': f"Newsworthy {segment.news_category} segment (Score: {segment.importance_score:.2f})"
                }
                clips.append(clip)
                
                logger.info(f"News clip {i+1}: {segment.start_time:.2f}s-{segment.end_time:.2f}s "
                           f"({segment.news_category}, score: {segment.importance_score:.2f})")
            
            return clips
            
        except Exception as e:
            logger.error(f"Newsworthy clip extraction failed: {str(e)}")
            return []
    
    def analyze_video_for_news(self, video_path: str) -> Dict[str, Any]:
        """
        Comprehensive news analysis of video
        
        Returns:
            Analysis report with newsworthy segments and metadata
        """
        try:
            # Extract newsworthy clips
            clips = self.extract_newsworthy_clips(video_path)
            
            # Calculate overall news score
            if clips:
                overall_score = sum(clip['importance_score'] for clip in clips) / len(clips)
                top_category = max(set(clip['news_category'] for clip in clips), 
                                 key=lambda x: sum(1 for clip in clips if clip['news_category'] == x))
            else:
                overall_score = 0.0
                top_category = 'none'
            
            # Generate report
            report = {
                'overall_news_score': overall_score,
                'primary_news_category': top_category,
                'newsworthy_segments': len(clips),
                'total_newsworthy_duration': sum(clip['duration'] for clip in clips),
                'clips': clips,
                'has_breaking_news': any(clip['news_category'] == 'breaking' for clip in clips),
                'key_topics': list(set(phrase for clip in clips for phrase in clip['key_phrases']))
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Video news analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def get_ai_enhanced_analysis(self, text: str) -> Dict[str, Any]:
        """
        Use AI to enhance news analysis (placeholder for OpenAI integration)
        
        Args:
            text: Text to analyze
            
        Returns:
            Enhanced analysis with AI insights
        """
        # This would integrate with OpenAI GPT for enhanced analysis
        # For now, return enhanced mock analysis
        
        analysis = {
            'sentiment': 'neutral',
            'urgency_level': 'medium',
            'target_audience': 'general',
            'news_angle': 'informational',
            'viral_potential': 0.6,
            'credibility_score': 0.8,
            'emotional_impact': 0.7
        }
        
        return analysis