"""
Fallback methods for YouTube video access when direct download fails
"""

import logging
import os
import json
import requests
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)

class YouTubeFallback:
    """Provides fallback methods for accessing YouTube content"""
    
    def __init__(self):
        self.invidious_instances = [
            "https://yewtu.be",
            "https://invidious.snopyta.org",
            "https://invidious.kavin.rocks",
            "https://vid.puffyan.us",
            "https://invidious.namazso.eu"
        ]
        
        # Alternative video info APIs
        self.noembed_api = "https://noembed.com/embed"
        self.youtube_oembed = "https://www.youtube.com/oembed"
        
    def get_video_info_fallback(self, url: str) -> Optional[Dict]:
        """
        Get video information using fallback methods
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video information dict or None
        """
        video_id = self._extract_video_id(url)
        if not video_id:
            return None
            
        # Try oEmbed API first
        try:
            response = requests.get(
                self.youtube_oembed,
                params={"url": url, "format": "json"},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', 'Unknown'),
                    'author': data.get('author_name', 'Unknown'),
                    'thumbnail': data.get('thumbnail_url', ''),
                    'provider': 'youtube_oembed'
                }
        except Exception as e:
            logger.debug(f"oEmbed failed: {e}")
            
        # Try noembed
        try:
            response = requests.get(
                self.noembed_api,
                params={"url": url},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return {
                    'title': data.get('title', 'Unknown'),
                    'author': data.get('author_name', 'Unknown'),
                    'thumbnail': data.get('thumbnail_url', ''),
                    'provider': 'noembed'
                }
        except Exception as e:
            logger.debug(f"Noembed failed: {e}")
            
        return None
        
    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        import re
        
        patterns = [
            r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
            r'(?:embed\/)([0-9A-Za-z_-]{11})',
            r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
                
        return None
        
    def suggest_alternatives(self) -> Dict[str, str]:
        """Suggest alternative approaches for video access"""
        return {
            "browser_extension": {
                "description": "Use a browser extension to download videos",
                "tools": [
                    "Video DownloadHelper",
                    "4K Video Downloader",
                    "y2mate.com"
                ]
            },
            "cookie_authentication": {
                "description": "Export cookies from your browser",
                "instruction": "Run: python export_youtube_cookies.py"
            },
            "api_services": {
                "description": "Use third-party API services",
                "note": "May require API keys or have rate limits"
            },
            "manual_download": {
                "description": "Download videos manually and upload them",
                "instruction": "Download locally and use the file upload feature"
            }
        }