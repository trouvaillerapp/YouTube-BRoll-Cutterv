"""
Video downloader module using yt-dlp for YouTube video extraction
"""

import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
import yt_dlp
from tqdm import tqdm

logger = logging.getLogger(__name__)

class VideoDownloader:
    """Handles YouTube video downloading with yt-dlp"""
    
    def __init__(self, output_dir: str = None, quality: str = "720p", temp_dir: str = None):
        """
        Initialize the video downloader
        
        Args:
            output_dir: Directory to save downloaded videos
            quality: Video quality (480p, 720p, 1080p, 4k)
            temp_dir: Temporary directory for downloads
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./downloads")
        self.temp_dir = Path(temp_dir) if temp_dir else Path(tempfile.gettempdir()) / "broll_cutter"
        self.quality = quality
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality mapping for yt-dlp format selection
        self.quality_formats = {
            "480p": "best[height<=480][ext=mp4]",
            "720p": "best[height<=720][ext=mp4]", 
            "1080p": "best[height<=1080][ext=mp4]",
            "4k": "best[height<=2160][ext=mp4]",
            "best": "best[ext=mp4]"
        }
        
        # Configure yt-dlp options
        self.ydl_opts = {
            'format': self.quality_formats.get(quality, self.quality_formats["720p"]),
            'outtmpl': str(self.temp_dir / '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'extractaudio': False,
            'writeinfojson': True,
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
        
        self.progress_callback = None
        
    def set_progress_callback(self, callback: Callable[[Dict], None]):
        """Set a callback function for download progress updates"""
        self.progress_callback = callback
        
    def _progress_hook(self, d):
        """Internal progress hook for yt-dlp"""
        if self.progress_callback and d['status'] == 'downloading':
            # Extract progress information
            progress_info = {
                'status': d['status'],
                'downloaded_bytes': d.get('downloaded_bytes', 0),
                'total_bytes': d.get('total_bytes', 0),
                'speed': d.get('speed', 0),
                'eta': d.get('eta', 0),
                'filename': d.get('filename', '')
            }
            
            if progress_info['total_bytes'] > 0:
                progress_info['percent'] = (progress_info['downloaded_bytes'] / progress_info['total_bytes']) * 100
            
            self.progress_callback(progress_info)
    
    def get_video_info(self, url: str) -> Dict:
        """
        Extract video information without downloading
        
        Args:
            url: YouTube video URL
            
        Returns:
            Dictionary with video metadata
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                video_info = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'uploader': info.get('uploader', 'Unknown'),
                    'upload_date': info.get('upload_date', ''),
                    'view_count': info.get('view_count', 0),
                    'description': info.get('description', ''),
                    'thumbnail': info.get('thumbnail', ''),
                    'formats': len(info.get('formats', [])),
                    'id': info.get('id', ''),
                    'webpage_url': info.get('webpage_url', url)
                }
                
                logger.info(f"Video info extracted: {video_info['title']} ({video_info['duration']}s)")
                return video_info
                
        except Exception as e:
            logger.error(f"Failed to extract video info from {url}: {str(e)}")
            raise
    
    def download_video(self, url: str, custom_filename: str = None) -> str:
        """
        Download a single YouTube video
        
        Args:
            url: YouTube video URL
            custom_filename: Optional custom filename
            
        Returns:
            Path to downloaded video file
        """
        try:
            logger.info(f"Starting download from: {url}")
            
            # Update yt-dlp options for this download
            opts = self.ydl_opts.copy()
            
            if custom_filename:
                opts['outtmpl'] = str(self.temp_dir / f"{custom_filename}.%(ext)s")
            
            # Add progress hook if callback is set
            if self.progress_callback:
                opts['progress_hooks'] = [self._progress_hook]
            
            downloaded_file = None
            
            with yt_dlp.YoutubeDL(opts) as ydl:
                # Extract info first to get the actual filename
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'video')
                
                logger.info(f"Downloading: {video_title}")
                
                # Download the video
                ydl.download([url])
                
                # Find the downloaded file
                if custom_filename:
                    expected_file = self.temp_dir / f"{custom_filename}.mp4"
                else:
                    # Look for files with the video title
                    for file in self.temp_dir.glob("*.mp4"):
                        if video_title.replace(' ', '_').lower() in file.name.lower():
                            expected_file = file
                            break
                    else:
                        # Fallback: get the most recent .mp4 file
                        mp4_files = list(self.temp_dir.glob("*.mp4"))
                        if mp4_files:
                            expected_file = max(mp4_files, key=lambda x: x.stat().st_mtime)
                        else:
                            raise FileNotFoundError("Downloaded file not found")
                
                if expected_file.exists():
                    downloaded_file = str(expected_file)
                    logger.info(f"Download completed: {downloaded_file}")
                else:
                    raise FileNotFoundError(f"Expected file not found: {expected_file}")
            
            return downloaded_file
            
        except Exception as e:
            logger.error(f"Download failed for {url}: {str(e)}")
            raise
    
    def batch_download(self, urls: List[str], max_concurrent: int = 3) -> List[str]:
        """
        Download multiple YouTube videos
        
        Args:
            urls: List of YouTube video URLs
            max_concurrent: Maximum concurrent downloads
            
        Returns:
            List of paths to downloaded video files
        """
        downloaded_files = []
        
        logger.info(f"Starting batch download of {len(urls)} videos")
        
        with tqdm(total=len(urls), desc="Downloading videos") as pbar:
            for i, url in enumerate(urls):
                try:
                    custom_name = f"batch_video_{i+1:03d}"
                    file_path = self.download_video(url, custom_name)
                    downloaded_files.append(file_path)
                    
                    pbar.set_description(f"Downloaded: {Path(file_path).name}")
                    pbar.update(1)
                    
                except Exception as e:
                    logger.error(f"Failed to download {url}: {str(e)}")
                    pbar.update(1)
                    continue
        
        logger.info(f"Batch download completed: {len(downloaded_files)}/{len(urls)} successful")
        return downloaded_files
    
    def validate_url(self, url: str) -> bool:
        """
        Validate if URL is a valid YouTube video
        
        Args:
            url: URL to validate
            
        Returns:
            True if valid YouTube URL
        """
        youtube_domains = [
            'youtube.com', 'youtu.be', 'www.youtube.com', 
            'm.youtube.com', 'music.youtube.com'
        ]
        
        try:
            # Quick validation
            if not any(domain in url for domain in youtube_domains):
                return False
            
            # Try to extract info without downloading
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                ydl.extract_info(url, download=False)
                return True
                
        except Exception:
            return False
    
    def cleanup_temp_files(self, keep_recent: int = 5):
        """
        Clean up temporary downloaded files
        
        Args:
            keep_recent: Number of recent files to keep
        """
        try:
            temp_files = list(self.temp_dir.glob("*"))
            temp_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            files_to_delete = temp_files[keep_recent:]
            
            for file in files_to_delete:
                if file.is_file():
                    file.unlink()
                    logger.info(f"Cleaned up: {file.name}")
            
            logger.info(f"Cleanup completed: {len(files_to_delete)} files removed")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")
    
    def get_available_formats(self, url: str) -> List[Dict]:
        """
        Get all available video formats for a URL
        
        Args:
            url: YouTube video URL
            
        Returns:
            List of available formats with quality info
        """
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True}) as ydl:
                info = ydl.extract_info(url, download=False)
                formats = info.get('formats', [])
                
                # Filter and organize format information
                video_formats = []
                for fmt in formats:
                    if fmt.get('vcodec') != 'none':  # Has video
                        format_info = {
                            'format_id': fmt.get('format_id'),
                            'ext': fmt.get('ext'),
                            'quality': fmt.get('format_note', ''),
                            'filesize': fmt.get('filesize', 0),
                            'fps': fmt.get('fps', 0),
                            'vcodec': fmt.get('vcodec', ''),
                            'acodec': fmt.get('acodec', ''),
                            'height': fmt.get('height', 0),
                            'width': fmt.get('width', 0)
                        }
                        video_formats.append(format_info)
                
                return video_formats
                
        except Exception as e:
            logger.error(f"Failed to get formats for {url}: {str(e)}")
            return []