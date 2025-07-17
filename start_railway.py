#!/usr/bin/env python3
"""
Railway-specific startup script with enhanced error handling
"""

import os
import sys
import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check and report on critical dependencies"""
    logger.info("🔍 Checking dependencies...")
    
    critical_deps = ['cv2', 'numpy', 'yt_dlp', 'fastapi', 'uvicorn']
    missing_deps = []
    
    for dep in critical_deps:
        try:
            __import__(dep)
            logger.info(f"✅ {dep}: OK")
        except ImportError:
            logger.error(f"❌ {dep}: MISSING")
            missing_deps.append(dep)
    
    # Check ffmpeg
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("✅ ffmpeg: OK")
        else:
            logger.warning("⚠️ ffmpeg: Available but with errors")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.warning("⚠️ ffmpeg: NOT FOUND (will use OpenCV fallback)")
    
    return len(missing_deps) == 0

def setup_directories():
    """Create necessary directories"""
    logger.info("📁 Setting up directories...")
    
    dirs_to_create = [
        Path("web_output"),
        Path("temp"),
        Path("logs")
    ]
    
    for dir_path in dirs_to_create:
        try:
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✅ Directory: {dir_path}")
        except Exception as e:
            logger.error(f"❌ Failed to create {dir_path}: {e}")

def check_broll_module():
    """Check if B-Roll Cutter module loads correctly"""
    logger.info("🎬 Checking B-Roll Cutter module...")
    
    try:
        # Add current directory to path
        sys.path.insert(0, str(Path.cwd()))
        
        from broll_cutter import YouTubeBRollCutter
        logger.info("✅ YouTubeBRollCutter: Importable")
        
        # Test initialization
        cutter = YouTubeBRollCutter(
            output_dir="./web_output",
            scene_threshold=0.2,
            min_scene_length=2.0,
            max_scene_length=20.0
        )
        logger.info("✅ YouTubeBRollCutter: Initializable")
        return True
        
    except Exception as e:
        logger.error(f"❌ B-Roll Cutter failed: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    logger.info("🚀 Starting FastAPI server...")
    
    try:
        # Import and run the app
        from backend.app import app
        import uvicorn
        
        port = int(os.getenv('PORT', 8000))
        
        logger.info(f"🌐 Starting server on port {port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        
    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        sys.exit(1)

def main():
    """Main startup sequence"""
    logger.info("🎬 YouTube B-Roll Cutter - Railway Startup")
    logger.info("=" * 50)
    
    # Check system
    if not check_dependencies():
        logger.error("❌ Critical dependencies missing")
        sys.exit(1)
    
    # Setup environment
    setup_directories()
    
    # Check B-Roll module
    if not check_broll_module():
        logger.error("❌ B-Roll Cutter module not working")
        sys.exit(1)
    
    # Start server
    logger.info("✅ All checks passed - starting server")
    start_server()

if __name__ == "__main__":
    main()