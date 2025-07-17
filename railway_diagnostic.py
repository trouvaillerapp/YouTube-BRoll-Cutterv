#!/usr/bin/env python3
"""
Railway Deployment Diagnostic Script
Run this on Railway to identify deployment issues
"""

import os
import sys
import json
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_system_info():
    """Check basic system information"""
    print("🖥️  SYSTEM INFORMATION")
    print("=" * 50)
    print(f"Platform: {sys.platform}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"User: {os.getenv('USER', 'unknown')}")
    print(f"Home: {os.getenv('HOME', 'unknown')}")
    print()

def check_environment():
    """Check environment variables"""
    print("🌍 ENVIRONMENT VARIABLES")
    print("=" * 50)
    
    important_vars = [
        'PORT', 'RAILWAY_ENVIRONMENT', 'RAILWAY_PROJECT_ID',
        'YOUTUBE_COOKIES_FILE', 'YOUTUBE_COOKIES',
        'PATH', 'PYTHONPATH'
    ]
    
    for var in important_vars:
        value = os.getenv(var, 'NOT SET')
        if var in ['YOUTUBE_COOKIES']:
            value = 'SET (hidden)' if value != 'NOT SET' else 'NOT SET'
        print(f"{var}: {value}")
    print()

def check_dependencies():
    """Check Python dependencies"""
    print("📦 PYTHON DEPENDENCIES")
    print("=" * 50)
    
    critical_deps = [
        'cv2', 'numpy', 'yt_dlp', 'fastapi', 'uvicorn',
        'PIL', 'requests', 'pathlib'
    ]
    
    for dep in critical_deps:
        try:
            module = __import__(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep}: {version}")
        except ImportError as e:
            print(f"❌ {dep}: MISSING - {e}")
    print()

def check_ffmpeg():
    """Check ffmpeg availability"""
    print("🎬 FFMPEG CHECK")
    print("=" * 50)
    
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version_line = result.stdout.split('\n')[0]
            print(f"✅ FFmpeg: {version_line}")
        else:
            print(f"❌ FFmpeg error: {result.stderr}")
    except FileNotFoundError:
        print("❌ FFmpeg: NOT FOUND")
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg: TIMEOUT")
    except Exception as e:
        print(f"❌ FFmpeg: ERROR - {e}")
    print()

def check_file_system():
    """Check file system permissions and structure"""
    print("📁 FILE SYSTEM CHECK")
    print("=" * 50)
    
    # Check write permissions
    try:
        test_dir = Path('./test_write')
        test_dir.mkdir(exist_ok=True)
        test_file = test_dir / 'test.txt'
        test_file.write_text('test')
        test_file.unlink()
        test_dir.rmdir()
        print("✅ Write permissions: OK")
    except Exception as e:
        print(f"❌ Write permissions: FAILED - {e}")
    
    # Check project structure
    important_paths = [
        './backend/app.py',
        './broll_cutter/',
        './requirements.txt',
        './nixpacks.toml'
    ]
    
    for path in important_paths:
        if os.path.exists(path):
            print(f"✅ {path}: EXISTS")
        else:
            print(f"❌ {path}: MISSING")
    print()

def check_broll_cutter():
    """Check B-Roll Cutter module"""
    print("🎯 B-ROLL CUTTER MODULE")
    print("=" * 50)
    
    try:
        # Add to path if needed
        if os.path.exists('./broll_cutter'):
            sys.path.insert(0, '.')
        
        from broll_cutter import YouTubeBRollCutter
        print("✅ YouTubeBRollCutter: IMPORTABLE")
        
        # Try to initialize
        cutter = YouTubeBRollCutter(output_dir='./test_output')
        print("✅ YouTubeBRollCutter: INITIALIZABLE")
        
        # Test a simple function
        print(f"✅ Scene threshold: {cutter.scene_detector.threshold}")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
    print()

def test_video_processing():
    """Test basic video processing capabilities"""
    print("🎥 VIDEO PROCESSING TEST")
    print("=" * 50)
    
    try:
        import cv2
        
        # Test OpenCV video capabilities
        print(f"✅ OpenCV version: {cv2.__version__}")
        
        # Check available video codecs
        fourcc_codes = ['mp4v', 'XVID', 'MJPG', 'X264']
        available_codecs = []
        
        for codec in fourcc_codes:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                available_codecs.append(codec)
            except:
                pass
        
        print(f"✅ Available codecs: {available_codecs}")
        
    except Exception as e:
        print(f"❌ Video processing check failed: {e}")
    print()

def run_health_check():
    """Test the health endpoint"""
    print("🏥 HEALTH CHECK")
    print("=" * 50)
    
    try:
        import requests
        
        # Test local health endpoint
        try:
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print("✅ Health endpoint: ACCESSIBLE")
                print(f"   Status: {health_data.get('status', 'unknown')}")
                deps = health_data.get('dependencies', {})
                for dep, version in deps.items():
                    status = "✅" if version not in ['missing', 'error'] else "❌"
                    print(f"   {status} {dep}: {version}")
            else:
                print(f"❌ Health endpoint: HTTP {response.status_code}")
        except requests.exceptions.ConnectionError:
            print("⏳ Health endpoint: Server not running")
        except Exception as e:
            print(f"❌ Health endpoint: {e}")
            
    except ImportError:
        print("❌ Requests module not available")
    print()

def main():
    """Run all diagnostic checks"""
    print("🔍 RAILWAY DIAGNOSTIC REPORT")
    print("=" * 60)
    print()
    
    check_system_info()
    check_environment()
    check_dependencies()
    check_ffmpeg()
    check_file_system()
    check_broll_cutter()
    test_video_processing()
    run_health_check()
    
    print("🏁 DIAGNOSTIC COMPLETE")
    print("=" * 60)
    print("Send this output to help debug Railway deployment issues.")

if __name__ == "__main__":
    main()