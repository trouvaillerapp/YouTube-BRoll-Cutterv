#!/usr/bin/env python3
"""
Quick start script for YouTube B-Roll Cutter Web Server
"""

import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    try:
        import fastapi
        import uvicorn
        import cv2
        print("✅ All required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False

def install_requirements():
    """Install requirements"""
    print("📦 Installing web server requirements...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements-web.txt"
        ])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def start_server():
    """Start the web server"""
    print("🚀 Starting YouTube B-Roll Cutter Web Server...")
    print("📱 Opening web interface at: http://localhost:8000")
    print("🔧 API documentation at: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Change to backend directory
        backend_dir = Path(__file__).parent / "backend"
        os.chdir(backend_dir)
        
        # Start the server
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n⏹️  Server stopped by user")
    except Exception as e:
        print(f"❌ Server failed to start: {e}")

def main():
    """Main function"""
    print("🎬 YouTube B-Roll Cutter - Web Server Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("broll_cutter").exists():
        print("❌ Please run this script from the YouTube-BRoll-Cutter directory")
        return
    
    # Create necessary directories
    Path("web_output").mkdir(exist_ok=True)
    Path("backend").mkdir(exist_ok=True)
    
    # Check requirements
    if not check_requirements():
        print("\n📦 Installing missing requirements...")
        if not install_requirements():
            print("❌ Failed to install requirements. Please install manually:")
            print("   pip install -r requirements-web.txt")
            return
    
    # Start server
    print("\n🌐 Starting web server...")
    start_server()

if __name__ == "__main__":
    main()