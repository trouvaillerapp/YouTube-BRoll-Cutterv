#!/usr/bin/env python3
"""
Test script for N8N API integration
"""

import requests
import json
import time
import sys

# Configuration
API_BASE_URL = "http://localhost:8000/api/v1"
TEST_VIDEO_URL = "https://youtube.com/watch?v=dQw4w9WgXcQ"  # Rick Roll for testing

def test_api_endpoints():
    """Test all N8N API endpoints"""
    
    print("🧪 Testing YouTube B-Roll Cutter N8N API Integration")
    print("=" * 60)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"📊 Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: API Documentation
    print("\n2️⃣ Testing API Documentation...")
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ API docs accessible")
            docs = response.json()
            print(f"📚 API Title: {docs.get('title')}")
            print(f"🔢 Version: {docs.get('version')}")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs error: {e}")
    
    # Test 3: Synchronous Processing
    print("\n3️⃣ Testing Synchronous Processing...")
    try:
        sync_payload = {
            "url": TEST_VIDEO_URL,
            "settings": {
                "clip_duration": 5.0,
                "quality": "480p",
                "max_clips_per_video": 2,
                "start_time": 30,
                "max_duration": 60
            }
        }
        
        print(f"🚀 Processing: {TEST_VIDEO_URL}")
        print(f"⚙️ Settings: {json.dumps(sync_payload['settings'], indent=2)}")
        
        response = requests.post(
            f"{API_BASE_URL}/process-sync",
            json=sync_payload,
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Synchronous processing completed")
            print(f"📊 Status: {result.get('status')}")
            print(f"🎬 Clips count: {result.get('clips_count')}")
            print(f"📁 Clips: {result.get('clips')}")
            print(f"🔗 Download links: {result.get('download_links')}")
            
            # Test downloading one clip
            if result.get('clips'):
                print(f"\n   📥 Testing download of first clip...")
                clip_name = result['clips'][0]
                download_url = f"{API_BASE_URL.replace('/api/v1', '')}/download/{clip_name}"
                
                download_response = requests.get(download_url, stream=True)
                if download_response.status_code == 200:
                    file_size = len(download_response.content)
                    print(f"   ✅ Download successful: {clip_name} ({file_size} bytes)")
                else:
                    print(f"   ❌ Download failed: {download_response.status_code}")
            
        else:
            print(f"❌ Synchronous processing failed: {response.status_code}")
            print(f"📝 Error: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ Synchronous processing timed out (this is normal for longer videos)")
    except Exception as e:
        print(f"❌ Synchronous processing error: {e}")
    
    # Test 4: Asynchronous Processing
    print("\n4️⃣ Testing Asynchronous Processing...")
    try:
        async_payload = {
            "url": TEST_VIDEO_URL,
            "settings": {
                "clip_duration": 6.0,
                "quality": "720p",
                "max_clips_per_video": 3,
                "start_time": 60,
                "max_duration": 90
            }
        }
        
        print(f"🚀 Starting async processing: {TEST_VIDEO_URL}")
        
        response = requests.post(
            f"{API_BASE_URL}/extract-broll",
            json=async_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            
            print("✅ Async processing started")
            print(f"🆔 Job ID: {job_id}")
            print(f"📊 Status: {result.get('status')}")
            print(f"⏱️ Estimated duration: {result.get('estimated_duration')}")
            print(f"🔗 Status URL: {result.get('status_url')}")
            
            # Poll job status
            print(f"\n   📊 Polling job status...")
            max_polls = 30  # Maximum 5 minutes of polling
            poll_count = 0
            
            while poll_count < max_polls:
                time.sleep(10)  # Wait 10 seconds between polls
                poll_count += 1
                
                status_response = requests.get(f"{API_BASE_URL}/status/{job_id}")
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    progress = status.get('progress', 0)
                    message = status.get('message', '')
                    job_status = status.get('status', '')
                    
                    print(f"   🔄 Poll {poll_count}: {job_status} - {progress}% - {message}")
                    
                    if job_status == 'completed':
                        print("   ✅ Job completed successfully!")
                        print(f"   🎬 Clips count: {status.get('clips_count')}")
                        print(f"   📁 Clips: {status.get('clips')}")
                        print(f"   ⏱️ Elapsed time: {status.get('elapsed_time_seconds')}s")
                        
                        # Test bulk download
                        if status.get('clips'):
                            print(f"   📥 Testing bulk download...")
                            bulk_url = f"{API_BASE_URL}/download/{job_id}"
                            
                            bulk_response = requests.get(bulk_url)
                            if bulk_response.status_code == 200:
                                zip_size = len(bulk_response.content)
                                print(f"   ✅ Bulk download successful: {zip_size} bytes (ZIP)")
                            else:
                                print(f"   ❌ Bulk download failed: {bulk_response.status_code}")
                        
                        break
                    elif job_status == 'failed':
                        print(f"   ❌ Job failed: {status.get('error')}")
                        break
                else:
                    print(f"   ❌ Status check failed: {status_response.status_code}")
                    break
            
            if poll_count >= max_polls:
                print("   ⏰ Job polling timed out")
                
        else:
            print(f"❌ Async processing failed: {response.status_code}")
            print(f"📝 Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Async processing error: {e}")
    
    # Test 5: Batch Processing
    print("\n5️⃣ Testing Batch Processing...")
    try:
        batch_payload = {
            "urls": [
                TEST_VIDEO_URL,
                "https://youtube.com/watch?v=dQw4w9WgXcQ"  # Same video twice for testing
            ],
            "settings": {
                "clip_duration": 5.0,
                "quality": "480p",
                "max_clips_per_video": 2
            }
        }
        
        print(f"🚀 Starting batch processing: {len(batch_payload['urls'])} videos")
        
        response = requests.post(
            f"{API_BASE_URL}/batch-extract",
            json=batch_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result.get('job_id')
            
            print("✅ Batch processing started")
            print(f"🆔 Job ID: {job_id}")
            print(f"📊 Status: {result.get('status')}")
            print(f"⏱️ Estimated duration: {result.get('estimated_duration')}")
            
            # Check initial status
            time.sleep(5)
            status_response = requests.get(f"{API_BASE_URL}/status/{job_id}")
            
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"   📊 Initial status: {status.get('status')} - {status.get('message')}")
            
        else:
            print(f"❌ Batch processing failed: {response.status_code}")
            print(f"📝 Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Batch processing error: {e}")
    
    print("\n🎯 API Integration Test Complete!")
    print("=" * 60)
    print("ℹ️  For complete N8N integration, see N8N_INTEGRATION.md")
    print("🌐 API Documentation: http://localhost:8000/api/v1/docs")
    print("🖥️  Web Interface: http://localhost:8000")

def test_webhook_simulation():
    """Simulate webhook functionality"""
    print("\n📡 Webhook Simulation Test")
    print("-" * 30)
    
    # This would typically be your n8n webhook URL
    webhook_url = "https://httpbin.org/post"  # Test webhook endpoint
    
    webhook_payload = {
        "url": TEST_VIDEO_URL,
        "settings": {
            "clip_duration": 8.0,
            "quality": "720p",
            "max_clips_per_video": 3
        },
        "webhook_url": webhook_url
    }
    
    print(f"🚀 Testing webhook notification to: {webhook_url}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/extract-broll",
            json=webhook_payload
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Webhook-enabled job started")
            print(f"🆔 Job ID: {result.get('job_id')}")
            print("📡 Webhook will be called when processing completes")
        else:
            print(f"❌ Webhook test failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Webhook test error: {e}")

if __name__ == "__main__":
    print("🔧 YouTube B-Roll Cutter N8N API Integration Test")
    print("Make sure the server is running: python backend/app.py")
    print()
    
    # Check if server is running
    try:
        response = requests.get(f"{API_BASE_URL.replace('/api/v1', '')}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding. Please start the server first.")
            sys.exit(1)
    except:
        print("❌ Cannot connect to server. Please start the server first:")
        print("   python backend/app.py")
        sys.exit(1)
    
    # Run tests
    test_api_endpoints()
    test_webhook_simulation()