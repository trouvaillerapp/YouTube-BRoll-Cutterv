#!/usr/bin/env python3
"""
Test script for AI-powered newsworthy content extraction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from broll_cutter import YouTubeBRollCutter
from broll_cutter.news_extractor import NewsExtractor
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_news_extractor():
    """Test the news extractor component"""
    
    logger.info("Testing News Extractor component...")
    
    # Initialize news extractor
    extractor = NewsExtractor(
        importance_threshold=0.5,
        min_segment_duration=5.0,
        max_segment_duration=60.0
    )
    
    # Test with mock transcript
    mock_transcript = [
        {
            'start_time': 0.0,
            'end_time': 30.0,
            'text': 'Breaking news: Government announces major policy changes affecting millions of citizens across the country.',
            'confidence': 0.95
        },
        {
            'start_time': 30.0,
            'end_time': 60.0,
            'text': 'Market analysts reveal unprecedented trends in the financial sector with significant implications.',
            'confidence': 0.90
        },
        {
            'start_time': 60.0,
            'end_time': 90.0,
            'text': 'Weather update: routine temperature changes expected for the weekend.',
            'confidence': 0.85
        }
    ]
    
    # Analyze for news importance
    news_segments = extractor.analyze_news_importance(mock_transcript)
    
    logger.info(f"Found {len(news_segments)} newsworthy segments:")
    for i, segment in enumerate(news_segments):
        logger.info(f"  {i+1}. {segment.start_time:.1f}s-{segment.end_time:.1f}s "
                   f"(Score: {segment.importance_score:.2f}, Category: {segment.news_category})")
        logger.info(f"     Summary: {segment.summary}")
    
    return len(news_segments) > 0

def test_newsworthy_extraction():
    """Test full newsworthy extraction"""
    
    logger.info("Testing full newsworthy extraction...")
    
    # Initialize the cutter
    cutter = YouTubeBRollCutter(
        output_dir="./test_output",
        max_clips_per_video=3
    )
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=wi42YjWR5B4"
    
    try:
        # Extract newsworthy clips
        clips = cutter.extract_newsworthy_clips(
            video_url=test_url,
            max_clips=3,
            importance_threshold=0.5
        )
        
        if clips:
            logger.info(f"✅ Success! Extracted {len(clips)} newsworthy clips:")
            for i, clip in enumerate(clips):
                logger.info(f"  {i+1}. {clip}")
        else:
            logger.warning("No newsworthy clips extracted")
            
    except Exception as e:
        logger.error(f"❌ Newsworthy extraction failed: {str(e)}")
        return False
    
    return len(clips) > 0

def test_news_analysis():
    """Test news content analysis"""
    
    logger.info("Testing news content analysis...")
    
    # Initialize the cutter
    cutter = YouTubeBRollCutter()
    
    # Test URL
    test_url = "https://www.youtube.com/watch?v=wi42YjWR5B4"
    
    try:
        # Analyze video for news content
        analysis = cutter.analyze_video_news_content(test_url)
        
        logger.info("📊 News Analysis Results:")
        logger.info(f"  Overall Score: {analysis.get('overall_news_score', 0):.2f}")
        logger.info(f"  Primary Category: {analysis.get('primary_news_category', 'none')}")
        logger.info(f"  Newsworthy Segments: {analysis.get('newsworthy_segments', 0)}")
        logger.info(f"  Breaking News: {analysis.get('has_breaking_news', False)}")
        
        if analysis.get('clips'):
            logger.info(f"  Top clips:")
            for i, clip in enumerate(analysis['clips'][:3]):
                logger.info(f"    {i+1}. {clip['news_category']} (Score: {clip['importance_score']:.2f})")
                logger.info(f"       {clip['summary']}")
        
        return analysis.get('newsworthy_segments', 0) > 0
        
    except Exception as e:
        logger.error(f"❌ News analysis failed: {str(e)}")
        return False

def demo_api_usage():
    """Demo API usage for newsworthy extraction"""
    
    logger.info("📚 API Usage Examples:")
    print("\n" + "="*60)
    print("API USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. 📰 Extract Newsworthy Clips:")
    print("POST /api/v1/extract-newsworthy")
    print(json.dumps({
        "url": "https://youtube.com/watch?v=YOUR_VIDEO_ID",
        "settings": {
            "importance_threshold": 0.6,
            "max_clips": 5
        }
    }, indent=2))
    
    print("\n2. 🔍 Analyze News Content:")
    print("POST /api/v1/analyze-news")
    print(json.dumps({
        "url": "https://youtube.com/watch?v=YOUR_VIDEO_ID"
    }, indent=2))
    
    print("\n3. 🌐 Web Interface:")
    print("- Select '📰 AI News Analysis' mode")
    print("- System will find most important/newsworthy segments")
    print("- Clips named with category and importance score")
    
    print("\n4. 🎯 What it analyzes:")
    print("- Breaking news indicators (urgent, breaking, developing)")
    print("- News categories (politics, business, technology, etc.)")
    print("- Importance keywords (exclusive, major, significant)")
    print("- Key phrases and named entities")
    print("- Content length and substance")

if __name__ == "__main__":
    print("📰 Testing AI-Powered Newsworthy Content Extraction")
    print("=" * 60)
    
    # Test 1: News extractor component
    print("\n1. Testing News Extractor Component...")
    if test_news_extractor():
        print("✅ News extractor component test passed")
    else:
        print("❌ News extractor component test failed")
    
    # Test 2: News analysis
    print("\n2. Testing News Content Analysis...")
    if test_news_analysis():
        print("✅ News analysis test passed")
    else:
        print("❌ News analysis test failed")
    
    # Test 3: Full newsworthy extraction
    print("\n3. Testing Full Newsworthy Extraction...")
    if test_newsworthy_extraction():
        print("✅ Newsworthy extraction test passed")
    else:
        print("❌ Newsworthy extraction test failed")
    
    # Demo API usage
    print("\n4. API Usage Examples:")
    demo_api_usage()
    
    print("\n" + "="*60)
    print("🎬 AI News Extraction Feature is Ready!")
    print("="*60)
    print("\n🚀 Key Features:")
    print("- Intelligent transcript analysis")
    print("- News importance scoring (0.0-1.0)")
    print("- Category classification (politics, business, tech, etc.)")
    print("- Key phrase extraction")
    print("- Variable clip lengths based on content importance")
    print("- No fixed time constraints - content-driven extraction")
    
    print("\n🎯 Perfect for:")
    print("- News content creators")
    print("- Journalists extracting key moments")
    print("- Social media content from news videos")
    print("- Automated highlight reels")
    print("- Content curation and summarization")