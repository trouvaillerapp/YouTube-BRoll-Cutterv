#!/usr/bin/env python3
"""
Helper script to export YouTube cookies for authentication
This helps bypass bot detection on cloud services
"""

import json
import sys
import os
from pathlib import Path

def main():
    print("🍪 YouTube Cookie Export Helper")
    print("=" * 50)
    print()
    print("This script helps you export YouTube cookies for authentication.")
    print("You'll need to be logged into YouTube in your browser.")
    print()
    print("Choose your method:")
    print("1. Use browser extension (recommended)")
    print("2. Manual cookie extraction")
    print("3. Use yt-dlp browser extraction")
    print()
    
    choice = input("Enter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\n📌 Browser Extension Method:")
        print("1. Install 'EditThisCookie' or 'Cookie-Editor' extension")
        print("2. Go to https://www.youtube.com and make sure you're logged in")
        print("3. Click the extension icon and export cookies as JSON")
        print("4. Save the file as 'youtube_cookies.json'")
        print()
        
        cookies_path = input("Enter path to cookies JSON file (or press Enter for 'youtube_cookies.json'): ").strip()
        if not cookies_path:
            cookies_path = "youtube_cookies.json"
            
        if os.path.exists(cookies_path):
            with open(cookies_path, 'r') as f:
                cookies = json.load(f)
            print(f"\n✅ Found {len(cookies)} cookies")
            
            # Save in Netscape format
            output_path = "youtube_cookies.txt"
            with open(output_path, 'w') as f:
                f.write("# Netscape HTTP Cookie File\n")
                f.write("# This is a generated file!  Do not edit.\n\n")
                
                for cookie in cookies:
                    domain = cookie.get('domain', '.youtube.com')
                    flag = 'TRUE' if domain.startswith('.') else 'FALSE'
                    path = cookie.get('path', '/')
                    secure = 'TRUE' if cookie.get('secure', False) else 'FALSE'
                    expiry = str(int(cookie.get('expirationDate', 0)))
                    name = cookie.get('name', '')
                    value = cookie.get('value', '')
                    
                    if name and value:
                        f.write(f"{domain}\t{flag}\t{path}\t{secure}\t{expiry}\t{name}\t{value}\n")
            
            print(f"✅ Cookies saved to: {output_path}")
            print("\n📌 To use on Railway/cloud services:")
            print(f"1. Set environment variable: YOUTUBE_COOKIES_FILE=/app/{output_path}")
            print(f"2. Upload {output_path} to your deployment")
            
        else:
            print(f"❌ File not found: {cookies_path}")
            
    elif choice == "2":
        print("\n📌 Manual Method:")
        print("1. Open Chrome DevTools (F12) on YouTube")
        print("2. Go to Application > Cookies > https://www.youtube.com")
        print("3. Look for these important cookies:")
        print("   - VISITOR_INFO1_LIVE")
        print("   - PREF")
        print("   - LOGIN_INFO")
        print("   - SID")
        print("   - HSID")
        print("   - SSID")
        print("   - APISID")
        print("   - SAPISID")
        print()
        print("This method is more complex. Consider using method 1 instead.")
        
    elif choice == "3":
        print("\n📌 yt-dlp Browser Extraction:")
        print("Run this command to extract cookies from your browser:")
        print()
        print("For Chrome:")
        print("  yt-dlp --cookies-from-browser chrome --cookies youtube_cookies.txt https://www.youtube.com")
        print()
        print("For Firefox:") 
        print("  yt-dlp --cookies-from-browser firefox --cookies youtube_cookies.txt https://www.youtube.com")
        print()
        print("For Safari:")
        print("  yt-dlp --cookies-from-browser safari --cookies youtube_cookies.txt https://www.youtube.com")
        print()
        print("Then set YOUTUBE_COOKIES_FILE=/app/youtube_cookies.txt in your environment")
        
    else:
        print("❌ Invalid choice")
        return
        
    print("\n📌 Environment Variable Setup:")
    print("For Railway deployment, add one of these environment variables:")
    print("- YOUTUBE_COOKIES_FILE=/app/youtube_cookies.txt (if using file)")
    print("- YOUTUBE_COOKIES='[cookie JSON array]' (if using JSON)")
    print()
    print("⚠️  Important: Keep your cookies secure and rotate them periodically!")

if __name__ == "__main__":
    main()