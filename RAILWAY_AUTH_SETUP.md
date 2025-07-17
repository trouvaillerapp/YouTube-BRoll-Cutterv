# 🔐 Railway Authentication Setup for YouTube B-Roll Cutter

## ⚠️ Issue: YouTube Bot Detection

YouTube is blocking requests from Railway servers with the error:
> "Sign in to confirm you're not a bot"

This is because YouTube detects automated requests from cloud IP addresses.

## 🛠️ Solutions

### Solution 1: Cookie Authentication (Recommended)

#### Step 1: Export YouTube Cookies
1. **Install a browser extension**:
   - Chrome: [EditThisCookie](https://chrome.google.com/webstore/detail/editthiscookie/fngmhnnpilhplaeedifhccceomclgfbg)
   - Firefox: [Cookie-Editor](https://addons.mozilla.org/en-US/firefox/addon/cookie-editor/)

2. **Export cookies**:
   - Go to https://www.youtube.com and ensure you're logged in
   - Click the extension icon
   - Export cookies as JSON
   - Save as `youtube_cookies.json`

#### Step 2: Convert Cookies Format
Run the helper script locally:
```bash
python export_youtube_cookies.py
```
This will create `youtube_cookies.txt` in Netscape format.

#### Step 3: Deploy to Railway
1. Add the cookies file to your repository (add to .gitignore after deployment)
2. Set environment variable in Railway:
   ```
   YOUTUBE_COOKIES_FILE=/app/youtube_cookies.txt
   ```

### Solution 2: Environment Variable Cookies

Instead of a file, you can pass cookies as JSON in an environment variable:

1. Export cookies as JSON (using browser extension)
2. Minify the JSON (remove whitespace)
3. Set in Railway environment:
   ```
   YOUTUBE_COOKIES='[{"name":"VISITOR_INFO1_LIVE","value":"...","domain":".youtube.com",...},...]'
   ```

### Solution 3: Use yt-dlp Cookie Extraction

On your local machine:
```bash
# For Chrome
yt-dlp --cookies-from-browser chrome --cookies youtube_cookies.txt https://www.youtube.com

# For Firefox
yt-dlp --cookies-from-browser firefox --cookies youtube_cookies.txt https://www.youtube.com

# For Safari
yt-dlp --cookies-from-browser safari --cookies youtube_cookies.txt https://www.youtube.com
```

Then upload `youtube_cookies.txt` to Railway.

## 🚀 Quick Setup Guide

1. **Local Setup**:
   ```bash
   # Clone the repo
   git clone <your-repo>
   cd YouTube-BRoll-Cutter
   
   # Export cookies
   python export_youtube_cookies.py
   ```

2. **Railway Environment Variables**:
   ```
   YOUTUBE_COOKIES_FILE=/app/youtube_cookies.txt
   ```

3. **Test the deployment**:
   - Visit: https://web-production-f517.up.railway.app/
   - Try a YouTube URL
   - Check logs if issues persist

## 🔄 Alternative Approaches

### 1. Proxy Service
Use a residential proxy service to avoid IP-based blocking:
- Environment variable: `PROXY_URL=http://user:pass@proxy.service.com:8080`

### 2. Manual Upload Feature
If automated download fails:
- Download videos locally
- Use the upload feature in the web interface
- Process uploaded videos instead

### 3. API Services
Consider using YouTube API with proper authentication:
- Requires API key from Google Cloud Console
- Has quota limits but more reliable

## 🔍 Debugging

Check Railway logs for specific errors:
```bash
railway logs
```

Look for:
- "Using cookies file:" - Confirms cookies are loaded
- "Sign in to confirm" - Cookies not working or expired
- "HTTP Error 429" - Rate limiting

## 📝 Important Notes

1. **Cookie Expiration**: YouTube cookies expire. Refresh them monthly.
2. **Security**: Never commit cookies to your repository.
3. **Rate Limits**: Add delays between requests to avoid detection.
4. **User Agent**: The code now rotates user agents automatically.

## 🆘 Need Help?

If authentication still fails:
1. Check cookie expiration
2. Ensure you're logged into YouTube when exporting
3. Try a different extraction method
4. Consider using the manual upload feature as fallback

## 🔐 Security Best Practices

1. **Use Railway secrets** for sensitive data
2. **Rotate cookies** regularly
3. **Monitor usage** to detect issues early
4. **Have fallbacks** ready (manual upload, API, etc.)