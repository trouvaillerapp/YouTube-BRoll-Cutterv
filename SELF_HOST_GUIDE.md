# 🏠 Self-Hosting YouTube B-Roll Cutter

This guide shows you how to self-host the YouTube B-Roll Cutter to avoid cloud provider YouTube blocking.

## 🚀 Quick Start Options

### Option 1: Direct Docker (Localhost)
```bash
# 1. Build the Docker image
docker build -t youtube-broll-cutter .

# 2. Run the container
docker run -d -p 8000:8000 -v $(pwd)/output:/app/output youtube-broll-cutter

# 3. Access at http://localhost:8000
```

### Option 2: Docker Compose (Recommended)
```bash
# 1. Start the application
docker-compose up -d

# 2. Access at http://localhost:8000
```

### Option 3: With Cloudflare Tunnel (Public Access)
```bash
# 1. Install cloudflared
brew install cloudflare/cloudflare/cloudflared
# OR download from: https://github.com/cloudflare/cloudflared/releases

# 2. Start the application
docker-compose up -d

# 3. Create tunnel (gives you public URL)
cloudflared tunnel --url http://localhost:8000
```

## 📱 Alternative: Use the Script
```bash
# Run the automated setup script
./start-tunnel.sh
```

## 🔧 Manual Setup Steps

### 1. Prerequisites
- Docker installed on your machine
- (Optional) Cloudflared for public access

### 2. Clone and Setup
```bash
git clone https://github.com/trouvaillerapp/YouTube-BRoll-Cutterv.git
cd YouTube-BRoll-Cutterv
```

### 3. Configure Environment (Optional)
```bash
# Create .env file for YouTube cookies if needed
echo "YOUTUBE_COOKIES_FILE=/app/cookies.txt" > .env
```

### 4. Build and Run
```bash
# Build Docker image
docker build -t youtube-broll-cutter .

# Run with volume mounts
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/web_output:/app/web_output \
  --name youtube-broll-cutter \
  youtube-broll-cutter
```

### 5. Access Your Application
- **Local:** http://localhost:8000
- **Public (with tunnel):** URL provided by cloudflared

## 🌐 Public Access Methods

### Method 1: Cloudflare Tunnel (Free)
```bash
# Start tunnel
cloudflared tunnel --url http://localhost:8000

# You'll get a URL like: https://random-words.trycloudflare.com
```

### Method 2: Ngrok (Free tier)
```bash
# Install ngrok from https://ngrok.com
# Start tunnel
ngrok http 8000

# You'll get a URL like: https://abc123.ngrok.io
```

### Method 3: Router Port Forwarding
1. Forward port 8000 on your router to your computer
2. Use your public IP address
3. Access via http://YOUR_PUBLIC_IP:8000

## 🔒 Security Considerations

- For public access, consider adding authentication
- Use HTTPS in production (add reverse proxy)
- Keep your Docker image updated
- Monitor logs for unusual activity

## 📊 Monitoring

```bash
# View logs
docker logs youtube-broll-cutter

# View container status
docker ps

# Stop/start container
docker stop youtube-broll-cutter
docker start youtube-broll-cutter
```

## 🛠 Troubleshooting

### Container won't start
```bash
# Check logs
docker logs youtube-broll-cutter

# Check if port is already in use
lsof -i :8000
```

### YouTube access blocked
- Self-hosting uses your home IP (should work)
- If still blocked, add YouTube cookies via environment variable
- Check health endpoint: http://localhost:8000/health

### Performance issues
- Increase Docker resources in Docker Desktop
- Use SSD storage for better video processing
- Consider upgrading to paid hosting if needed

## 🎯 Why Self-Hosting Works

✅ **Uses your home IP** - Not blocked by YouTube
✅ **Full control** - Install any dependencies
✅ **Cost effective** - No hosting fees
✅ **Better performance** - Direct hardware access
✅ **Privacy** - Videos processed locally

## 📞 Support

If you encounter issues:
1. Check the logs: `docker logs youtube-broll-cutter`
2. Test health endpoint: `curl http://localhost:8000/health`
3. Verify YouTube access: `curl http://localhost:8000/test-youtube-access`

Happy self-hosting! 🎬