FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Create output directory
RUN mkdir -p /app/output /app/web_output

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]