FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (incl. ffmpeg for PyAV)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgtk-3-0 \
    libgl1-mesa-glx \
    wget \
    curl \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy environment file (optional; envs can also be provided via docker-compose)
COPY .env .

# Create necessary directories
RUN mkdir -p logs tmp static models/trained models/pretrained data/raw/images data/raw/videos

# Download YOLOv8 pretrained weights (cached at build)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Command to run the application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
