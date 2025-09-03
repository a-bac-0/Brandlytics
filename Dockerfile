FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

Set working directory
WORKDIR /app

Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

Copy application code
COPY . .

Create necessary directories
RUN mkdir -p logs tmp static models/trained models/pretrained data/raw/images data/raw/videos

Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

Expose port
EXPOSE 8000

Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

Command to run the application with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]