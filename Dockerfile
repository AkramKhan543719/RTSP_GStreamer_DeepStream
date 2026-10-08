FROM python:3.9-slim

# --------------------------------------------------
# Python configuration
# --------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

# --------------------------------------------------
# Working directory
# --------------------------------------------------

WORKDIR /app

# --------------------------------------------------
# System dependencies
# --------------------------------------------------

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# Python dependencies
# --------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --------------------------------------------------
# Copy project source
# --------------------------------------------------

COPY . .

# --------------------------------------------------
# Create runtime directories
# --------------------------------------------------

RUN mkdir -p \
    /app/outputs \
    /app/experiments \
    /app/models \
    /app/test_videos \
    /tmp/Ultralytics

# --------------------------------------------------
# Default container command
# --------------------------------------------------

CMD ["tail", "-f", "/dev/null"]