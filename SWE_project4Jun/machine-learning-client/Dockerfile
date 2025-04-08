FROM python:3.11-slim

# Install system dependencies for audio and scientific computing
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libasound2-dev \
        ffmpeg \
        portaudio19-dev \
        libsndfile1 \
        && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all files into container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the main game script
CMD ["python", "VoiceFlappy.py"]


ENV PIP_DEFAULT_TIMEOUT=100