FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories if they don't exist
RUN mkdir -p /app/static
RUN mkdir -p /app/templates
RUN mkdir -p /app/tests

# Make app.py executable
RUN chmod +x app.py

# Expose the port Flask runs on
EXPOSE 5001

# Run the Flask application
CMD ["python", "app.py"]