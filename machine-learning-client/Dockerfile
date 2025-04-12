FROM python:3.9-slim

WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy ML model and files
COPY Client.py .
COPY cnn_model.h5 .
COPY cnn_label_encoder.pkl .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV API_PORT=5002
ENV MONGO_URI=mongodb://mongo:27017/
ENV MONGO_DB=voice_flappy_game
ENV MODEL_PATH=cnn_model.h5
ENV ENCODER_PATH=cnn_label_encoder.pkl

# Expose API port
EXPOSE 5002

# Run the API server
CMD ["python", "Client.py", "--api"]