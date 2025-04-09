#!/usr/bin/env python3
"""
Voice Command Game Web App

A Flask web application that provides a web interface for a voice-controlled
Flappy Bird-style game and connects to a MongoDB database for score storage.
"""

import os
import time
import logging
from datetime import datetime, timezone
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_socketio import SocketIO, emit
import numpy as np
import librosa
import base64
import tempfile
from pymongo import MongoClient
from bson.json_util import dumps
from dotenv import load_dotenv
import tensorflow as tf
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# MongoDB configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo:27017/")
MONGO_DB = os.getenv("MONGO_DB", "voice_flappy_game")

# Flask and SocketIO setup
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev_secret_key")
socketio = SocketIO(app, cors_allowed_origins="*")

# ML model paths
MODEL_PATH = os.getenv("MODEL_PATH", "cnn_model.h5")
ENCODER_PATH = os.getenv("ENCODER_PATH", "cnn_label_encoder.pkl")

# Connect to MongoDB with retry logic
def connect_mongo():
    """Connect to MongoDB with retry logic."""
    max_retries = 5
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            client = MongoClient(MONGO_URI)
            db = client[MONGO_DB]
            # Test connection
            client.admin.command('ping')
            logger.info("Successfully connected to MongoDB")
            return client, db
        except Exception as e:
            logger.warning(f"MongoDB connection attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    logger.error("Failed to connect to MongoDB after multiple attempts")
    return None, None

# Connect to MongoDB
mongo_client, db = connect_mongo()
if db is not None:
    scores_collection = db.game_scores
    commands_collection = db.commands
else:
    logger.warning("Using app without MongoDB connection")
    scores_collection = None
    commands_collection = None

# Load ML model with retry logic
def load_model_func():
    """Load the voice command recognition model and label encoder."""
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH)
            
            logger.info(f"Loading label encoder from {ENCODER_PATH}")
            with open(ENCODER_PATH, "rb") as f:
                label_encoder = joblib.load(f)
                
            logger.info(f"Model loaded successfully. Classes: {label_encoder.classes_}")
            return model, label_encoder
        except Exception as e:
            logger.warning(f"Model loading attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                logger.info("Retrying in 5 seconds...")
                time.sleep(5)
    
    logger.error("Failed to load model after multiple attempts")
    return None, None

# Try to load ML model if files exist
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model, label_encoder = load_model_func()
else:
    logger.warning("Model files not found. Voice command recognition will not work.")
    model, label_encoder = None, None

# Routes
@app.route("/")
def index():
    """Render the game page."""
    return render_template("index.html")

@app.route("/scores")
def view_scores():
    """Render the scores page."""
    try:
        if scores_collection is not None:
            latest_scores = list(scores_collection.find().sort("timestamp", -1).limit(10))
            return render_template("scores.html", scores=latest_scores)
        else:
            return render_template("scores.html", scores=[], error="Database not connected")
    except Exception as e:
        logger.error(f"Error retrieving scores: {e}")
        return render_template("scores.html", scores=[], error=str(e))

@app.route("/score", methods=["POST"])
def receive_score():
    """Receive and store game scores."""
    try:
        data = request.get_json()
        score_value = data.get("score", 0)
        
        if score_value > 0 and scores_collection is not None:
            scores_collection.insert_one({
                "score": score_value,
                "timestamp": datetime.now(timezone.utc)
            })
            logger.info(f"Score saved: {score_value}")
            
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error(f"Error saving score: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/commands")
def get_commands():
    """API endpoint to retrieve recent commands."""
    try:
        if commands_collection is not None:
            recent_commands = list(commands_collection.find().sort("timestamp", -1).limit(20))
            return dumps(recent_commands), 200, {'Content-Type': 'application/json'}
        else:
            return jsonify({"error": "Database not connected"}), 503
    except Exception as e:
        logger.error(f"Error retrieving commands: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/dashboard")
def dashboard():
    """Render the dashboard page."""
    return render_template("dashboard.html")

# Socket.IO events
@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on("audio")
def handle_audio(data_url):
    """Process audio data and predict commands."""
    try:
        if not model or not label_encoder:
            emit("command", "stop")
            return
            
        # Decode base64 audio data
        header, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f_webm:
            f_webm.write(audio_bytes)
            webm_path = f_webm.name
        
        # Convert audio to WAV format (using pydub)
        from pydub import AudioSegment
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_path = webm_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        # Predict command
        command = predict_command(wav_path)
        logger.info(f"Predicted command: {command}")
        
        # Emit command to client
        emit("command", command)
        
        # Save command to database if connected
        if commands_collection is not None:
            commands_collection.insert_one({
                "command": command,
                "timestamp": datetime.now(timezone.utc),
                "processed": True
            })

        # Clean up temporary files
        try:
            os.remove(webm_path)
            os.remove(wav_path)
        except Exception as e:
            logger.warning(f"Error removing temporary files: {e}")
            
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        emit("command", "stop")  # Default command on error

def predict_command(wav_path):
    """
    Predict voice command from audio file.
    
    Args:
        wav_path: Path to WAV audio file
        
    Returns:
        command: Predicted command string
    """
    try:
        # Load audio at 16kHz.
        y, sr = librosa.load(wav_path, sr=16000)
        
        # Ensure the audio is exactly 1 second (16,000 samples)
        TARGET_SAMPLES = 16000
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
        else:
            y = y[:TARGET_SAMPLES]
        
        # Compute the mel spectrogram with 128 mel bins
        N_MELS = 128
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel_spec)
        
        # Ensure fixed time frames (44 frames)
        SPEC_FRAMES = 44
        if log_mel.shape[1] < SPEC_FRAMES:
            pad_width = SPEC_FRAMES - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
        else:
            log_mel = log_mel[:, :SPEC_FRAMES]
        
        # At this point, log_mel has shape (128, 44).
        # For channels_last input, we want shape (1, 128, 44, 1).
        input_data = np.expand_dims(log_mel, axis=0)   # Now (1, 128, 44)
        input_data = np.expand_dims(input_data, axis=-1) # Now (1, 128, 44, 1)
        
        # Make prediction
        predictions = model.predict(input_data)
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index])
        command = label_encoder.inverse_transform([predicted_index])[0]
        logger.info(f"Command: {command}, Confidence: {confidence:.4f}")
        
        # If confidence is low, return a fallback (for example, 'background')
        if confidence < 0.3:
            command = "background"
            
        return command
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "stop"


if __name__ == "__main__":
    logger.info("Starting Voice Flappy Game web server...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
