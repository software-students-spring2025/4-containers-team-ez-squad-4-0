#!/usr/bin/env python3
"""
Voice Command Game Web App

A Flask web application that provides a web interface for a voice-controlled
Flappy Bird-style game and connects to a MongoDB database for score storage.--
"""

# Standard library imports
import os
import time
import logging
import base64
import tempfile
from datetime import datetime, timezone

# Third-party imports
import numpy as np
import librosa
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from pymongo import MongoClient
from bson.json_util import dumps
from dotenv import load_dotenv
import requests

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

# ML Client API URL (use environment variable if available)
ML_CLIENT_API_URL = os.getenv("ML_CLIENT_API_URL", "http://ml-client:5002/api/predict")

# Flask and SocketIO setup
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev_secret_key")
socketio = SocketIO(app, cors_allowed_origins="*")


# Connect to MongoDB with retry logic
def connect_mongo():
    """Connect to MongoDB with retry logic."""
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            client = MongoClient(MONGO_URI)
            mongo_db = client[MONGO_DB]
            # Test connection
            client.admin.command("ping")
            logger.info("Successfully connected to MongoDB")
            return client, mongo_db
        except Exception as e:
            logger.warning(
                "MongoDB connection attempt %d/%d failed: %s",
                attempt + 1,
                max_retries,
                e,
            )
            if attempt < max_retries - 1:
                logger.info("Retrying in %d seconds...")
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


def predict_command(wav_path):
    """
    Send audio to ML client for prediction.

    Args:
        wav_path: Path to WAV audio file

    Returns:
        command: Predicted command string
    """
    try:
        # Add retries for API calls
        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            try:
                with open(wav_path, "rb") as audio_file:
                    files = {
                        "audio": (os.path.basename(wav_path), audio_file, "audio/wav")
                    }

                    # Make request to the ML client API with proper timeout
                    response = requests.post(
                        ML_CLIENT_API_URL,
                        files=files,
                        timeout=3,  # Reduced timeout to prevent UI lag
                        headers={"Accept": "application/json"},
                    )

                if response.status_code == 200:
                    prediction_data = response.json()
                    command = prediction_data["command"]
                    confidence = prediction_data.get("confidence", 0)

                    # Log the result
                    logger.info(
                        f"Command from ML client: {command}, Confidence: {confidence:.4f}"
                    )

                    # Don't forward background noise commands to the game
                    if command == "background":
                        logger.info("Ignoring background noise")
                        return None  # Return None instead of "background"

                    return command
                else:
                    logger.error(
                        f"ML client error (attempt {attempt+1}): Status {response.status_code}, Response: {response.text}"
                    )
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        return "stop"  # Default on persistent failure
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed (attempt {attempt+1}): {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    return "stop"  # Default on persistent failure
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "stop"


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
            latest_scores = list(
                scores_collection.find().sort("timestamp", -1).limit(10)
            )
            return render_template("scores.html", scores=latest_scores)
        return render_template("scores.html", scores=[], error="Database not connected")
    except Exception as e:
        logger.error("Error retrieving scores: %s", e)
        return render_template("scores.html", scores=[], error=str(e))


@app.route("/score", methods=["POST"])
def receive_score():
    """Receive and store game scores."""
    try:
        data = request.get_json()
        score_value = data.get("score", 0)

        if score_value > 0 and scores_collection is not None:
            scores_collection.insert_one(
                {"score": score_value, "timestamp": datetime.now(timezone.utc)}
            )
            logger.info("Score saved: %d", score_value)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        logger.error("Error saving score: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/commands")
def get_commands():
    """API endpoint to retrieve recent commands."""
    try:
        if commands_collection is not None:
            recent_commands = list(
                commands_collection.find().sort("timestamp", -1).limit(20)
            )
            return dumps(recent_commands), 200, {"Content-Type": "application/json"}
        return jsonify({"error": "Database not connected"}), 503
    except Exception as e:
        logger.error("Error retrieving commands: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/dashboard")
def dashboard():
    """Render the dashboard page."""
    return render_template("dashboard.html")


# Socket.IO events
@socketio.on("connect")
def handle_connect():
    """Handle client connection."""
    logger.info("Client connected: %s", request.sid)


@socketio.on("disconnect")
def handle_disconnect():
    """Handle client disconnection."""
    logger.info("Client disconnected: %s", request.sid)


@socketio.on("audio")
def handle_audio(data_url):
    """Process audio data and predict commands."""
    try:
        # Decode base64 audio data
        _, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f_webm:
            f_webm.write(audio_bytes)
            webm_path = f_webm.name

        # Import here to avoid top-level import of a potentially unnecessary dependency
        from pydub import AudioSegment

        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_path = webm_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        # Get prediction from ML client
        command = predict_command(wav_path)

        # Only emit command if it's not None (i.e., not background noise)
        if command is not None:
            logger.info(f"Emitting command: {command}")
            emit("command", command)

            # Save command to database if connected
            if commands_collection is not None:
                commands_collection.insert_one(
                    {
                        "command": command,
                        "timestamp": datetime.now(timezone.utc),
                        "processed": True,
                    }
                )
        else:
            logger.info("Ignoring background noise, not emitting command")

        # Clean up temporary files
        try:
            os.remove(webm_path)
            os.remove(wav_path)
        except (IOError, OSError) as e:
            logger.warning(f"Error removing temporary files: {e}")

    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        emit("command", "stop")  # Default command on error


if __name__ == "__main__":
    logger.info("Starting Voice Flappy Game web server...")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
