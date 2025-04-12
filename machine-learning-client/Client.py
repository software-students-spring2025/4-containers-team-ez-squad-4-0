#!/usr/bin/env python3
"""
Voice Command Recognition Machine Learning Client

This module implements a client that processes voice commands,
performs machine learning predictions, and interacts with MongoDB.
Recording is no longer done locally—audio files should be passed in,
for example, by a web front end.
"""

# Standard library imports
import os
import sys
import time
import logging
import datetime

# Third-party imports
import numpy as np

# pylint: disable=no-name-in-module, import-error
import tensorflow as tf
from flask import Flask, request, jsonify
# pylint: enable=no-name-in-module, import-error
from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    OperationFailure,
)
import librosa

from dotenv import load_dotenv
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

# Model paths
MODEL_PATH = os.getenv("MODEL_PATH", "cnn_model.h5")
ENCODER_PATH = os.getenv("ENCODER_PATH", "cnn_label_encoder.pkl")

# Audio parameters (for processing recorded files)
SAMPLE_RATE = 16000  # In Hz, our model expects 16kHz
DURATION = 1  # seconds, expected duration of each audio sample

# Initialize Flask app
app = Flask(__name__)

# Create a global client instance
client = None

class VoiceCommandClient:
    """Client for voice command recognition and processing."""

    def __init__(self):
        """Initialize the voice command client."""
        # Connect to MongoDB
        self.connect_to_mongodb()

        # Load the model and label encoder
        self.load_model()

        # Command statistics
        self.command_counts = {
            "up": 0,
            "down": 0,
            "left": 0,
            "right": 0,
            "go": 0,
            "stop": 0,
            "background": 0,
        }

    def connect_to_mongodb(self):
        """Connect to MongoDB database with retry logic."""
        try:
            max_retries = 5
            retry_delay = 5  # seconds
            for attempt in range(max_retries):
                try:
                    self.mongo_client = MongoClient(MONGO_URI)
                    self.db = self.mongo_client[MONGO_DB]
                    self.collection = self.db.commands
                    # Test connection
                    self.mongo_client.admin.command("ping")
                    logger.info("Successfully connected to MongoDB")
                    return
                except (
                    ConnectionFailure,
                    ServerSelectionTimeoutError,
                ) as e:
                    logger.warning(
                        "MongoDB connection attempt %d/%d failed: %s",
                        attempt + 1,
                        max_retries,
                        e,
                    )
                    if attempt < max_retries - 1:
                        logger.info("Retrying in %d seconds...", retry_delay)
                        time.sleep(retry_delay)
            raise ConnectionError(
                "Failed to connect to MongoDB after multiple attempts"
            )
        except Exception as e:
            logger.error("Failed to connect to MongoDB: %s", e)
            raise

    def load_model(self):
        """Load the trained CNN model and label encoder."""
        try:
            logger.info("Loading model from %s", MODEL_PATH)
            # Try with TensorFlow compatibility options
            try:
                # First, try to load the model directly
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            except (TypeError, ValueError) as e:
                # If that fails with batch_shape error, try with custom objects
                logger.warning(f"First attempt to load model failed: {e}, trying alternative approach")
                
                # Custom InputLayer class to handle batch_shape
                class CustomInputLayer(tf.keras.layers.InputLayer):
                    def __init__(self, batch_shape=None, **kwargs):
                        if batch_shape is not None:
                            kwargs['input_shape'] = batch_shape[1:]
                        super().__init__(**kwargs)
                
                # Try to load with custom object scope
                with tf.keras.utils.custom_object_scope({'InputLayer': CustomInputLayer}):
                    self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                    
                logger.info("Successfully loaded model with custom object scope")
            
            # Compile the model
            self.model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            
            logger.info("Loading label encoder from %s", ENCODER_PATH)
            # Load using joblib for compatibility
            self.label_encoder = joblib.load(ENCODER_PATH)
            logger.info(
                "Model loaded successfully. Classes: %s", self.label_encoder.classes_
            )
        except Exception as e:
            logger.error("Failed to load model: %s", e)
            raise

    def extract_features(self, audio_data):
        """
        Extract log-mel spectrogram features from audio data.
        This matches the feature extraction in extract_features.py.
        """
        try:
            # Ensure the audio is exactly 1 second (16,000 samples)
            TARGET_SAMPLES = 16000
            if len(audio_data) < TARGET_SAMPLES:
                audio_data = np.pad(audio_data, (0, TARGET_SAMPLES - len(audio_data)))
            else:
                audio_data = audio_data[:TARGET_SAMPLES]

            # Compute the mel spectrogram with 128 mel bins
            N_MELS = 128
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, 
                sr=SAMPLE_RATE, 
                n_mels=N_MELS,
                n_fft=2048,
                hop_length=512
            )
            log_mel = librosa.power_to_db(mel_spec)

            # Ensure fixed time frames (44 frames)
            SPEC_FRAMES = 44
            if log_mel.shape[1] < SPEC_FRAMES:
                pad_width = SPEC_FRAMES - log_mel.shape[1]
                log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode="constant")
            else:
                log_mel = log_mel[:, :SPEC_FRAMES]

            return log_mel
        except (ValueError, RuntimeError) as e:
            logger.error("Error extracting log-mel features: %s", e)
            raise

    def predict(self, audio_data):
        """
        Predict the voice command class from audio data.

        Args:
            audio_data: Numpy array of recorded audio data.

        Returns:
            predicted_class: The predicted command name.
            confidence: The confidence score.
        """
        try:
            # Extract log-mel spectrogram features
            features = self.extract_features(audio_data)
            
            # Expand dimensions for batch and channel: shape (1, 128, 44, 1)
            features = np.expand_dims(features, axis=0)  # Now (1, 128, 44)
            features = np.expand_dims(features, axis=-1)  # Now (1, 128, 44, 1)
            
            # Debug information
            logger.info(f"Feature shape: {features.shape}")
            
            # Make prediction
            predictions = self.model.predict(features)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index])
            predicted_class = self.label_encoder.inverse_transform([predicted_index])[0]
            
            # Apply confidence threshold
            if confidence < 0.6:
                predicted_class = "background"
                
            if predicted_class in self.command_counts:
                self.command_counts[predicted_class] += 1
                
            return predicted_class, confidence
        except (ValueError, RuntimeError) as e:
            logger.error("Error during prediction: %s", e)
            return "stop", 0.0  # Return "stop" on error (safer than "error")
        except Exception as e:
            logger.error(f"Unexpected error in prediction: {e}")
            return "stop", 0.0

    def save_to_database(self, prediction_data):
        """Save prediction data to MongoDB."""
        try:
            result = self.collection.insert_one(prediction_data)
            logger.info("Saved prediction to database with ID: %s", result.inserted_id)
        except (ConnectionFailure, OperationFailure) as e:
            logger.error(f"Error saving to database: {e}")
            # Don't raise here, just log the error so prediction can still work without DB
        except Exception as e:
            logger.error(f"Unexpected error saving to database: {e}")

    def process_audio_file(self, file_path):
        """
        Process an audio file and perform prediction.

        Args:
            file_path: Path to an audio file (e.g., WAV format)

        Returns:
            predicted_class: Predicted command name.
            confidence: Confidence score.
        """
        try:
            # Load audio file with correct sample rate
            y, _ = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Debug information
            logger.info(f"Loaded audio with shape: {y.shape}")
            
            # Get prediction
            predicted_class, confidence = self.predict(y)
            
            # Create prediction data
            timestamp = datetime.datetime.now()
            prediction_data = {
                "timestamp": timestamp,
                "command": predicted_class,
                "confidence": confidence,
                "file_path": file_path,
                "processed": True,
            }
            
            # Save to database but don't fail if database is unavailable
            try:
                self.save_to_database(prediction_data)
            except Exception as e:
                logger.error(f"Could not save to database, but continuing with prediction: {e}")
                
            logger.info(
                "File %s: Predicted %s (confidence: %.2f)",
                file_path,
                predicted_class,
                confidence,
            )
            return predicted_class, confidence
        except (IOError, ValueError) as e:
            logger.error("Error processing audio file %s: %s", file_path, e)
            return "stop", 0.0  # Return "stop" instead of "error" as a safer default command
        except Exception as e:
            logger.error(f"Unexpected error processing audio file: {e}")
            return "stop", 0.0

    def process_directory(self, directory_path):
        """
        Process all audio files (e.g., WAV) in a directory.
        """
        try:
            audio_files = [f for f in os.listdir(directory_path) if f.endswith(".wav")]
            logger.info("Found %d audio files in %s", len(audio_files), directory_path)
            for filename in audio_files:
                file_path = os.path.join(directory_path, filename)
                self.process_audio_file(file_path)
            logger.info("Command statistics:")
            for cmd, count in self.command_counts.items():
                logger.info("  %s: %d", cmd, count)
        except (IOError, OSError) as e:
            logger.error("Error processing directory %s: %s", directory_path, e)

    def record_audio(self):
        """(Legacy) Record audio from microphone. Not used since recording happens in the web."""
        logger.info("Recording is disabled; use web-based recording instead.")
        # Return a one-second silent audio array for testing:
        return np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.int16)

    def start_listening(self):
        """(Legacy) Continuous voice command recognition. Disabled in favor of web recording."""
        logger.info(
            "Continuous listening is disabled. Use the web client for recording."
        )

    def stop_listening(self):
        """Stop listening to audio input."""
        logger.info("Listening already stopped.")

    def close(self):
        """Clean up resources."""
        if hasattr(self, "mongo_client"):
            self.mongo_client.close()
        logger.info("Resources cleaned up.")


# API routes
@app.route('/api/predict', methods=['POST'])
def predict_api():
    """API endpoint for real-time voice command prediction."""
    global client
    
    try:
        # Initialize client if not already done
        if client is None:
            try:
                client = VoiceCommandClient()
            except Exception as e:
                logger.error(f"Failed to initialize client: {e}")
                return jsonify({"error": "Failed to initialize ML client", "command": "stop"}), 500
            
        # Check if there's a file in the request
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided", "command": "stop"}), 400
            
        audio_file = request.files['audio']
        
        # Save to a temporary file
        temp_path = "/tmp/temp_audio.wav"
        audio_file.save(temp_path)
        
        # Process the audio file
        try:
            predicted_class, confidence = client.process_audio_file(temp_path)
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return jsonify({"error": str(e), "command": "stop"}), 500
        
        # Return the prediction
        return jsonify({
            "command": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        logger.error(f"Error in prediction API: {e}")
        return jsonify({"error": str(e), "command": "stop"}), 500


def main():
    """Main entry point for the application."""
    global client
    
    try:
        if len(sys.argv) > 1:
            try:
                client = VoiceCommandClient()
            except Exception as e:
                logger.error(f"Failed to initialize client: {e}")
                sys.exit(1)
                
            # Process command-line arguments
            if sys.argv[1] == "--api":
                # Start the Flask API server
                logger.info("Starting ML prediction API on port 5002")
                app.run(host="0.0.0.0", port=int(os.getenv("API_PORT", "5002")), debug=False)
            elif sys.argv[1] == "--process-dir" and len(sys.argv) > 2:
                directory_path = sys.argv[2]
                logger.info(f"Processing directory: {directory_path}")
                client.process_directory(directory_path)
            elif sys.argv[1] == "--process-file" and len(sys.argv) > 2:
                file_path = sys.argv[2]
                logger.info(f"Processing file: {file_path}")
                client.process_audio_file(file_path)
            else:
                logger.error("Invalid command line arguments")
                print("Usage:")
                print("  python Client.py --api               # Start the prediction API")
                print("  python Client.py --process-dir DIR   # Process all audio files in DIR")
                print("  python Client.py --process-file FILE # Process a single audio file")
            
            if sys.argv[1] != "--api":
                client.close()
        else:
            # Default: Start the Flask API server
            logger.info("Starting ML prediction API on port 5002")
            try:
                # Initialize client before starting the server
                client = VoiceCommandClient()
                app.run(host="0.0.0.0", port=int(os.getenv("API_PORT", "5002")), debug=False)
            except Exception as e:
                logger.error(f"Error initializing client or starting API server: {e}")
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
        if client:
            client.close()


if __name__ == "__main__":
    main()