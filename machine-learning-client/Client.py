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
import threading

# Third-party imports
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    OperationFailure,
)
import librosa
from flask import Flask, request, jsonify
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
        """Load the trained CNN model and label encoder with robust error handling."""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")

            # First, try the standard approach
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
                logger.info("Successfully loaded model with standard method")
            except Exception as e:
                logger.warning(
                    f"Standard model loading failed: {e}, trying with custom options"
                )

                # Try with different options if standard approach fails
                options = tf.saved_model.LoadOptions(
                    experimental_io_device="/job:localhost"
                )

                try:
                    self.model = tf.keras.models.load_model(
                        MODEL_PATH, compile=False, options=options
                    )
                    logger.info("Successfully loaded model with custom options")
                except Exception as inner_e:
                    logger.warning(
                        f"Custom options loading failed: {inner_e}, trying tf.keras.models.load_model"
                    )

                    # Last resort - custom object scope approach
                    custom_objects = {
                        "InputLayer": tf.keras.layers.InputLayer,
                        "Conv2D": tf.keras.layers.Conv2D,
                        "MaxPooling2D": tf.keras.layers.MaxPooling2D,
                        "Flatten": tf.keras.layers.Flatten,
                        "Dense": tf.keras.layers.Dense,
                        "Dropout": tf.keras.layers.Dropout,
                    }

                    with tf.keras.utils.custom_object_scope(custom_objects):
                        self.model = tf.keras.models.load_model(
                            MODEL_PATH, compile=False
                        )
                    logger.info("Successfully loaded model with custom object scope")

            # Compile the model
            self.model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )

            # Log model summary
            self.model.summary(print_fn=logger.info)

            # Load label encoder
            logger.info(f"Loading label encoder from {ENCODER_PATH}")
            self.label_encoder = joblib.load(ENCODER_PATH)
            logger.info(
                f"Model loaded successfully. Classes: {self.label_encoder.classes_}"
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract_features(self, audio_data):
        """
        Extract log-mel spectrogram features from audio data.
        This exactly matches the feature extraction in extract_features.py.

        Args:
            audio_data: Numpy array of audio data

        Returns:
            log_mel_spec: Log-Mel spectrogram features (128, 44)
        """
        try:
            # Ensure consistent length by padding or truncating to 1 second at 16kHz
            if len(audio_data) < SAMPLE_RATE:
                audio_data = np.pad(audio_data, (0, SAMPLE_RATE - len(audio_data)))
            else:
                audio_data = audio_data[:SAMPLE_RATE]

            # Convert to Mel spectrogram with 128 mel bins
            mel_spec = librosa.feature.melspectrogram(
                y=audio_data, sr=SAMPLE_RATE, n_mels=128, n_fft=2048, hop_length=512
            )

            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec)

            # Ensure consistent time dimension to exactly 44 frames
            if log_mel_spec.shape[1] < 44:
                pad_width = 44 - log_mel_spec.shape[1]
                log_mel_spec = np.pad(
                    log_mel_spec, ((0, 0), (0, pad_width)), mode="constant"
                )
            else:
                log_mel_spec = log_mel_spec[:, :44]

            return log_mel_spec
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise

    def predict(self, audio_data):
        """
        Predict the voice command class from audio data with improved decision logic.

        Args:
            audio_data: Numpy array of recorded audio data.

        Returns:
            predicted_class: The predicted command name.
            confidence: The confidence score.
        """
        try:
            # Check audio energy level first - reject very quiet audio
            audio_energy = np.sqrt(np.mean(np.square(audio_data)))
            if audio_energy < 0.01:
                logger.info(f"Audio energy too low: {audio_energy:.6f}")
                return "background", 0.0

            # Initialize command history if it doesn't exist
            if not hasattr(self, "command_history"):
                self.command_history = []
                self.up_down_confusion_count = 0

            # Extract log-mel spectrogram features
            features = self.extract_features(audio_data)

            # Debug the feature shape
            logger.info(f"Feature shape before batch dimension: {features.shape}")

            # Expand dimensions for batch and channel: shape (1, 128, 44, 1)
            features = np.expand_dims(features, axis=0)  # Add batch dimension
            features = np.expand_dims(features, axis=-1)  # Add channel dimension

            logger.info(f"Feature shape after adding dimensions: {features.shape}")

            # Make prediction with error handling
            try:
                predictions = self.model.predict(
                    features, verbose=0
                )  # Set verbose=0 to reduce log noise
            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                # Try reshaping if needed
                try:
                    if len(features.shape) != 4:
                        logger.info("Attempting to reshape features to 4D tensor...")
                        features = features.reshape(1, 128, 44, 1)
                    predictions = self.model.predict(features, verbose=0)
                except Exception as inner_e:
                    logger.error(f"Reshaping and prediction retry failed: {inner_e}")
                    return "stop", 0.0

            # Get top predictions for better decision making
            top_indices = np.argsort(predictions[0])[-3:][::-1]  # Get top 3 predictions
            top_classes = self.label_encoder.inverse_transform(top_indices)
            top_confidences = [float(predictions[0][idx]) for idx in top_indices]

            # Log all top predictions for debugging
            for i, (cls, conf) in enumerate(zip(top_classes, top_confidences)):
                logger.info(f"Top {i+1} prediction: class={cls}, confidence={conf:.4f}")

            # Get the top prediction
            predicted_class = top_classes[0]
            confidence = top_confidences[0]

            # Special handling for up/down confusion - CHECK IF "DOWN" IS MISRECOGNIZING "UP"
            if predicted_class == "down" and "up" in top_classes:
                # Find index of "up" in top_classes
                up_idx = list(top_classes).index("up")
                up_confidence = top_confidences[up_idx]

                # If "up" and "down" are very close in confidence, bias toward "up"
                if abs(confidence - up_confidence) < 0.2:  # They're close
                    logger.info(
                        f"🔄 Up/Down confusion detected! down:{confidence:.4f} vs up:{up_confidence:.4f}"
                    )
                    # Bias toward "up" since it seems to be under-recognized
                    if (
                        up_confidence > confidence * 0.8
                    ):  # Add significant bias toward "up"
                        predicted_class = "up"
                        confidence = up_confidence
                        logger.info(
                            f"🔄 Resolving up/down confusion: choosing 'up' command"
                        )

            # Apply more sophisticated decision logic

            # 1. If the top prediction is very confident, use it
            if confidence > 0.7:
                logger.info(
                    f"High confidence prediction: {predicted_class} ({confidence:.4f})"
                )
                # Just a sanity check for the known commands
                if predicted_class not in ["up", "down", "go", "stop", "background"]:
                    logger.warning(
                        f"Unknown command predicted with high confidence: {predicted_class}"
                    )
                    return "background", confidence

            # 2. For game control commands (up/down), use a lower threshold
            elif predicted_class in ["up", "down"] and confidence > 0.3:
                logger.info(
                    f"Game control command accepted: {predicted_class} ({confidence:.4f})"
                )

            # 3. For other commands, use a medium threshold
            elif predicted_class in ["go", "stop"] and confidence > 0.4:
                logger.info(
                    f"Game command accepted: {predicted_class} ({confidence:.4f})"
                )

            # 4. If top two predictions are close, prefer game commands over background
            elif len(top_confidences) >= 2 and (
                top_confidences[0] - top_confidences[1] < 0.2
            ):
                # If the difference is small and second prediction is a game command, use it instead
                if top_classes[0] == "background" and top_classes[1] in [
                    "up",
                    "down",
                    "go",
                    "stop",
                ]:
                    predicted_class = top_classes[1]
                    confidence = top_confidences[1]
                    logger.info(
                        f"Preferring game command over background: {predicted_class} ({confidence:.4f})"
                    )
                # Similarly, prefer up/down over other commands when close
                elif (
                    top_classes[1] in ["up", "down"]
                    and top_classes[0] not in ["up", "down"]
                    and top_confidences[1] > 0.35
                ):
                    predicted_class = top_classes[1]
                    confidence = top_confidences[1]
                    logger.info(
                        f"Preferring directional command: {predicted_class} ({confidence:.4f})"
                    )

            # 5. If confidence is too low, return background
            elif confidence < 0.3:
                logger.info(
                    f"Low confidence ({confidence:.4f}), treating as background"
                )
                predicted_class = "background"

            # Additional adaptive correction based on command history
            if len(self.command_history) > 0:
                # Track confusion patterns
                if predicted_class == "down" and self.command_history[-1] == "up":
                    self.up_down_confusion_count += 1

                # If we've detected consistent confusion, apply stronger correction
                if self.up_down_confusion_count >= 2 and predicted_class == "down":
                    # Look for "up" in any of the top predictions
                    for idx, cls in enumerate(top_classes):
                        if cls == "up" and top_confidences[idx] > 0.3:
                            predicted_class = "up"
                            confidence = top_confidences[idx]
                            logger.info(
                                "🔄 Applied adaptive correction: 'down' → 'up' based on pattern"
                            )
                            break

            # Store command in history (limit to last 5)
            self.command_history.append(predicted_class)
            if len(self.command_history) > 5:
                self.command_history.pop(0)

            # Update command statistics
            if predicted_class in self.command_counts:
                self.command_counts[predicted_class] += 1

            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Unexpected error in prediction: {e}")
            return "stop", 0.0  # Return "stop" as a safe default

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
            # Load audio file with correct sample rate and error handling
            try:
                y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                logger.info(f"Loaded audio with shape: {y.shape}, sample rate: {sr}")

                # Additional check for valid audio data
                if len(y) == 0:
                    logger.warning("Empty audio file detected")
                    return "stop", 0.0

                # Check for mostly silence
                if np.abs(y).max() < 0.01:  # Very quiet audio
                    logger.warning("Audio file contains mostly silence")
                    return "background", 0.0

            except Exception as e:
                logger.error(f"Error loading audio file: {e}")
                return "stop", 0.0

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
                logger.error(
                    f"Could not save to database, but continuing with prediction: {e}"
                )

            logger.info(
                f"File {file_path}: Predicted {predicted_class} (confidence: {confidence:.2f})"
            )
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Unexpected error processing audio file: {e}")
            return "stop", 0.0  # Return "stop" as a safer default command

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

    def close(self):
        """Clean up resources."""
        if hasattr(self, "mongo_client"):
            self.mongo_client.close()
        logger.info("Resources cleaned up.")


# API routes
@app.route("/api/predict", methods=["POST"])
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
                return (
                    jsonify(
                        {
                            "error": "Failed to initialize ML client",
                            "command": "stop",
                            "confidence": 0.0,
                        }
                    ),
                    500,
                )

        # Check if there's a file in the request
        if "audio" not in request.files:
            return (
                jsonify(
                    {
                        "error": "No audio file provided",
                        "command": "stop",
                        "confidence": 0.0,
                    }
                ),
                400,
            )

        audio_file = request.files["audio"]

        # Save to a temporary file with proper error handling
        try:
            temp_path = "/tmp/temp_audio.wav"
            audio_file.save(temp_path)

            # Verify file was saved and is not empty
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                logger.error("Failed to save audio file or file is empty")
                return (
                    jsonify(
                        {
                            "error": "Failed to save audio file or file is empty",
                            "command": "stop",
                            "confidence": 0.0,
                        }
                    ),
                    500,
                )
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return (
                jsonify(
                    {
                        "error": f"Error saving audio file: {str(e)}",
                        "command": "stop",
                        "confidence": 0.0,
                    }
                ),
                500,
            )

        # Process the audio file
        try:
            predicted_class, confidence = client.process_audio_file(temp_path)

            # Log detailed info
            logger.info(
                f"API prediction result: {predicted_class} with confidence {confidence:.4f}"
            )

            # Return the prediction
            return jsonify({"command": predicted_class, "confidence": confidence})
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            return jsonify({"error": str(e), "command": "stop", "confidence": 0.0}), 500
    except Exception as e:
        logger.error(f"Unexpected error in prediction API: {e}")
        return jsonify({"error": str(e), "command": "stop", "confidence": 0.0}), 500


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
                app.run(
                    host="0.0.0.0", port=int(os.getenv("API_PORT", "5002")), debug=False
                )
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
                print(
                    "  python Client.py --api               # Start the prediction API"
                )
                print(
                    "  python Client.py --process-dir DIR   # Process all audio files in DIR"
                )
                print(
                    "  python Client.py --process-file FILE # Process a single audio file"
                )

            if sys.argv[1] != "--api":
                client.close()
        else:
            # Default: Start the Flask API server
            logger.info("Starting ML prediction API on port 5002")
            try:
                # Initialize client before starting the server
                client = VoiceCommandClient()
                app.run(
                    host="0.0.0.0", port=int(os.getenv("API_PORT", "5002")), debug=False
                )
            except Exception as e:
                logger.error(f"Error initializing client or starting API server: {e}")
                sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
        if client:
            client.close()


if __name__ == "__main__":
    main()
