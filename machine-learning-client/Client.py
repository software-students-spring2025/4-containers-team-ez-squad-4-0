#!/usr/bin/env python3
"""
Voice Command Recognition Machine Learning Client

This module implements a client that processes voice commands,
performs machine learning predictions, and interacts with MongoDB.
Recording is no longer done locally—audio files should be passed in,
for example, by a web front end.
"""

import os
import sys
import time
import logging
import datetime
import threading
import numpy as np
import tensorflow as tf
from pymongo import MongoClient
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
SAMPLE_RATE = 16000  # In seconds, our model expects 16kHz
DURATION = 1         # seconds, expected duration of each audio sample

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
            "background": 0
        }
        # Note: We no longer initialize PyAudio here since recording happens on the web

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
                    self.mongo_client.admin.command('ping')
                    logger.info("Successfully connected to MongoDB")
                    return
                except Exception as e:
                    logger.warning(f"MongoDB connection attempt {attempt+1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
            raise Exception("Failed to connect to MongoDB after multiple attempts")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    def load_model(self):
        """Load the trained CNN model and label encoder."""
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy']
            )
            logger.info(f"Loading label encoder from {ENCODER_PATH}")
            # Load using joblib for compatibility
            self.label_encoder = joblib.load(ENCODER_PATH)
            logger.info(f"Model loaded successfully. Classes: {self.label_encoder.classes_}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def extract_features(self, audio_data):
        """
        Extract MFCC features from audio data.
        Audio data should be a numpy array (int16) recorded at SAMPLE_RATE.
        Returns MFCC features with time-dimension fixed (e.g. 44 frames).
        """
        try:
            # Convert to float in [-1,1]
            audio_data = audio_data.astype(np.float32) / 32768.0
            mfccs = librosa.feature.mfcc(
                y=audio_data,
                sr=SAMPLE_RATE,
                n_mfcc=13,
                hop_length=512,
                n_fft=2048
            )
            # Fix the time dimension to 44 frames
            if mfccs.shape[1] < 44:
                pad_width = 44 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode="constant")
            else:
                mfccs = mfccs[:, :44]
            return mfccs
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
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
            features = self.extract_features(audio_data)
            # Expand dimensions for batch and channel: shape (1, 13, 44, 1)
            features = np.expand_dims(features, axis=[0, -1])
            predictions = self.model.predict(features)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index])
            predicted_class = self.label_encoder.inverse_transform([predicted_index])[0]
            if predicted_class in self.command_counts:
                self.command_counts[predicted_class] += 1
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "error", 0.0

    def save_to_database(self, prediction_data):
        """Save prediction data to MongoDB."""
        try:
            result = self.collection.insert_one(prediction_data)
            logger.info(f"Saved prediction to database with ID: {result.inserted_id}")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")

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
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            predicted_class, confidence = self.predict(y)
            timestamp = datetime.datetime.now()
            prediction_data = {
                "timestamp": timestamp,
                "command": predicted_class,
                "confidence": confidence,
                "file_path": file_path,
                "processed": True
            }
            self.save_to_database(prediction_data)
            logger.info(f"File {file_path}: Predicted {predicted_class} (confidence: {confidence:.2f})")
            return predicted_class, confidence
        except Exception as e:
            logger.error(f"Error processing audio file {file_path}: {e}")
            return "error", 0.0

    def process_directory(self, directory_path):
        """
        Process all audio files (e.g., WAV) in a directory.
        """
        try:
            audio_files = [f for f in os.listdir(directory_path) if f.endswith('.wav')]
            logger.info(f"Found {len(audio_files)} audio files in {directory_path}")
            for filename in audio_files:
                file_path = os.path.join(directory_path, filename)
                self.process_audio_file(file_path)
            logger.info("Command statistics:")
            for cmd, count in self.command_counts.items():
                logger.info(f"  {cmd}: {count}")
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {e}")

    # Note: The methods below (record_audio and continuous listening) are kept for legacy use,
    # but since you want recording to occur in the web client, they are not used.
    def record_audio(self):
        """(Legacy) Record audio from microphone. Not used since recording happens in the web."""
        logger.info("Recording is disabled; use web-based recording instead.")
        # Return a one-second silent audio array for testing:
        return np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.int16)

    def start_listening(self):
        """(Legacy) Continuous voice command recognition. Disabled in favor of web recording."""
        logger.info("Continuous listening is disabled. Use the web client for recording.")
        return

    def stop_listening(self):
        logger.info("Listening already stopped.")

    def close(self):
        """Clean up resources."""
        if hasattr(self, 'mongo_client'):
            self.mongo_client.close()
        logger.info("Resources cleaned up.")


def main():
    """Main function to process audio files or start the client service."""
    client = VoiceCommandClient()
    try:
        time.sleep(5)
        if len(sys.argv) > 1:
            if sys.argv[1] == "--process-dir" and len(sys.argv) > 2:
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
                print("  python Client.py                   # (No continuous listening; use web-based recording)")
                print("  python Client.py --process-dir DIR # Process all audio files in DIR")
                print("  python Client.py --process-file FILE # Process a single audio file")
        else:
            logger.info("No command-line processing selected. Waiting for audio via web client.")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Exiting...")
    finally:
        client.close()


if __name__ == "__main__":
    main()
