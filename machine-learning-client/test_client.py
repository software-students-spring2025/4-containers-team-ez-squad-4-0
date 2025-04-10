#!/usr/bin/env python3
"""
Tests for the Voice Command ML Client.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import numpy as np
import datetime
from Client import VoiceCommandClient


class TestVoiceCommandClient(unittest.TestCase):
    @patch("pymongo.MongoClient")
    @patch("tensorflow.keras.models.load_model")
    @patch("joblib.load")
    def setUp(self, mock_joblib, mock_load_model, mock_mongo):
        # Mock dependencies
        self.mock_model = mock_load_model.return_value
        self.mock_label_encoder = mock_joblib.return_value
        self.mock_label_encoder.classes_ = [
            "up",
            "down",
            "left",
            "right",
            "go",
            "stop",
            "background",
        ]
        self.mock_mongo_client = mock_mongo.return_value
        self.mock_db = self.mock_mongo_client.__getitem__.return_value
        self.mock_collection = self.mock_db.__getitem__.return_value

        # Initialize client
        self.client = VoiceCommandClient()

    def test_extract_features(self):
        """Test feature extraction from audio data."""
        # Create dummy audio data (16kHz, 1 second)
        audio_data = np.zeros(16000, dtype=np.int16)

        # Test feature extraction
        features = self.client.extract_features(audio_data)

        # Check output shape
        self.assertEqual(features.shape[0], 13)  # 13 MFCCs
        self.assertEqual(features.shape[1], 44)  # 44 time frames

    @patch("numpy.argmax")
    def test_predict(self, mock_argmax):
        """Test prediction from audio data."""
        # Setup mocks
        mock_argmax.return_value = 0
        self.client.model.predict.return_value = np.array([[0.9, 0.05, 0.05]])
        self.client.label_encoder.inverse_transform.return_value = np.array(["up"])

        # Create dummy audio data
        audio_data = np.zeros(16000, dtype=np.float32)

        # Test prediction
        command, confidence = self.client.predict(audio_data)

        # Check results
        self.assertEqual(command, "up")
        self.assertAlmostEqual(confidence, 0.9)
        self.assertEqual(self.client.command_counts["up"], 1)

    def test_save_to_database(self):
        """Test saving prediction data to database."""
        # Setup test data
        prediction_data = {
            "command": "up",
            "confidence": 0.9,
            "timestamp": datetime.datetime.now(),
        }

        # Mock the insert_one result
        self.client.collection.insert_one.return_value = MagicMock()
        self.client.collection.insert_one.return_value.inserted_id = "test_id"

        # Call the method
        self.client.save_to_database(prediction_data)

        # Verify the call
        self.client.collection.insert_one.assert_called_once_with(prediction_data)

    @patch("librosa.load")
    def test_process_audio_file(self, mock_librosa_load):
        """Test processing an audio file."""
        # Setup mocks
        mock_librosa_load.return_value = (np.zeros(16000), 16000)

        # Mock the predict method
        self.client.predict = MagicMock(return_value=("up", 0.95))
        self.client.save_to_database = MagicMock()

        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            # Test the method
            command, confidence = self.client.process_audio_file(temp_file.name)

            # Verify results
            self.assertEqual(command, "up")
            self.assertEqual(confidence, 0.95)
            mock_librosa_load.assert_called_once()
            self.client.predict.assert_called_once()
            self.client.save_to_database.assert_called_once()

    @patch("os.listdir")
    def test_process_directory(self, mock_listdir):
        """Test processing a directory of audio files."""
        # Setup mock
        mock_listdir.return_value = ["file1.wav", "file2.wav", "other.txt"]

        # Mock process_audio_file
        self.client.process_audio_file = MagicMock(return_value=("up", 0.9))

        # Test method
        self.client.process_directory("/test/dir")

        # Verify correct files were processed
        self.assertEqual(self.client.process_audio_file.call_count, 2)
        # Check the calls were made with the correct filenames
        expected_calls = [
            unittest.mock.call("/test/dir/file1.wav"),
            unittest.mock.call("/test/dir/file2.wav"),
        ]
        self.client.process_audio_file.assert_has_calls(expected_calls, any_order=True)

    def test_record_audio(self):
        """Test the record_audio method returns silent data of correct shape."""
        audio_data = self.client.record_audio()
        self.assertEqual(audio_data.shape, (16000,))
        self.assertEqual(audio_data.dtype, np.int16)
        self.assertEqual(np.sum(audio_data), 0)  # Should be silent

    def test_close(self):
        """Test cleanup of resources."""
        self.client.close()
        self.client.mongo_client.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
