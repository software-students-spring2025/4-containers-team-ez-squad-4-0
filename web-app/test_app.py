#!/usr/bin/env python3
"""
Tests for the Voice Command Game Web App
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import base64
import datetime
from io import BytesIO

import pytest
from flask import Flask
from flask.testing import FlaskClient
from pymongo.collection import Collection
import requests

# Import pydub to make it available for patching
import pydub

# Import the application to test
import app
from app import socketio


class TestVoiceCommandGameApp(unittest.TestCase):
    """Test cases for the Voice Command Game Web App."""

    def setUp(self):
        """Set up the test client and mock MongoDB collections."""
        # Configure app for testing
        app.app.config["TESTING"] = True
        app.app.config["SERVER_NAME"] = "localhost"

        # Create a test client
        self.client = app.app.test_client()

        # Mock MongoDB collections
        self.mock_scores_collection = MagicMock(spec=Collection)
        self.mock_commands_collection = MagicMock(spec=Collection)

        # Patch the MongoDB collections
        self.scores_patch = patch.object(
            app, "scores_collection", self.mock_scores_collection
        )
        self.commands_patch = patch.object(
            app, "commands_collection", self.mock_commands_collection
        )

        # Start the patches
        self.scores_patch.start()
        self.commands_patch.start()

    def tearDown(self):
        """Clean up after each test."""
        # Stop the patches
        self.scores_patch.stop()
        self.commands_patch.stop()

    # ========== Test Routes ==========

    def test_index_route(self):
        """Test the index route returns the game page."""
        response = self.client.get("/")
        assert response.status_code == 200
        # Check that it returns HTML
        assert b"<!DOCTYPE html>" in response.data

    def test_scores_route_with_scores(self):
        """Test the scores route when scores are available."""
        # Mock the find method to return some test scores
        mock_scores = [
            {"score": 10, "timestamp": datetime.datetime.now()},
            {"score": 20, "timestamp": datetime.datetime.now()},
        ]
        self.mock_scores_collection.find.return_value.sort.return_value.limit.return_value = (
            mock_scores
        )

        response = self.client.get("/scores")
        assert response.status_code == 200
        # Should include the scores in the response
        assert b"score" in response.data

    def test_scores_route_no_scores(self):
        """Test the scores route when no scores are available."""
        # Return an empty list from the mock
        self.mock_scores_collection.find.return_value.sort.return_value.limit.return_value = (
            []
        )

        response = self.client.get("/scores")
        assert response.status_code == 200
        assert b"No scores recorded yet" in response.data

    def test_scores_route_no_db(self):
        """Test the scores route when the database is not connected."""
        # Temporarily replace scores_collection with None
        with patch.object(app, "scores_collection", None):
            response = self.client.get("/scores")
            assert response.status_code == 200
            assert b"Database not connected" in response.data

    def test_dashboard_route(self):
        """Test the dashboard route returns the dashboard page."""
        response = self.client.get("/dashboard")
        assert response.status_code == 200
        # Check that it returns HTML
        assert b"<!DOCTYPE html>" in response.data
        assert b"Voice Command Analytics" in response.data

    def test_receive_score_valid(self):
        """Test the score endpoint with a valid score."""
        score_data = {"score": 42}

        response = self.client.post(
            "/score", data=json.dumps(score_data), content_type="application/json"
        )

        assert response.status_code == 200
        # Check that the score was saved to the database
        self.mock_scores_collection.insert_one.assert_called_once()
        assert json.loads(response.data)["status"] == "success"

    def test_receive_score_invalid(self):
        """Test the score endpoint with an invalid score."""
        # Mock insert_one to raise an exception
        self.mock_scores_collection.insert_one.side_effect = Exception("Database error")

        score_data = {"score": 42}
        response = self.client.post(
            "/score", data=json.dumps(score_data), content_type="application/json"
        )

        assert response.status_code == 500
        assert json.loads(response.data)["status"] == "error"

    def test_receive_score_zero(self):
        """Test the score endpoint with a zero score (shouldn't be saved)."""
        score_data = {"score": 0}

        response = self.client.post(
            "/score", data=json.dumps(score_data), content_type="application/json"
        )

        assert response.status_code == 200
        # Check that no score was saved
        self.mock_scores_collection.insert_one.assert_not_called()

    def test_get_commands_with_data(self):
        """Test the commands API endpoint when commands are available."""
        # Mock find to return some test commands
        mock_commands = [
            {"command": "up", "timestamp": datetime.datetime.now()},
            {"command": "down", "timestamp": datetime.datetime.now()},
        ]
        self.mock_commands_collection.find.return_value.sort.return_value.limit.return_value = (
            mock_commands
        )

        response = self.client.get("/api/commands")
        assert response.status_code == 200
        # Data should be JSON
        assert "application/json" in response.content_type

    def test_get_commands_no_db(self):
        """Test the commands API endpoint when database is not connected."""
        # Temporarily replace commands_collection with None
        with patch.object(app, "commands_collection", None):
            response = self.client.get("/api/commands")
            assert response.status_code == 503
            data = json.loads(response.data)
            assert "error" in data
            assert data["error"] == "Database not connected"

    # ========== Test Audio Processing ==========

    def test_handle_audio_valid_command(self):
        """Test the audio socket handler with a valid command."""
        with patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch(
            "base64.b64decode"
        ) as mock_b64decode, patch("app.predict_command") as mock_predict, patch(
            "os.remove"
        ) as mock_remove:

            # Setup mocks
            mock_file = MagicMock()
            mock_file.name = "/tmp/temp_audio.webm"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            mock_b64decode.return_value = b"test_audio_data"

            # Create a mock AudioSegment
            mock_audio = MagicMock()
            mock_audio.set_frame_rate.return_value.set_channels.return_value = (
                mock_audio
            )

            # Mock the AudioSegment import and instantiation
            with patch.dict("sys.modules", {"pydub": MagicMock()}):
                # Mock the imported AudioSegment within the function
                with patch(
                    "pydub.AudioSegment", MagicMock()
                ) as mock_audio_segment_class:
                    mock_audio_segment_class.from_file.return_value = mock_audio

                    # Mock the prediction to return 'up'
                    mock_predict.return_value = "up"

                    # Create a socket test client
                    socket_client = socketio.test_client(app.app)

                    # Emit the audio event with mock data
                    socket_client.emit(
                        "audio", "data:audio/webm;base64,SGVsbG8gV29ybGQ="
                    )

                    # Check that the socket handler called predict_command
                    mock_predict.assert_called_once()

                    # Check that socket received a command
                    received = socket_client.get_received()
                    assert len(received) > 0
                    assert received[0]["name"] == "command"
                    assert received[0]["args"][0] == "up"

                    # Check that the command was saved to the database
                    self.mock_commands_collection.insert_one.assert_called_once()

    def test_handle_audio_background_noise(self):
        """Test the audio socket handler with background noise (no command)."""
        with patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch(
            "base64.b64decode"
        ) as mock_b64decode, patch("app.predict_command") as mock_predict, patch(
            "os.remove"
        ) as mock_remove:

            # Setup mocks
            mock_file = MagicMock()
            mock_file.name = "/tmp/temp_audio.webm"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            mock_b64decode.return_value = b"test_audio_data"

            # Create a mock AudioSegment
            mock_audio = MagicMock()
            mock_audio.set_frame_rate.return_value.set_channels.return_value = (
                mock_audio
            )

            # Mock the AudioSegment import and instantiation
            with patch.dict("sys.modules", {"pydub": MagicMock()}):
                # Mock the imported AudioSegment within the function
                with patch(
                    "pydub.AudioSegment", MagicMock()
                ) as mock_audio_segment_class:
                    mock_audio_segment_class.from_file.return_value = mock_audio

                    # Mock the prediction to return None (background noise)
                    mock_predict.return_value = None

                    # Create a socket test client
                    socket_client = socketio.test_client(app.app)

                    # Emit the audio event with mock data
                    socket_client.emit(
                        "audio", "data:audio/webm;base64,SGVsbG8gV29ybGQ="
                    )

                    # Check that the socket handler called predict_command
                    mock_predict.assert_called_once()

                    # Check that no command was sent (no events received)
                    received = socket_client.get_received()
                    assert len(received) == 0

                    # Check that no command was saved to the database
                    self.mock_commands_collection.insert_one.assert_not_called()

    def test_handle_audio_error(self):
        """Test the audio socket handler when an error occurs."""
        with patch("tempfile.NamedTemporaryFile") as mock_tempfile, patch(
            "base64.b64decode"
        ) as mock_b64decode, patch("os.remove") as mock_remove:

            # Setup mocks
            mock_file = MagicMock()
            mock_file.name = "/tmp/temp_audio.webm"
            mock_tempfile.return_value.__enter__.return_value = mock_file
            mock_b64decode.return_value = b"test_audio_data"

            # Mock the AudioSegment import to raise an exception
            with patch.dict("sys.modules", {"pydub": MagicMock()}):
                # Make AudioSegment raise an exception
                with patch(
                    "pydub.AudioSegment", MagicMock()
                ) as mock_audio_segment_class:
                    mock_audio_segment_class.from_file.side_effect = Exception(
                        "Audio processing error"
                    )

                    # Create a socket test client
                    socket_client = socketio.test_client(app.app)

                    # Emit the audio event with mock data
                    socket_client.emit(
                        "audio", "data:audio/webm;base64,SGVsbG8gV29ybGQ="
                    )

                    # Check that socket received the default 'stop' command
                    received = socket_client.get_received()
                    assert len(received) > 0
                    assert received[0]["name"] == "command"
                    assert received[0]["args"][0] == "stop"

    # ========== Test Prediction ==========

    def test_predict_command_success(self):
        """Test successful prediction from ML client."""
        # Setup mock response
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"command": "up", "confidence": 0.95}
            mock_post.return_value = mock_response

            # Mock file opening
            m = mock_open(read_data=b"test audio data")
            with patch("builtins.open", m):
                # Call the function
                result = app.predict_command("/tmp/test.wav")

            # Check that it returned the correct command
            assert result == "up"

            # Check that it made the API request correctly
            mock_post.assert_called_once()
            args, kwargs = mock_post.call_args
            assert args[0] == app.ML_CLIENT_API_URL
            assert "files" in kwargs

    def test_predict_command_background(self):
        """Test prediction that returns background noise."""
        # Setup mock response
        with patch("requests.post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "command": "background",
                "confidence": 0.4,
            }
            mock_post.return_value = mock_response

            # Mock file opening
            m = mock_open(read_data=b"test audio data")
            with patch("builtins.open", m):
                # Call the function
                result = app.predict_command("/tmp/test.wav")

            # Check that it returned None for background noise
            assert result is None

    def test_predict_command_api_error(self):
        """Test prediction when API returns an error."""
        # Setup mock to raise an exception
        with patch("requests.post") as mock_post:
            mock_post.side_effect = requests.exceptions.RequestException("API error")

            # Mock file opening
            m = mock_open(read_data=b"test audio data")
            with patch("builtins.open", m):
                # Call the function
                result = app.predict_command("/tmp/test.wav")

            # Check that it returned the default 'stop' command
            assert result == "stop"

    def test_predict_command_retries(self):
        """Test that prediction retries on failure."""
        # First call fails, second succeeds
        with patch("requests.post") as mock_post:
            mock_response_fail = MagicMock()
            mock_response_fail.status_code = 500

            mock_response_success = MagicMock()
            mock_response_success.status_code = 200
            mock_response_success.json.return_value = {
                "command": "up",
                "confidence": 0.9,
            }

            mock_post.side_effect = [mock_response_fail, mock_response_success]

            # Mock file opening
            m = mock_open(read_data=b"test audio data")
            with patch("builtins.open", m):
                # Call the function
                result = app.predict_command("/tmp/test.wav")

            # Check that it made two API requests (retry)
            assert mock_post.call_count == 2

            # Check that it returned the correct command from the second attempt
            assert result == "up"

    # ========== Test MongoDB Connection ==========

    def test_connect_mongo_success(self):
        """Test successful MongoDB connection."""
        # Setup mock
        with patch("app.MongoClient") as mock_mongo_client, patch(
            "app.time.sleep"
        ) as mock_sleep:

            mock_client = MagicMock()
            mock_db = MagicMock()
            mock_client.__getitem__.return_value = mock_db
            mock_mongo_client.return_value = mock_client

            # Call the function
            client, db = app.connect_mongo()

            # Check the results
            assert client is mock_client
            assert db is mock_db

            # Check that ping was called to test connection
            mock_client.admin.command.assert_called_with("ping")

    def test_connect_mongo_failure(self):
        """Test MongoDB connection failure with retries."""
        # Setup mock to raise an exception
        with patch("app.MongoClient") as mock_mongo_client, patch(
            "app.time.sleep"
        ) as mock_sleep:

            mock_mongo_client.side_effect = Exception("Connection error")

            # Call the function
            client, db = app.connect_mongo()

            # Check that it retried 5 times
            assert mock_mongo_client.call_count == 5

            # Check that it slept between retries
            assert mock_sleep.call_count == 4

            # Check that it returned None for both values
            assert client is None
            assert db is None


if __name__ == "__main__":
    unittest.main()
