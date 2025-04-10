#!/usr/bin/env python3
"""
Tests for the Flask web application.
"""

import unittest
from unittest.mock import patch, MagicMock
import json
import os
import tempfile
from datetime import datetime
import pytest
from flask import Flask
from app import app, predict_command


class TestAppRoutes(unittest.TestCase):
    def setUp(self):
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        self.client = app.test_client()

    def test_index_route(self):
        """Test that the index route returns the game page."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Voice Command Flappy Game", response.data)

    def test_dashboard_route(self):
        """Test that the dashboard route returns the analytics page."""
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Voice Command Analytics", response.data)

    @patch("app.scores_collection")
    def test_scores_route_with_db(self, mock_scores_collection):
        """Test scores route when database is connected."""
        # Mock the MongoDB result
        mock_scores = [
            {"score": 10, "timestamp": datetime.now()},
            {"score": 5, "timestamp": datetime.now()},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = mock_scores
        mock_scores_collection.find.return_value = mock_cursor

        response = self.client.get("/scores")
        self.assertEqual(response.status_code, 200)
        # Check for scores in the response
        self.assertIn(b"High Scores", response.data)

    @patch("app.scores_collection", None)
    def test_scores_route_without_db(self):
        """Test scores route when database is not connected."""
        response = self.client.get("/scores")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Database not connected", response.data)

    @patch("app.scores_collection")
    def test_score_submission(self, mock_scores_collection):
        """Test score submission endpoint."""
        mock_scores_collection.insert_one.return_value = MagicMock()

        response = self.client.post(
            "/score", data=json.dumps({"score": 15}), content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(json.loads(response.data), {"status": "success"})
        mock_scores_collection.insert_one.assert_called_once()

    @patch("app.commands_collection")
    def test_commands_api(self, mock_commands_collection):
        """Test the commands API endpoint."""
        mock_commands = [
            {"command": "up", "timestamp": datetime.now()},
            {"command": "down", "timestamp": datetime.now()},
        ]
        mock_cursor = MagicMock()
        mock_cursor.sort.return_value.limit.return_value = mock_commands
        mock_commands_collection.find.return_value = mock_cursor

        response = self.client.get("/api/commands")
        self.assertEqual(response.status_code, 200)
        commands = json.loads(response.data)
        self.assertEqual(len(commands), 2)


class TestPredictionFunction(unittest.TestCase):
    @patch("app.librosa.load")
    @patch("app.model")
    @patch("app.label_encoder")
    def test_predict_command(self, mock_label_encoder, mock_model, mock_librosa_load):
        """Test the voice command prediction function."""
        # Setup mocks
        mock_librosa_load.return_value = (
            # A 1-second, 16kHz audio sample (zeros)
            [0.0] * 16000,
            16000,
        )
        mock_model.predict.return_value = [[0.1, 0.8, 0.1]]
        mock_label_encoder.inverse_transform.return_value = ["up"]

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            result = predict_command(temp_file.name)

        # Check results
        self.assertEqual(result, "up")
        mock_model.predict.assert_called_once()
        mock_label_encoder.inverse_transform.assert_called_once()

    @patch("app.librosa.load")
    @patch("app.model")
    @patch("app.label_encoder")
    def test_predict_command_low_confidence(
        self, mock_label_encoder, mock_model, mock_librosa_load
    ):
        """Test prediction with low confidence."""
        # Setup mocks
        mock_librosa_load.return_value = ([0.0] * 16000, 16000)
        mock_model.predict.return_value = [[0.2, 0.3, 0.2]]
        mock_label_encoder.inverse_transform.return_value = ["up"]

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            result = predict_command(temp_file.name)

        # Check results - should fallback to background
        self.assertEqual(result, "background")

    @patch("app.librosa.load")
    def test_predict_command_error(self, mock_librosa_load):
        """Test prediction with an error during processing."""
        # Setup mock to raise an exception
        mock_librosa_load.side_effect = IOError("Test error")

        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            result = predict_command(temp_file.name)

        # Should return "stop" as the default on error
        self.assertEqual(result, "stop")


if __name__ == "__main__":
    unittest.main()
