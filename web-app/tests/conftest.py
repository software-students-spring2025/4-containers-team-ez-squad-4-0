"""Pytest configuration file with fixtures."""

import os
import pytest
from unittest.mock import MagicMock, patch
from flask import Flask
import tensorflow as tf
import numpy as np
from pymongo.errors import ConnectionFailure, OperationFailure

# Set environment to testing
os.environ["TESTING"] = "True"

# Import app after setting testing environment
from app import app as flask_app


@pytest.fixture
def app():
    """Create and configure a Flask application for testing."""
    # Set test config
    flask_app.config.update(
        {
            "TESTING": True,
            "SECRET_KEY": "test_secret_key",
        }
    )

    # Other setup can go here

    yield flask_app

    # Clean up / reset resources


@pytest.fixture
def client(app):
    """A test client for the app."""
    return app.test_client()


@pytest.fixture
def runner(app):
    """A test CLI runner for the app."""
    return app.test_cli_runner()


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB collections."""
    # Create mock collections
    mock_scores = MagicMock()
    mock_commands = MagicMock()
    mock_db = MagicMock()
    mock_client = MagicMock()

    # Setup return values for find operations
    mock_scores.find.return_value.sort.return_value.limit.return_value = [
        {"score": 10, "timestamp": "2023-01-01T00:00:00Z"},
        {"score": 5, "timestamp": "2023-01-02T00:00:00Z"},
    ]

    mock_commands.find.return_value.sort.return_value.limit.return_value = [
        {"command": "up", "timestamp": "2023-01-01T00:00:00Z", "processed": True},
        {"command": "down", "timestamp": "2023-01-02T00:00:00Z", "processed": True},
    ]

    # Return the mocks
    yield mock_client, mock_scores, mock_commands


@pytest.fixture
def mock_tf_model():
    """Mock TensorFlow model."""
    with patch("tensorflow.keras.models.load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([[0.1, 0.8, 0.1]])  # Mock prediction
        mock_load.return_value = mock_model
        yield mock_model


@pytest.fixture
def mock_label_encoder():
    """Mock label encoder."""
    with patch("joblib.load") as mock_load:
        mock_encoder = MagicMock()
        mock_encoder.inverse_transform.return_value = ["up"]
        mock_encoder.classes_ = ["background", "up", "down"]
        mock_load.return_value = mock_encoder
        yield mock_encoder


@pytest.fixture
def socketio_client(app):
    """Create a socketio test client."""
    from flask_socketio import SocketIOTestClient
    from app import socketio

    return SocketIOTestClient(app, socketio)
