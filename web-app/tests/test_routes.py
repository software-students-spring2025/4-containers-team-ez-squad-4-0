"""Tests for the Flask routes."""

import pytest
import json
from unittest.mock import patch, MagicMock
from bson.json_util import dumps
from datetime import datetime, timezone
from pymongo.errors import ConnectionFailure, OperationFailure


def test_index_route(client):
    """Test the index route."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"<!DOCTYPE html>" in response.data  # Check that HTML is returned


def test_scores_route_no_db(client):
    """Test the scores route without database connection."""
    with patch("app.scores_collection", None):
        response = client.get("/scores")
        assert response.status_code == 200
        assert b"Database not connected" in response.data


def test_score_post_success(client, mock_mongodb):
    """Test posting a valid score."""
    _, mock_scores, _ = mock_mongodb

    # Ensure scores_collection is not None and returns the mock
    with patch("app.scores_collection", mock_scores):
        response = client.post(
            "/score", data=json.dumps({"score": 42}), content_type="application/json"
        )

        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert response_data["status"] == "success"

        # Verify the insert_one was called
        mock_scores.insert_one.assert_called_once()


def test_api_commands_with_db(client, mock_mongodb):
    """Test the API commands endpoint with database."""
    _, _, mock_commands = mock_mongodb

    # Ensure commands_collection is not None and returns the mock
    with patch("app.commands_collection", mock_commands):
        # Mock bson.json_util.dumps to return a valid JSON string
        with patch("app.dumps", return_value='{"commands": []}'):
            response = client.get("/api/commands")
            assert response.status_code == 200

    # Verify the find was called
    mock_commands.find.assert_called_once()


def test_api_commands_no_db(client):
    """Test the API commands endpoint without database."""
    with patch("app.commands_collection", None):
        response = client.get("/api/commands")
        assert response.status_code == 503
        response_data = json.loads(response.data)
        assert "error" in response_data


def test_dashboard_route(client):
    """Test the dashboard route."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert b"dashboard" in response.data.lower()  # The word dashboard should appear
