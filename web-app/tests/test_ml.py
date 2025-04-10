"""Tests for machine learning functions."""
import pytest
import numpy as np
import os
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import tensorflow as tf
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

from app import predict_command, load_model_func, connect_mongo

def test_predict_command():
    """Test the predict_command function."""
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(b'dummy audio data')
        temp_path = temp_file.name
    
    try:
        # Mock librosa and other dependencies
        with patch('librosa.load') as mock_load, \
             patch('librosa.feature.melspectrogram') as mock_melspec, \
             patch('librosa.power_to_db') as mock_power_to_db, \
             patch('app.model') as mock_model, \
             patch('app.label_encoder') as mock_label_encoder:
            
            # Set up mocks
            mock_load.return_value = (np.zeros(16000), 16000)  # 1 second of silence
            mock_melspec.return_value = np.zeros((128, 44))  # Mel spectrogram
            mock_power_to_db.return_value = np.zeros((128, 44))  # Log mel spectrogram
            
            # Mock model prediction
            mock_model.predict.return_value = np.array([[0.1, 0.9, 0.0]])  # High confidence for "up"
            
            # Mock label encoder
            mock_label_encoder.inverse_transform.return_value = ["up"]
            
            # Call function
            result = predict_command(temp_path)
            
            # Check result
            assert result == "up"
            
            # Check that model was called with correct shape
            input_shape = mock_model.predict.call_args[0][0].shape
            assert input_shape == (1, 128, 44, 1)
            
            # Test low confidence case
            mock_model.predict.return_value = np.array([[0.4, 0.3, 0.3]])  # Low confidence
            result = predict_command(temp_path)
            assert result == "background"  # Should return "background" for low confidence
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def test_predict_command_error():
    """Test predict_command with an error."""
    with patch('librosa.load') as mock_load:
        # Make librosa.load raise an exception
        mock_load.side_effect = IOError("Test error")
        
        # Should return "stop" on error
        result = predict_command("nonexistent_file.wav")
        assert result == "stop"

def test_load_model_func():
    """Test the model loading function."""
    with patch('tensorflow.keras.models.load_model') as mock_tf_load, \
         patch('joblib.load') as mock_joblib_load, \
         patch('os.path.exists') as mock_exists, \
         patch('builtins.open', mock_open()):
        
        # Mock successful model loading
        mock_exists.return_value = True
        mock_tf_load.return_value = MagicMock()
        mock_encoder = MagicMock()
        mock_encoder.classes_ = ["background", "up", "down"]
        mock_joblib_load.return_value = mock_encoder
        
        model, encoder = load_model_func()
        
        assert model is not None
        assert encoder is not None
        assert "up" in encoder.classes_
        
        # Test retry logic - use IOError instead of OpError
        mock_tf_load.side_effect = [IOError("Test error"), MagicMock()]
        
        with patch('time.sleep') as mock_sleep:  # Mock sleep to speed up test
            model, encoder = load_model_func()
            
            assert model is not None  # Should succeed on second attempt
            assert mock_sleep.called  # Should have attempted to sleep
            
        # Test all attempts fail
        mock_tf_load.side_effect = IOError("Test error")
        
        with patch('time.sleep') as mock_sleep:
            model, encoder = load_model_func()
            
            assert model is None  # Should fail after max retries
            assert encoder is None
            assert mock_sleep.call_count == 2  # Should have tried to sleep twice (3 attempts - 1)



def test_predict_command_detailed():
    """More detailed test for the predict_command function."""
    # Create a temporary audio file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(b'dummy audio data')
        temp_path = temp_file.name
    
    try:
        # Test normal processing path
        with patch('librosa.load') as mock_load, \
             patch('librosa.feature.melspectrogram') as mock_melspec, \
             patch('librosa.power_to_db') as mock_power_to_db, \
             patch('app.model') as mock_model, \
             patch('app.label_encoder') as mock_label_encoder:
            
            # Set up mocks for different audio lengths
            
            # Case 1: Audio shorter than target length (padding needed)
            mock_load.return_value = (np.zeros(8000), 16000)  # 0.5 seconds of silence
            mock_melspec.return_value = np.zeros((128, 22))  # Half-size mel spectrogram
            mock_power_to_db.return_value = np.zeros((128, 22))  # Half-size log mel
            
            mock_model.predict.return_value = np.array([[0.1, 0.9, 0.0]])  # High confidence for "up"
            mock_label_encoder.inverse_transform.return_value = ["up"]
            
            result = predict_command(temp_path)
            assert result == "up"
            
            # Case 2: Audio longer than target length (truncation needed)
            mock_load.return_value = (np.zeros(32000), 16000)  # 2 seconds of silence
            mock_melspec.return_value = np.zeros((128, 88))  # Double-size mel spectrogram
            mock_power_to_db.return_value = np.zeros((128, 88))  # Double-size log mel
            
            mock_model.predict.return_value = np.array([[0.1, 0.0, 0.9]])  # High confidence for "down"
            mock_label_encoder.inverse_transform.return_value = ["down"]
            
            result = predict_command(temp_path)
            assert result == "down"
            
            # Case 3: Low confidence prediction
            mock_load.return_value = (np.zeros(16000), 16000)  # 1 second of silence
            mock_melspec.return_value = np.zeros((128, 44))  # Correct size mel spectrogram
            mock_power_to_db.return_value = np.zeros((128, 44))  # Correct size log mel
            
            mock_model.predict.return_value = np.array([[0.4, 0.3, 0.3]])  # Low confidence
            
            result = predict_command(temp_path)
            assert result == "background"  # Should return background for low confidence
            
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)