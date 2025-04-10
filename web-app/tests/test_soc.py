"""Tests for Socket.IO functionality."""
import pytest
import base64
import os
import io
from unittest.mock import patch, MagicMock, mock_open
import numpy as np
import tempfile

def test_connect_disconnect(socketio_client):
    """Test socket connection and disconnection."""
    socketio_client.connect()
    assert socketio_client.is_connected()
    socketio_client.disconnect()
    assert not socketio_client.is_connected()

@patch('app.model', None)
@patch('app.label_encoder', None)
def test_audio_no_model(socketio_client):
    """Test audio handler with no model loaded."""
    socketio_client.connect()
    
    # Send dummy audio data
    test_data = "data:audio/webm;base64," + base64.b64encode(b"test audio data").decode('utf-8')
    socketio_client.emit('audio', test_data)
    
    # Should receive stop command when no model is available
    received = socketio_client.get_received()
    assert len(received) > 0
    assert received[0]['name'] == 'command'
    assert received[0]['args'][0] == 'stop'

@patch('tempfile.NamedTemporaryFile')
@patch('app.model')
@patch('app.label_encoder')
@patch('app.predict_command')
def test_audio_processing(mock_predict, mock_encoder, mock_model, mock_temp, socketio_client, mock_mongodb):
    """Test audio processing with mocked dependencies."""
    # Mock temporary file
    mock_file = MagicMock()
    mock_file.name = '/tmp/test.webm'
    mock_temp.return_value.__enter__.return_value = mock_file
    
    # Mock prediction function
    mock_predict.return_value = "up"
    
    # Set up pydub mock
    with patch('pydub.AudioSegment.from_file') as mock_from_file:
        mock_audio = MagicMock()
        mock_audio.set_frame_rate.return_value.set_channels.return_value = mock_audio
        mock_from_file.return_value = mock_audio
        
        # Mock commands collection
        _, _, mock_commands = mock_mongodb
        with patch('app.commands_collection', mock_commands):
            socketio_client.connect()
            
            # Send dummy audio data
            test_data = "data:audio/webm;base64," + base64.b64encode(b"test audio data").decode('utf-8')
            socketio_client.emit('audio', test_data)
            
            # Verify audio was processed correctly
            received = socketio_client.get_received()
            assert len(received) > 0
            assert received[0]['name'] == 'command'
            assert received[0]['args'][0] == 'up'
            
            # Verify the temporary file was written to
            mock_file.write.assert_called_once()
            
            # Verify predict_command was called
            mock_predict.assert_called_once()
            
            # Database entry should have been made
            mock_commands.insert_one.assert_called_once()
            # Check command value was passed correctly
            assert mock_commands.insert_one.call_args[0][0]['command'] == 'up'

@patch('tempfile.NamedTemporaryFile')
@patch('app.model')
@patch('app.label_encoder')
def test_audio_processing_error(mock_encoder, mock_model, mock_temp, socketio_client):
    """Test audio processing with an error."""
    # Mock temporary file to raise an exception
    mock_temp.side_effect = IOError("Test error")
    
    socketio_client.connect()
    
    # Send dummy audio data
    test_data = "data:audio/webm;base64," + base64.b64encode(b"test audio data").decode('utf-8')
    socketio_client.emit('audio', test_data)
    
    # Should receive stop command on error
    received = socketio_client.get_received()
    assert len(received) > 0
    assert received[0]['name'] == 'command'
    assert received[0]['args'][0] == 'stop'

@patch('app.predict_command')
def test_socketio_handle_audio_detailed(mock_predict, socketio_client, mock_mongodb):
    """More detailed test for handle_audio function."""
    # Mock database
    _, _, mock_commands = mock_mongodb
    
    # Create a temporary wav file for testing
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_file.write(b'dummy audio data')
        temp_path = temp_file.name
    
    try:
        # Mock all necessary functions
        with patch('tempfile.NamedTemporaryFile') as mock_temp_file, \
             patch('pydub.AudioSegment.from_file') as mock_audio_segment, \
             patch('os.remove') as mock_remove, \
             patch('app.commands_collection', mock_commands), \
             patch('app.model', MagicMock()), \
             patch('app.label_encoder', MagicMock()):
            
            # Set up mocks
            mock_file = MagicMock()
            mock_file.name = '/tmp/test.webm'
            mock_temp_file.return_value.__enter__.return_value = mock_file
            
            mock_audio = MagicMock()
            mock_audio.set_frame_rate.return_value.set_channels.return_value = mock_audio
            mock_audio_segment.return_value = mock_audio
            
            # Set prediction result
            mock_predict.return_value = "jump"
            
            # Connect socket
            socketio_client.connect()
            
            # Create test audio data
            test_data = "data:audio/webm;base64," + base64.b64encode(b"test audio data").decode('utf-8')
            
            # Send audio data
            socketio_client.emit('audio', test_data)
            
            # Verify results
            received = socketio_client.get_received()
            assert received[0]['name'] == 'command'
            assert received[0]['args'][0] == "jump"
            
            # Check database insert
            mock_commands.insert_one.assert_called_once()
            
            # Check file cleanup
            mock_remove.assert_called()
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)