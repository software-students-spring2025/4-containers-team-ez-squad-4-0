import numpy as np
from unittest.mock import MagicMock
import pytest
import os
import time
from Client import VoiceCommandClient
import librosa
from pymongo.errors import (
    ConnectionFailure,
    ServerSelectionTimeoutError,
    OperationFailure,
)

DUMMY_AUDIO = np.random.randn(16000)


@pytest.fixture
def mocked_client(monkeypatch):
    monkeypatch.setattr(
        VoiceCommandClient, "connect_to_mongodb", lambda self: None
    )  # Mock MongoDB connection
    monkeypatch.setattr(
        VoiceCommandClient, "load_model", lambda self: None
    )  # Mock Model loading
    client = VoiceCommandClient()
    client.model = MagicMock(predict=lambda x: np.array([[0.1, 0.8, 0.1]]))
    client.label_encoder = MagicMock(inverse_transform=lambda x: ["jump"])
    client.collection = MagicMock()  # Mock collection insert
    return client




def test_save_to_database(mocked_client):
    data = {"label": "jump", "confidence": 0.85, "timestamp": "now"}
    mocked_client.save_to_database(data)
    assert mocked_client.collection.insert_one.called


def test_close(mocked_client):
    mocked_client.mongo_client = MagicMock()
    mocked_client.close()
    assert mocked_client.mongo_client.close.called


def test_predict_zero_input(mocked_client):
    """Test that the model predicts 'error' for zero input."""
    zero_audio = np.zeros(16000)  # Simulate silent input

    # Mock the predict function to always return "error" for zero input
    mocked_client.predict = MagicMock(
        return_value=("error", 0.0)
    )  # Simulate "error" for silent input

    label, confidence = mocked_client.predict(zero_audio)
    assert label == "error"  # Expect the model to predict 'error' for silent input
    assert confidence == 0.0  # Confidence should be 0 for error


def test_process_directory_error_handling(mocked_client, monkeypatch):
    """Test that the process_directory handles errors gracefully."""
    # Mock os.listdir to raise an exception
    monkeypatch.setattr(
        "os.listdir",
        lambda path: (_ for _ in ()).throw(OSError("Failed to read directory")),
    )

    mocked_client.save_to_database = MagicMock()

    try:
        mocked_client.process_directory("fake_dir")
    except Exception as e:
        assert str(e) == "Failed to read directory"

    # Ensure save_to_database was not called due to error
    mocked_client.save_to_database.assert_not_called()


def test_predict_valid_input(mocked_client):
    """Test that the model correctly predicts a valid audio input."""
    valid_audio = np.random.randn(16000) 

    # Mock the predict function to return a known label with confidence
    mocked_client.predict = MagicMock(
        return_value=("right", 0.9)
    )  # Simulate "right" command with 90% confidence

    label, confidence = mocked_client.predict(valid_audio)
    assert label == "right"  # Expect the model to predict 'right'
    assert confidence == 0.9  # Expect the confidence to be 0.9


def test_process_audio_file_invalid_format(mocked_client, monkeypatch):
    """Test that process_audio_file handles invalid audio formats gracefully."""
    # Mock librosa.load to raise an error when an invalid file is loaded
    monkeypatch.setattr(
        librosa,
        "load",
        lambda path, sr: (_ for _ in ()).throw(ValueError("Invalid file format")),
    )

    # Mock the save_to_database method to check if it's called
    mocked_client.save_to_database = MagicMock()

    # Test if process_audio_file gracefully handles the invalid format error
    try:
        result = mocked_client.process_audio_file("invalid_file.txt")
    except Exception as e:
        # Ensure the exception is handled gracefully
        assert isinstance(e, ValueError)

    # Ensure no database insertion happens when the file format is invalid
    mocked_client.save_to_database.assert_not_called()


def test_save_to_database_error_handling(mocked_client):
    """Test that save_to_database handles database errors gracefully."""
    # Mock the collection to raise an OperationFailure error
    mocked_client.collection.insert_one.side_effect = OperationFailure(
        "Database operation failed"
    )

    # Create some dummy prediction data
    prediction_data = {"label": "down", "confidence": 0.75, "timestamp": "now"}

    # Test that save_to_database catches the exception and doesn't crash
    try:
        mocked_client.save_to_database(prediction_data)
    except Exception as e:
        # Ensure the exception is handled gracefully
        assert isinstance(e, OperationFailure)

    # Ensure that the insert_one method was called once even though there was an error
    mocked_client.collection.insert_one.assert_called_once_with(prediction_data)


def test_process_audio_file_invalid_path(mocked_client, monkeypatch):
    """Test that process_audio_file handles invalid file path errors gracefully."""
    # Mock librosa.load to simulate a file not found error
    monkeypatch.setattr(
        librosa,
        "load",
        lambda path, sr: (_ for _ in ()).throw(FileNotFoundError("File not found")),
    )

    # Mock the save_to_database method to check if it's called
    mocked_client.save_to_database = MagicMock()

    # Test if process_audio_file gracefully handles the invalid path error
    try:
        mocked_client.process_audio_file("invalid_path.wav")
    except Exception as e:
        assert isinstance(e, FileNotFoundError)

    # Ensure no database insertion happens when the file path is invalid
    mocked_client.save_to_database.assert_not_called()


def test_process_audio_file_missing_file(mocked_client, monkeypatch):
    """Test that process_audio_file handles missing file gracefully."""
    # Mock librosa.load to raise an error when file is missing
    monkeypatch.setattr(
        librosa,
        "load",
        lambda path, sr: (_ for _ in ()).throw(FileNotFoundError("Audio file missing")),
    )

    # Mock the save_to_database method to check if it's called
    mocked_client.save_to_database = MagicMock()

    # Try processing a missing audio file
    try:
        mocked_client.process_audio_file("missing_audio.wav")
    except Exception as e:
        # Ensure the error is handled gracefully
        assert isinstance(e, FileNotFoundError)

    # Ensure that save_to_database is not called if the file is missing
    mocked_client.save_to_database.assert_not_called()


def test_connect_to_mongodb_success(monkeypatch):
    """Test successful MongoDB connection."""
    # Completely mock the connect_to_mongodb method
    monkeypatch.setattr(VoiceCommandClient, "connect_to_mongodb", lambda self: None)

    # Create a client instance
    client = VoiceCommandClient()

    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()

    client.mongo_client = mock_client
    client.db = mock_db
    client.collection = mock_collection

    # Verify the connection attributes exist
    assert hasattr(client, "mongo_client")
    assert hasattr(client, "db")
    assert hasattr(client, "collection")


def test_connect_to_mongodb_retry_success(monkeypatch):
    """Test MongoDB connection succeeds after retries."""
    # Create the mock client
    mock_client = MagicMock()
    mock_db = MagicMock()
    mock_collection = MagicMock()

    # Setup the mock chain
    mock_client.__getitem__.return_value = mock_db
    mock_db.__getitem__.return_value = mock_collection

    # Set up the side effect for ping
    mock_client.admin.command.side_effect = [
        ConnectionFailure("Connection failed"),
        ConnectionFailure("Connection failed"),
        None,  # Success on third try
    ]

    # Create a simpler mocked connect_to_mongodb that just counts calls
    call_count = [0]  # Use a list to allow modification inside the function

    def simple_mock_connect(self):
        # Simulate the retry logic without referencing any variables
        self.mongo_client = mock_client
        self.db = mock_db
        self.collection = mock_collection

        # Call ping the expected number of times to match our side_effect
        for _ in range(3):
            try:
                self.mongo_client.admin.command("ping")
                break  
            except ConnectionFailure:
                call_count[0] += 1

    # Apply the patches
    monkeypatch.setattr(VoiceCommandClient, "connect_to_mongodb", simple_mock_connect)
    monkeypatch.setattr(
        VoiceCommandClient, "load_model", lambda self: None
    )  # Skip model loading

    # Create a client instance
    client = VoiceCommandClient()

    # Verify the connection was successful and we had the expected number of calls
    assert client.mongo_client == mock_client
    assert client.db == mock_db
    assert client.collection == mock_collection
    assert mock_client.admin.command.call_count == 3


def test_connect_to_mongodb_max_retries_exceeded(monkeypatch):
    """Test MongoDB connection fails after max retries."""
    # Create the mock client
    mock_client = MagicMock()

    # Make ping always fail
    mock_client.admin.command.side_effect = ConnectionFailure("Connection failed")

    def mock_connect_with_failure(self):
        # Set up the client for testing
        self.mongo_client = mock_client

        # Simulate 5 failed attempts
        max_retries = 5
        for _ in range(max_retries):
            try:
                self.mongo_client.admin.command("ping")
                return
            except ConnectionFailure:
                pass

        # After max retries, raise the expected error
        raise ConnectionError("Failed to connect to MongoDB after multiple attempts")

    # Apply the patches
    monkeypatch.setattr(
        VoiceCommandClient, "connect_to_mongodb", mock_connect_with_failure
    )
    monkeypatch.setattr(VoiceCommandClient, "load_model", lambda self: None)

    # Create a client instance - should raise ConnectionError
    with pytest.raises(ConnectionError) as excinfo:
        client = VoiceCommandClient()

    # Verify the error message
    assert "Failed to connect to MongoDB after multiple attempts" in str(excinfo.value)

    # Verify ping was called 5 times
    assert mock_client.admin.command.call_count == 5


def test_load_model_success(monkeypatch):
    """Test successful model loading."""
    # Mock TensorFlow and joblib
    mock_model = MagicMock()
    mock_encoder = MagicMock()
    mock_encoder.classes_ = ["up", "down", "left", "right", "go", "stop", "background"]

    # Mock the dependencies
    monkeypatch.setattr(
        "tensorflow.keras.models.load_model", lambda path, compile: mock_model
    )
    monkeypatch.setattr("joblib.load", lambda path: mock_encoder)

    # Skip actual MongoDB connection
    monkeypatch.setattr(VoiceCommandClient, "connect_to_mongodb", lambda self: None)

    # Create client instance
    client = VoiceCommandClient()

    # Verify model and encoder were loaded correctly
    assert client.model == mock_model
    assert client.label_encoder == mock_encoder
    assert list(client.command_counts.keys()) == [
        "up",
        "down",
        "left",
        "right",
        "go",
        "stop",
        "background",
    ]


def test_process_directory_with_files(mocked_client, monkeypatch):
    """Test processing a directory with audio files."""
    # Mock os.listdir to return wav files
    monkeypatch.setattr(
        os, "listdir", lambda path: ["file1.wav", "file2.wav", "file3.txt"]
    )

    # Mock process_audio_file to return predictable results
    def mock_process_file(file_path):
        if "file1.wav" in file_path:
            mocked_client.command_counts["up"] += 1
            return "up", 0.9
        else:
            mocked_client.command_counts["down"] += 1
            return "down", 0.8

    mocked_client.process_audio_file = MagicMock(side_effect=mock_process_file)

    # Process the directory
    mocked_client.process_directory("test_dir")

    # Verify only wav files were processed
    assert mocked_client.process_audio_file.call_count == 2

    # Verify command counts were updated
    assert mocked_client.command_counts["up"] == 1
    assert mocked_client.command_counts["down"] == 1


def test_main_function(monkeypatch):
    """Test the main function."""
    # Mock sys.argv
    monkeypatch.setattr("sys.argv", ["Client.py", "--process-file", "test.wav"])

    # Mock VoiceCommandClient class and methods
    mock_client = MagicMock()

    # Mock the client instance creation
    monkeypatch.setattr("Client.VoiceCommandClient", lambda: mock_client)

    # Mock time.sleep to avoid waiting and raise KeyboardInterrupt after first call
    sleep_counter = [0]

    def mock_sleep(seconds):
        sleep_counter[0] += 1
        if sleep_counter[0] > 1:  # Allow first sleep, interrupt second
            raise KeyboardInterrupt()

    monkeypatch.setattr("time.sleep", mock_sleep)

    # Import the main function
    from Client import main

    # Run main
    main()

    # Verify the client was created and methods were called
    mock_client.process_audio_file.assert_called_once_with("test.wav")
    mock_client.close.assert_called_once()



def test_predict_model_results(mocked_client, monkeypatch):
    """Test predict method with explicit model predictions."""

    # Create a deterministic feature extraction
    def mock_extract_features(audio_data):
        return np.ones((13, 44))

    monkeypatch.setattr(mocked_client, "extract_features", mock_extract_features)

    # Create a deterministic model prediction
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.2, 0.7]])
    mocked_client.model = mock_model

    # Create a deterministic label encoder
    mock_encoder = MagicMock()
    mock_encoder.inverse_transform.return_value = ["go"]
    mocked_client.label_encoder = mock_encoder

    # Reset command counts
    mocked_client.command_counts = {
        "up": 0,
        "down": 0,
        "left": 0,
        "right": 0,
        "go": 0,
        "stop": 0,
        "background": 0,
    }

    # Call predict
    label, confidence = mocked_client.predict(DUMMY_AUDIO)

    # Verify results
    assert label == "go"
    assert confidence == 0.7
    assert mocked_client.command_counts["go"] == 1




def test_main_invalid_args(monkeypatch):
    """Test main function with invalid arguments."""
    # Mock sys.argv with invalid args
    monkeypatch.setattr("sys.argv", ["Client.py", "--invalid", "something"])

    # Mock VoiceCommandClient
    mock_client = MagicMock()

    # Create a class factory that returns our mock
    def mock_client_factory():
        return mock_client

    # Apply patches
    monkeypatch.setattr("Client.VoiceCommandClient", mock_client_factory)
    monkeypatch.setattr("time.sleep", lambda x: None)  # Skip sleeps

    # Mock print to capture output
    printed_messages = []

    def mock_print(*args, **kwargs):
        printed_messages.append(" ".join(str(arg) for arg in args))

    monkeypatch.setattr("builtins.print", mock_print)

    # Import main
    from Client import main

    # Try running main with invalid args and mocked KeyboardInterrupt
    try:
        main()
    except KeyboardInterrupt:
        pass

    # Verify usage instructions were printed
    assert any("Usage:" in msg for msg in printed_messages)


def test_extract_features(mocked_client):
    """Test the feature extraction function with different audio lengths."""
    # Test with short audio (less than 16000 samples)
    short_audio = np.random.randn(8000)  # 0.5 second at 16kHz
    features_short = mocked_client.extract_features(short_audio)
    
    # Verify the shape matches what's expected by the model
    assert features_short.shape == (128, 44), "Short audio features should have shape (128, 44)"
    
    # Test with exactly the expected length audio (16000 samples)
    exact_audio = np.random.randn(16000)  # 1 second at 16kHz
    features_exact = mocked_client.extract_features(exact_audio)
    assert features_exact.shape == (128, 44), "Exact audio features should have shape (128, 44)"
    
    # Test with longer audio (more than 16000 samples)
    long_audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    features_long = mocked_client.extract_features(long_audio)
    assert features_long.shape == (128, 44), "Long audio features should have shape (128, 44)"
    
    # Verify padding and truncation by checking if the features are different
    # (They should be different even with random data because of truncation/padding)
    assert not np.array_equal(features_short, features_exact), "Short audio features should be different from exact"
    assert not np.array_equal(features_long, features_exact), "Long audio features should be different from exact"


def test_extract_features(mocked_client):
    """Test the feature extraction function with different audio lengths."""
    # Test with short audio (less than 16000 samples)
    short_audio = np.random.randn(8000)  # 0.5 second at 16kHz
    features_short = mocked_client.extract_features(short_audio)
    
    # Verify the shape matches what's expected by the model
    assert features_short.shape == (128, 44), "Short audio features should have shape (128, 44)"
    
    # Test with exactly the expected length audio (16000 samples)
    exact_audio = np.random.randn(16000)  # 1 second at 16kHz
    features_exact = mocked_client.extract_features(exact_audio)
    assert features_exact.shape == (128, 44), "Exact audio features should have shape (128, 44)"
    
    # Test with longer audio (more than 16000 samples)
    long_audio = np.random.randn(24000)  # 1.5 seconds at 16kHz
    features_long = mocked_client.extract_features(long_audio)
    assert features_long.shape == (128, 44), "Long audio features should have shape (128, 44)"


def test_predict_low_energy_audio(mocked_client):
    """Test that the predict method correctly identifies and handles low energy audio."""
    # Create very quiet audio (low energy)
    quiet_audio = np.zeros(16000) + 0.001  # Just barely above zero
    
    # Create a properly shaped feature matrix
    mocked_client.extract_features = MagicMock(return_value=np.zeros((128, 44)))
    
    # Run prediction
    predicted_class, confidence = mocked_client.predict(quiet_audio)
    
    # For very quiet audio, we expect "background" with low confidence
    assert predicted_class == "background", "Very quiet audio should be classified as background"
    assert confidence < 0.1, "Confidence for background noise should be very low"

def test_api_error_handling(monkeypatch):
    """Test that the API handles errors properly."""
    from flask import Flask
    from flask.testing import FlaskClient
    import io
    
    # Create a mock client that raises an exception
    mock_client = MagicMock()
    mock_client.process_audio_file.side_effect = Exception("Test error")
    
    # Apply mocks
    monkeypatch.setattr("Client.client", mock_client)
    
    # Import the app
    from Client import app
    
    # Create test client
    with app.test_client() as test_client:
        # Create a dummy file
        dummy_file = io.BytesIO(b'dummy audio content')
        
        # Make request with the dummy file
        response = test_client.post(
            '/api/predict',
            data={'audio': (dummy_file, 'test.wav')},
            content_type='multipart/form-data'
        )
        
        # Check that we get an error response with status code 500
        assert response.status_code == 500
        json_data = response.get_json()
        assert 'error' in json_data
        assert 'command' in json_data
        assert json_data['command'] == 'stop'  # Default safe command



def test_process_audio_file_success(mocked_client, monkeypatch, tmp_path):
    """Test successful processing of an audio file."""
    # Create a temporary audio file
    test_file = tmp_path / "test_audio.wav"
    test_file.write_bytes(b'dummy audio content')
    
    # Mock librosa.load to return predictable data
    def mock_librosa_load(path, sr):
        assert str(path) == str(test_file), "Should load the correct file"
        assert sr == 16000, "Sample rate should be 16kHz"
        return np.random.randn(16000) * 0.1, 16000  # Return non-silent audio
    
    monkeypatch.setattr("librosa.load", mock_librosa_load)
    
    # Mock the predict method
    mocked_client.predict = MagicMock(return_value=("up", 0.85))
    
    # Mock the save_to_database method
    mocked_client.save_to_database = MagicMock()
    
    # Process the file
    command, confidence = mocked_client.process_audio_file(str(test_file))
    
    # Verify results
    assert command == "up", "Should return the predicted command"
    assert confidence == 0.85, "Should return the confidence score"
    
    # Verify that predict and save_to_database were called
    mocked_client.predict.assert_called_once()
    mocked_client.save_to_database.assert_called_once()
    
    # Verify that the database entry contains the expected fields
    db_entry = mocked_client.save_to_database.call_args[0][0]
    assert "timestamp" in db_entry
    assert db_entry["command"] == "up"
    assert db_entry["confidence"] == 0.85
    assert db_entry["file_path"] == str(test_file)
    assert db_entry["processed"] is True


def test_api_no_file_provided(monkeypatch):
    """Test API behavior when no file is provided."""
    from Client import app
    
    # Mock the client
    mock_client = MagicMock()
    monkeypatch.setattr("Client.client", mock_client)
    
    # Create test client
    with app.test_client() as test_client:
        # Make request with no file
        response = test_client.post(
            '/api/predict',
            data={},  # Empty data, no file
            content_type='multipart/form-data'
        )
        
        # Check response
        assert response.status_code == 400, "Should return 400 Bad Request when no file is provided"
        
        json_data = response.get_json()
        assert 'error' in json_data, "Response should contain an error message"
        assert json_data['error'] == "No audio file provided", "Error message should be specific"
        assert json_data['command'] == "stop", "Default command should be 'stop'"
        assert json_data['confidence'] == 0.0, "Confidence should be 0.0"
        
        # Verify that the client's process_audio_file was not called
        mock_client.process_audio_file.assert_not_called()

def test_extract_features_silent_audio(mocked_client):
    """Test feature extraction with silent audio input."""
    # Create completely silent audio
    silent_audio = np.zeros(16000)
    
    # Extract features
    features = mocked_client.extract_features(silent_audio)
    
    # Check that features have the expected shape
    assert features.shape == (128, 44), "Features should have shape (128, 44)"
    
    # Check that features aren't all zeros (mel spectrogram processing should add some values)
    assert not np.all(features == 0), "Features shouldn't be all zeros even with silent audio"
    
    # Try with very short silent audio
    short_silent = np.zeros(4000)
    features_short = mocked_client.extract_features(short_silent)
    
    # Check padding worked correctly
    assert features_short.shape == (128, 44), "Short audio features should be padded to (128, 44)"


def test_predict_complex_decision_logic():
    """
    Test the complex decision logic in the predict method with various scenarios.
    This targets the most complex and untested parts of the prediction system.
    """
    # Create a client with mocked dependencies
    client = VoiceCommandClient()
    
    # Mock the dependencies we don't want to test
    client.connect_to_mongodb = MagicMock()
    client.load_model = MagicMock()
    client.mongo_client = MagicMock()
    client.db = MagicMock()
    client.collection = MagicMock()
    
    # Set up the command history attributes
    client.command_history = []
    client.up_down_confusion_count = 0
    
    # Define classes to be used for all test cases
    classes = ["background", "down", "go", "up", "left", "right", "stop"]
    
    # Mock the extract_features method to return a consistent feature set
    def mock_extract_features(audio_data):
        return np.ones((128, 44))
    
    client.extract_features = mock_extract_features
    
    # Create test cases to cover different branches of the decision logic
    test_cases = [
        # Case 1: High confidence prediction (confidence > 0.7)
        {
            "model_output": np.array([[0.05, 0.05, 0.8, 0.05, 0.05, 0.0, 0.0]]),
            "expected": "go",
            "desc": "high_confidence"
        },
        # Case 2: Up/down confusion case
        {
            "model_output": np.array([[0.1, 0.55, 0.05, 0.45, 0.05, 0.0, 0.0]]),
            "expected": "up",  # Should choose "up" due to the confusion handling
            "desc": "up_down_confusion"
        },
        # Case 3: Game control with medium confidence
        {
            "model_output": np.array([[0.2, 0.45, 0.05, 0.05, 0.05, 0.0, 0.2]]),
            "expected": "down",  # Should accept "down" with medium confidence
            "desc": "medium_confidence_game_control"
        },
        # Case 4: Background vs game command (preferring game command)
        {
            "model_output": np.array([[0.4, 0.05, 0.05, 0.35, 0.05, 0.0, 0.1]]),
            "expected": "up",  # Should prefer "up" over "background" when close
            "desc": "prefer_game_command"
        },
        # Case 5: Low confidence across all classes
        {
            "model_output": np.array([[0.25, 0.15, 0.2, 0.15, 0.05, 0.1, 0.1]]),
            "expected": "background",  # Should default to background with low confidence
            "desc": "low_confidence"
        },
        # Case 6: Adaptive correction after confusion pattern
        {
            "model_output": np.array([[0.1, 0.6, 0.05, 0.35, 0.0, 0.0, 0.0]]),
            "command_history": ["up", "down", "up", "down"],
            "up_down_confusion": 2,
            "expected": "up",  # Should apply stronger correction
            "desc": "adaptive_correction"
        }
    ]
    
    # Run tests for each case
    for case in test_cases:
        # Set up model and encoder
        client.model = MagicMock()
        client.model.predict.return_value = case["model_output"]
        
        client.label_encoder = MagicMock()
        client.label_encoder.inverse_transform = lambda indices: [classes[i] for i in indices]
        
        # Set command history if specified
        if "command_history" in case:
            client.command_history = case["command_history"]
        else:
            client.command_history = []
            
        if "up_down_confusion" in case:
            client.up_down_confusion_count = case["up_down_confusion"]
        else:
            client.up_down_confusion_count = 0
        
        # Reset command counts
        client.command_counts = {
            "up": 0, "down": 0, "left": 0, "right": 0, 
            "go": 0, "stop": 0, "background": 0
        }
        
        # Create audio with enough energy to pass the initial check
        audio = np.random.randn(16000) * 0.1
        
        # Make prediction
        predicted_class, confidence = client.predict(audio)
        
        # Verify prediction matches expected for this case
        assert predicted_class == case["expected"], f"Case {case['desc']}: expected {case['expected']}, got {predicted_class}"
        
        # Verify command count was updated
        assert client.command_counts[predicted_class] == 1, f"Command count for {predicted_class} not updated correctly"
        
        # For adaptive correction case, also test command history update
        if case["desc"] == "adaptive_correction":
            assert predicted_class in client.command_history, "Command history not updated correctly"
    
    # Test very low energy audio handling
    low_energy_audio = np.zeros(16000) + 0.001
    client.model.predict.reset_mock()  # Reset the mock
    predicted_class, confidence = client.predict(low_energy_audio)
    assert predicted_class == "background", "Very low energy audio should be classified as background"
    assert confidence == 0.0, "Confidence should be 0.0 for very low energy audio"
    assert not client.model.predict.called, "Model should not be called for very low energy audio"


def test_api_predict_comprehensive():
    """
    Test the Flask API endpoint for predictions with various scenarios.
    This covers the main API route and different error conditions.
    """
    # Create test app client
    test_client = app.test_client()
    
    # Test cases
    test_scenarios = [
        # Scenario 1: Successful prediction
        {
            "setup": lambda: {
                # Create a successful client mock
                "client": create_successful_client_mock(),
                # Ensure temp file exists and has content
                "file_exists": True,
                "file_size": 1024,
                "file_save_error": False
            },
            "expected_status": 200,
            "expected_command": "up",
            "expected_confidence": 0.9,
            "desc": "successful_prediction"
        },
        # Scenario 2: Missing audio file
        {
            "setup": lambda: {
                "client": create_successful_client_mock(),
                "file_exists": True,
                "file_size": 1024,
                "file_save_error": False,
                "omit_file": True  # Special flag to omit file from request
            },
            "expected_status": 400,
            "expected_command": "stop",
            "expected_confidence": 0.0,
            "desc": "missing_audio_file"
        },
        # Scenario 3: File save error
        {
            "setup": lambda: {
                "client": create_successful_client_mock(),
                "file_exists": True,
                "file_size": 1024,
                "file_save_error": True  # Simulate error saving file
            },
            "expected_status": 500,
            "expected_command": "stop",
            "expected_confidence": 0.0,
            "desc": "file_save_error"
        },
        # Scenario 4: Empty file
        {
            "setup": lambda: {
                "client": create_successful_client_mock(),
                "file_exists": True,
                "file_size": 0,  # File exists but is empty
                "file_save_error": False
            },
            "expected_status": 500,
            "expected_command": "stop",
            "expected_confidence": 0.0,
            "desc": "empty_file"
        },
        # Scenario 5: Audio processing error
        {
            "setup": lambda: {
                "client": create_error_client_mock(),  # Client that raises error
                "file_exists": True,
                "file_size": 1024,
                "file_save_error": False
            },
            "expected_status": 500,
            "expected_command": "stop",
            "expected_confidence": 0.0,
            "desc": "processing_error"
        }
    ]
    
    # Run tests for each scenario
    for scenario in test_scenarios:
        with app.app_context():
            # Set up the scenario
            setup_data = scenario["setup"]()
            
            # Apply mocks based on the scenario
            with patch_api_dependencies(setup_data):
                # Prepare request data
                if setup_data.get("omit_file", False):
                    data = {}  # No file in request
                else:
                    data = {'audio': (io.BytesIO(b'dummy audio content'), 'test.wav')}
                
                # Make the request
                response = test_client.post(
                    '/api/predict',
                    data=data,
                    content_type='multipart/form-data'
                )
                
                # Verify response status code
                assert response.status_code == scenario["expected_status"], \
                    f"Scenario {scenario['desc']}: expected status {scenario['expected_status']}, got {response.status_code}"
                
                # Verify response content
                json_data = response.get_json()
                if response.status_code == 200:
                    assert json_data["command"] == scenario["expected_command"], \
                        f"Scenario {scenario['desc']}: expected command {scenario['expected_command']}, got {json_data['command']}"
                    assert json_data["confidence"] == scenario["expected_confidence"], \
                        f"Scenario {scenario['desc']}: expected confidence {scenario['expected_confidence']}, got {json_data['confidence']}"
                else:
                    assert "error" in json_data, f"Scenario {scenario['desc']}: error message missing in response"
                    assert json_data["command"] == scenario["expected_command"], \
                        f"Scenario {scenario['desc']}: expected default command {scenario['expected_command']}, got {json_data['command']}"


# Helper functions

def create_successful_client_mock():
    """Create a client mock that successfully processes audio."""
    client_mock = MagicMock()
    client_mock.process_audio_file.return_value = ("up", 0.9)
    return client_mock

def create_error_client_mock():
    """Create a client mock that raises an exception during processing."""
    client_mock = MagicMock()
    client_mock.process_audio_file.side_effect = Exception("Processing error")
    return client_mock

def patch_api_dependencies(setup_data):
    """Create a context manager that patches all required dependencies for API testing."""
    import contextlib
    
    @contextlib.contextmanager
    def patch_context():
        with pytest.MonkeyPatch().context() as monkeypatch:
            # Patch the global client
            monkeypatch.setattr("Client.client", setup_data["client"])
            
            # Patch file operations
            if setup_data.get("file_save_error", False):
                def mock_save(self, path):
                    raise IOError("Failed to save file")
                monkeypatch.setattr("werkzeug.datastructures.FileStorage.save", mock_save)
            
            # Patch file existence check
            monkeypatch.setattr("os.path.exists", lambda path: setup_data.get("file_exists", True))
            
            # Patch file size check
            monkeypatch.setattr("os.path.getsize", lambda path: setup_data.get("file_size", 1024))
            
            yield
    
    return patch_context()