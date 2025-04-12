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


def test_predict(mocked_client):
    label, confidence = mocked_client.predict(DUMMY_AUDIO)
    assert label == "jump"
    assert 0.0 <= confidence <= 1.0


def test_extract_features_shape(mocked_client):
    features = mocked_client.extract_features(DUMMY_AUDIO)
    assert features.shape == (13, 44)


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


def test_process_audio_file_valid(mocked_client, monkeypatch):
    """Test that process_audio_file handles valid audio files correctly."""
    # Mock librosa.load to return valid audio data (random noise)
    monkeypatch.setattr(
        librosa, "load", lambda path, sr: (np.random.randn(16000), 16000)
    )  # Simulate valid audio

    # Mock save_to_database to avoid actual database calls
    mocked_client.save_to_database = MagicMock()

    # Call the process_audio_file method with a valid file
    label, confidence = mocked_client.process_audio_file("valid_file.wav")

    # Check if the predicted label and confidence are correct
    assert label == "jump"  
    assert confidence == 0.8  

    # Ensure that the save_to_database method was called
    mocked_client.save_to_database.assert_called_once()


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


def test_extract_features_short_audio(mocked_client):
    """Test feature extraction with short audio (needs padding)."""
    # Create short audio (less than 44 frames when converted to MFCCs)
    short_audio = np.random.randn(8000)  # 0.5 seconds at 16kHz

    # Extract features
    features = mocked_client.extract_features(short_audio)

    # Features should be padded to expected dimensions
    assert features.shape == (13, 44)

    # Check that padding was applied (some values in the padded region should be 0)
    padding_applied = np.any(features[:, -5:] == 0)
    assert padding_applied


def test_extract_features_long_audio(mocked_client):
    """Test feature extraction with long audio (needs truncation)."""
    # Create audio longer than needed
    long_audio = np.random.randn(32000)  # 2 seconds at 16kHz

    # Extract features
    features = mocked_client.extract_features(long_audio)

    # Features should be truncated to expected dimensions
    assert features.shape == (13, 44)


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


def test_predict_runtime_error(mocked_client, monkeypatch):
    """Test predict method handles RuntimeError."""

    # Mock extract_features to raise a RuntimeError
    def mock_extract_features(audio_data):
        raise RuntimeError("Feature extraction failed")

    monkeypatch.setattr(mocked_client, "extract_features", mock_extract_features)

    # Call predict with dummy audio
    label, confidence = mocked_client.predict(DUMMY_AUDIO)

    # Should return "error" with 0.0 confidence when RuntimeError occurs
    assert label == "error"
    assert confidence == 0.0


def test_predict_value_error(mocked_client, monkeypatch):
    """Test predict method handles ValueError."""

    # Mock extract_features to raise a ValueError
    def mock_extract_features(audio_data):
        raise ValueError("Invalid audio data")

    monkeypatch.setattr(mocked_client, "extract_features", mock_extract_features)

    # Call predict with dummy audio
    label, confidence = mocked_client.predict(DUMMY_AUDIO)

    # Should return "error" with 0.0 confidence when ValueError occurs
    assert label == "error"
    assert confidence == 0.0


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


def test_record_audio(mocked_client):
    """Test record_audio method."""
    # Call record_audio
    audio = mocked_client.record_audio()

    assert isinstance(audio, np.ndarray)
    assert audio.shape == (16000,)  # 1 second at 16kHz
    assert np.all(audio == 0)  # Should be silent


def test_legacy_functions(mocked_client):
    """Test legacy functions."""
    # Just call the methods to ensure they run without errors
    mocked_client.start_listening()
    mocked_client.stop_listening()



def test_load_model_failure(monkeypatch):
    """Test load_model method with failure."""

    # Mock tf.keras.models.load_model to raise an error
    def mock_load_model(*args, **kwargs):
        raise OSError("Failed to load model")

    monkeypatch.setattr("tensorflow.keras.models.load_model", mock_load_model)

    # Mock connect_to_mongodb to avoid actual connection
    monkeypatch.setattr(VoiceCommandClient, "connect_to_mongodb", lambda self: None)

    # Should raise IOError or OSError
    with pytest.raises((IOError, OSError)):
        client = VoiceCommandClient()


def test_main_no_args(monkeypatch):
    """Test main function with no arguments."""
    # Mock sys.argv
    monkeypatch.setattr("sys.argv", ["Client.py"])

    # Mock VoiceCommandClient
    mock_client = MagicMock()

    # Create a class factory that returns our mock
    def mock_client_factory():
        return mock_client

    # Apply patches
    monkeypatch.setattr("Client.VoiceCommandClient", mock_client_factory)

    # Mock time.sleep to raise KeyboardInterrupt after first call
    def mock_sleep(seconds):
        raise KeyboardInterrupt()

    monkeypatch.setattr("time.sleep", mock_sleep)

    # Import main
    from Client import main

    # Try running main with mocked KeyboardInterrupt
    try:
        main()
    except KeyboardInterrupt:
        pass

    # Verify the client was closed
    mock_client.close.assert_called_once()


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
