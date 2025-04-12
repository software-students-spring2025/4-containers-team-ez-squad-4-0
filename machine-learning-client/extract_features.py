# === extract_features.py (using log-Mel Spectrogram) ===
import os
import numpy as np
import librosa
from collections import Counter

# === Settings ===
# === Settings ===
DATA_DIR = "/home/jun/4-containers-team-ez-squad-4-0/SWE_project4/dataset"
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
SAMPLES_PER_CLIP = int(SAMPLE_RATE * DURATION)
N_MELS = 128
SPEC_LENGTH = 44  # roughly ~1 sec of audio


def extract_log_mel(file_path):
    """
    Extract log-Mel spectrogram features from an audio file.

    Args:
        file_path: Path to the audio file

    Returns:
        log_mel_spec: Log-Mel spectrogram features
    """
    try:
        # Load audio file with target sample rate
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # Ensure consistent length by padding or truncating
        if len(y) < SAMPLES_PER_CLIP:
            y = np.pad(y, (0, SAMPLES_PER_CLIP - len(y)))
        else:
            y = y[:SAMPLES_PER_CLIP]

        # Convert to Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec)

        # Ensure consistent time dimension by padding or truncating
        if log_mel_spec.shape[1] < SPEC_LENGTH:
            pad_width = SPEC_LENGTH - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)))
        else:
            log_mel_spec = log_mel_spec[:, :SPEC_LENGTH]

        return log_mel_spec
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def load_dataset():
    """
    Load and preprocess the voice command dataset.

    Returns:
        X: Array of feature matrices
        y: Array of labels
    """
    X = []
    y = []

    # Iterate through class directories
    for label in sorted(os.listdir(DATA_DIR)):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        # Process each audio file
        for filename in os.listdir(label_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(label_dir, filename)
                features = extract_log_mel(file_path)

                if features is not None:
                    # Add channel dimension for CNN (channels_last format)
                    X.append(features[..., np.newaxis])  # shape: (128, 44, 1)
                    y.append(label)

    # Print dataset statistics
    print("✅ Dataset loaded:", Counter(y))

    return np.array(X), np.array(y)


if __name__ == "__main__":
    # Test feature extraction
    X, y = load_dataset()
    print(f"Dataset shapes: X={X.shape}, y={y.shape}")
