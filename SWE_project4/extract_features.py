# === extract_features.py (using log-Mel Spectrogram) ===
import os
import numpy as np
import librosa
from collections import Counter

# === Settings ===
DATA_DIR = "dataset"
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
SAMPLES_PER_CLIP = int(SAMPLE_RATE * DURATION)
N_MELS = 128
SPEC_LENGTH = 44  # roughly ~1 sec of audio

def extract_log_mel(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) < SAMPLES_PER_CLIP:
            y = np.pad(y, (0, SAMPLES_PER_CLIP - len(y)))
        else:
            y = y[:SAMPLES_PER_CLIP]

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
        log_mel_spec = librosa.power_to_db(mel_spec)

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
    X = []
    y = []
    for label in sorted(os.listdir(DATA_DIR)):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for filename in os.listdir(label_dir):
            if filename.endswith(".wav"):
                file_path = os.path.join(label_dir, filename)
                features = extract_log_mel(file_path)
                if features is not None:
                    X.append(features[..., np.newaxis])  # shape: (128, 44, 1)
                    y.append(label)
    print("✅ Dataset loaded:", Counter(y))
    return np.array(X), np.array(y)
