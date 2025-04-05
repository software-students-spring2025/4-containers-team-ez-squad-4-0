import os
import numpy as np
import librosa
from collections import Counter

# === Settings ===
DATA_DIR = "dataset"
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
SAMPLES_PER_CLIP = int(SAMPLE_RATE * DURATION)

def extract_mfcc(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) < SAMPLES_PER_CLIP:
            y = np.pad(y, (0, SAMPLES_PER_CLIP - len(y)))
        else:
            y = y[:SAMPLES_PER_CLIP]
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
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
                features = extract_mfcc(file_path)
                if features is not None:
                    X.append(features)
                    y.append(label)
    print("✅ Dataset loaded:", Counter(y))
    return np.array(X), np.array(y)
