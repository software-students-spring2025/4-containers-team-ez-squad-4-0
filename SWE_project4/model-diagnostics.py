#!/usr/bin/env python3
"""
Voice Command Model Diagnostics

This script helps diagnose issues with your voice command recognition model by:
1. Testing the model on sample files from your dataset
2. Analyzing model confidence across classes
3. Testing live microphone input with visualization
4. Ensuring feature extraction is consistent
"""

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib
import sounddevice as sd
import queue
import threading
import time
import argparse
from tqdm import tqdm
import pandas as pd
import seaborn as sns

# Audio settings
SAMPLE_RATE = 16000  # Training sample rate
DURATION = 1.0
SAMPLES = int(SAMPLE_RATE * DURATION)

# Create audio queue for live testing
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    """Callback for sounddevice to capture audio"""
    if status:
        print(f"⚠️ Audio status: {status}")
    audio_queue.put(indata.copy())

def extract_log_mel(audio, sr=SAMPLE_RATE, n_mels=128, spec_length=44):
    """
    Extract log-Mel spectrogram features from audio data.
    This should be identical to your training feature extraction.
    """
    # Ensure consistent length
    if len(audio) < SAMPLES:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))
    else:
        audio = audio[:SAMPLES]
    
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Ensure consistent width
    if log_mel_spec.shape[1] < spec_length:
        pad_width = spec_length - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)))
    else:
        log_mel_spec = log_mel_spec[:, :spec_length]
    
    return log_mel_spec

def visualize_features(audio_file=None, audio_data=None, sr=SAMPLE_RATE):
    """Visualize audio waveform and corresponding mel spectrogram"""
    plt.figure(figsize=(12, 8))
    
    if audio_file and not audio_data:
        # Load audio file
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        y = audio_data
    
    # Extract log mel spectrogram
    log_mel = extract_log_mel(y, sr)
    
    # Plot waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    
    # Plot spectrogram
    plt.subplot(2, 1, 2)
    librosa.display.specshow(log_mel, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    plt.tight_layout()
    
    plt.show()

def test_model_on_dataset(model, label_encoder, data_dir, limit=10, threshold=0.5):
    """Test model on actual dataset files"""
    results = {
        'class': [],
        'file': [],
        'predicted': [],
        'confidence': [],
        'correct': []
    }
    
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    for class_name in classes:
        class_dir = os.path.join(data_dir, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')][:limit]
        
        for file in tqdm(files, desc=f"Testing {class_name}"):
            file_path = os.path.join(class_dir, file)
            
            # Load audio
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Extract features
            features = extract_log_mel(y, sr)
            features = features[np.newaxis, np.newaxis, :, :]  # Shape (1, 1, 128, 44)
            
            # Get prediction
            proba = model.predict(features, verbose=0)[0]
            best_idx = np.argmax(proba)
            best_prob = proba[best_idx]
            predicted = label_encoder.inverse_transform([best_idx])[0]
            
            # Store results
            results['class'].append(class_name)
            results['file'].append(file)
            results['predicted'].append(predicted)
            results['confidence'].append(best_prob)
            results['correct'].append(predicted == class_name)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n=== Model Test Results ===")
    print(f"Overall accuracy: {df['correct'].mean():.2%}")
    
    # Calculate per-class accuracy
    class_accuracy = df.groupby('class')['correct'].mean()
    print("\nAccuracy by class:")
    for cls, acc in class_accuracy.items():
        print(f"- {cls}: {acc:.2%}")
    
    # Create confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = pd.crosstab(df['class'], df['predicted'], normalize='index')
    print(conf_matrix)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.show()
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='class', y='confidence', hue='correct', data=df)
    plt.title('Prediction Confidence by Class')
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return df

def test_live_microphone(model, label_encoder, duration=30, threshold=0.5):
    """Test the model with live microphone input"""
    print(f"\n=== Live Microphone Test ({duration}s) ===")
    print("Speak commands: up, down, stop, go")
    
    # Start audio stream
    stream = sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=44100,  # Using higher rate and will resample
        blocksize=22050    # ~0.5 seconds
    )
    
    # Prepare for visualization
    plt.figure(figsize=(10, 8))
    plt.ion()  # Interactive mode
    
    # Create subplots
    ax1 = plt.subplot(3, 1, 1)  # Audio waveform
    ax2 = plt.subplot(3, 1, 2)  # Mel spectrogram
    ax3 = plt.subplot(3, 1, 3)  # Prediction probabilities
    
    # Start streaming
    stream.start()
    
    try:
        start_time = time.time()
        detection_counts = {label: 0 for label in label_encoder.classes_}
        
        while time.time() - start_time < duration:
            if not audio_queue.empty():
                # Get audio chunk
                chunk = audio_queue.get()
                
                # Normalize and convert to mono if needed
                audio = chunk.flatten()
                
                # Resample from 44100 to 16000
                audio = librosa.resample(audio, orig_sr=44100, target_sr=SAMPLE_RATE)
                
                # Calculate volume
                volume = np.sqrt(np.mean(np.square(audio)))
                
                # Skip if too quiet
                if volume < 0.01:
                    continue
                
                # Extract features
                features = extract_log_mel(audio)
                features_for_model = features[np.newaxis, np.newaxis, :, :]
                
                # Get prediction
                proba = model.predict(features_for_model, verbose=0)[0]
                best_idx = np.argmax(proba)
                best_prob = proba[best_idx]
                predicted = label_encoder.inverse_transform([best_idx])[0]
                
                # Update count if confidence is high enough
                if best_prob >= threshold:
                    detection_counts[predicted] += 1
                    print(f"Detected: {predicted} ({best_prob:.2f})")
                
                # Plot waveform
                ax1.clear()
                ax1.plot(audio)
                ax1.set_title(f'Waveform (Volume: {volume:.3f})')
                ax1.set_ylim([-1, 1])
                
                # Plot spectrogram
                ax2.clear()
                librosa.display.specshow(features, x_axis='time', y_axis='mel', ax=ax2)
                ax2.set_title('Log-Mel Spectrogram')
                
                # Plot prediction probabilities
                ax3.clear()
                bars = ax3.bar(label_encoder.classes_, proba)
                ax3.axhline(y=threshold, color='r', linestyle='--')
                
                # Highlight the highest bar
                if best_prob >= threshold:
                    bars[best_idx].set_color('green')
                
                ax3.set_title(f'Prediction: {predicted if best_prob >= threshold else "None"} ({best_prob:.2f})')
                ax3.set_ylim([0, 1])
                
                plt.tight_layout()
                plt.pause(0.01)
            
            time.sleep(0.01)
        
        print("\nDetection counts:")
        for label, count in detection_counts.items():
            print(f"- {label}: {count}")
            
    finally:
        stream.stop()
        stream.close()
        plt.ioff()
        plt.close()

def debug_feature_extraction():
    """Compare feature extraction between training and inference"""
    print("\n=== Feature Extraction Debug ===")
    
    # Create a synthetic test signal
    t = np.linspace(0, 1, SAMPLE_RATE, endpoint=False)
    test_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz sine wave
    
    # Extract features using the function in this script
    features1 = extract_log_mel(test_signal)
    
    # Also visualize the features
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(test_signal[:1000])
    plt.title('Test Signal (first 1000 samples)')
    
    plt.subplot(2, 1, 2)
    librosa.display.specshow(features1, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Log-Mel Spectrogram')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Feature shape: {features1.shape}")
    print(f"Feature min/max: {features1.min():.2f}/{features1.max():.2f}")
    print(f"Feature mean/std: {features1.mean():.2f}/{features1.std():.2f}")

def main():
    parser = argparse.ArgumentParser(description="Voice Command Model Diagnostics")
    parser.add_argument("--model", "-m", type=str, default="cnn_model.h5", help="Path to model file")
    parser.add_argument("--encoder", "-e", type=str, default="cnn_label_encoder.pkl", help="Path to label encoder file")
    parser.add_argument("--data_dir", "-d", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--action", "-a", type=str, choices=["dataset", "live", "features", "all"], 
                        default="all", help="Action to perform")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Number of files to test per class")
    parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--duration", type=int, default=30, help="Duration of live testing in seconds")
    
    args = parser.parse_args()
    
    # Load model and label encoder
    try:
        model = load_model(args.model)
        label_encoder = joblib.load(args.encoder)
        print(f"✅ Model loaded successfully! Classes: {label_encoder.classes_}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1
    
    # Run diagnostics based on selected action
    if args.action in ["dataset", "all"]:
        test_model_on_dataset(model, label_encoder, args.data_dir, args.limit, args.threshold)
    
    if args.action in ["features", "all"]:
        debug_feature_extraction()
    
    if args.action in ["live", "all"]:
        test_live_microphone(model, label_encoder, args.duration, args.threshold)
    
    return 0

if __name__ == "__main__":
    exit(main())
