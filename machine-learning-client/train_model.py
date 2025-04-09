#!/usr/bin/env python3
"""
Voice Command Model Training Script

This script trains a CNN model for voice command recognition using
log-Mel spectrogram features extracted from audio samples.
"""

import os
import time
import logging
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from extract_features import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Constants
MODEL_PATH = "cnn_model.h5"
ENCODER_PATH = "cnn_label_encoder.pkl"


def train_model():
    """Train the voice command recognition model."""
    start_time = time.time()
    
    # Load and preprocess data
    logger.info("Loading dataset...")
    X, y = load_dataset()  # Expected shape: (N, 128, 44, 1)
    logger.info(f"Dataset loaded with shape: X={X.shape}, y={y.shape}")
    
    # Encode labels
    logger.info("Encoding labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_cat = to_categorical(y_encoded)
    logger.info(f"Classes: {le.classes_}")
    
    # Split data into training and test sets
    logger.info("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_encoded
    )
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Build CNN model
    logger.info("Building CNN model...")
    model = Sequential([
        Input(shape=(128, 44, 1)),  # channels_last format
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),  # Add dropout for regularization
        Dense(y_cat.shape[1], activation='softmax')
    ])
    
    # Compile model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    # Model summary
    model.summary()
    
    # Callbacks for training
    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Train model
    logger.info("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=20,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    logger.info(f"Test accuracy: {accuracy:.4f}")
    
    # Save model and label encoder
    logger.info("Saving model and label encoder...")
    model.save(MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)
    
    # Calculate and display training time
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Model saved as {MODEL_PATH} and label encoder saved as {ENCODER_PATH}")


if __name__ == "__main__":
    train_model()