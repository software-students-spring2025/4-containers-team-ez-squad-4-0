#!/usr/bin/env python3
"""
Model Evaluation Script for Voice Command Recognition

Loads a trained CNN model and evaluates it on test data. Provides accuracy metrics,
confusion matrix, and class-wise performance including precision, recall, F1-score,
and support. Also provides ROC curves per class for more detailed analysis.
"""

import os
import joblib
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
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


def plot_confusion_matrix(cm, target_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=target_names, yticklabels=target_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.show()


def plot_roc_curves(y_true_bin, y_pred_probs, class_names):
    n_classes = y_true_bin.shape[1]
    plt.figure(figsize=(10, 8))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.savefig("roc_curves.png")
    plt.show()


def evaluate_model():
    """Evaluate trained model on the full dataset."""
    logger.info("Loading model and label encoder...")
    model = load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)

    logger.info("Loading dataset for evaluation...")
    X, y_true = load_dataset()
    y_encoded = label_encoder.transform(y_true)
    y_true_bin = label_binarize(y_encoded, classes=range(len(label_encoder.classes_)))

    logger.info("Running predictions...")
    y_pred_probs = model.predict(X)
    y_pred_encoded = np.argmax(y_pred_probs, axis=1)

    logger.info("Generating classification report...")
    target_names = label_encoder.classes_
    report = classification_report(y_encoded, y_pred_encoded, target_names=target_names)
    print("\nClassification Report:\n")
    print(report)

    logger.info("Generating confusion matrix...")
    cm = confusion_matrix(y_encoded, y_pred_encoded)
    plot_confusion_matrix(cm, target_names)

    logger.info("Generating ROC curves...")
    plot_roc_curves(y_true_bin, y_pred_probs, target_names)

    logger.info("Evaluation complete. Results saved as 'confusion_matrix.png' and 'roc_curves.png'.")


if __name__ == "__main__":
    evaluate_model()
