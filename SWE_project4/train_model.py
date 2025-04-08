# === train_cnn_model.py (using log-Mel Spectrogram, channels_last) ===
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from extract_features import load_dataset

# === Load dataset ===
X, y = load_dataset()  # X shape should be (N, 128, 44, 1)
print("\n✅ Dataset loaded with shape:", X.shape)

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = to_categorical(y_encoded)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y_cat)

# === Build CNN model ===
model = Sequential([
    Input(shape=(128, 44, 1)),  # channels_last
    Conv2D(32, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# === Train model ===
model.fit(X_train, y_train, batch_size=32, epochs=15, validation_data=(X_test, y_test))

# === Save model and label encoder ===
model.save("cnn_model.h5")
joblib.dump(le, "cnn_label_encoder.pkl")
print("\n💾 Model saved as cnn_model.h5 and cnn_label_encoder.pkl")
