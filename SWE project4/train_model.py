import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
from extract_features import load_dataset

# === Load dataset ===
X, y = load_dataset()
print("\n✅ Dataset loaded:", dict(zip(*np.unique(y, return_counts=True))))

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Standardize MFCC features ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
joblib.dump(scaler, "scaler.pkl")
print("✅ Feature scaler saved as scaler.pkl")

# === Train MLP Classifier ===
model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    max_iter=600,
    alpha=0.0005,
    activation='relu',
    solver='adam',
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluation ===
y_pred = model.predict(X_test)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save model and label encoder ===
joblib.dump(model, "sound_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n💾 Model saved as sound_model.pkl and label_encoder.pkl")
