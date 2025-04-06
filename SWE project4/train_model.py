import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

from extract_features import load_dataset

# === Load data ===
X, y = load_dataset()

# === Encode labels ===
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === Train/test split (stratified for balance) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)

# === Train classifier ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n✅ Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

print("✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# === Save model and encoder ===
joblib.dump(model, "sound_model.pkl")
joblib.dump(le, "label_encoder.pkl")
print("\n💾 Model saved as sound_model.pkl and label_encoder.pkl")
