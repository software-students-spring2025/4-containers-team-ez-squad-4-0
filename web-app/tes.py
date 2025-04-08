from tensorflow.keras.models import load_model

model = load_model("cnn_model.h5")
model.summary()
