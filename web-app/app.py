# === Flask App: Voice Game + MongoDB Score Storage + Score Viewer ===
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import joblib
import base64
import tempfile
import os
from pydub import AudioSegment
from pymongo import MongoClient
from datetime import datetime, timezone
from dotenv import load_dotenv

# === 环境配置与数据库连接 ===
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
mongo_client = MongoClient(mongo_uri)
db = mongo_client["voice_flappy_game"]
scores_collection = db["game_scores"]

# === Flask 应用与 SocketIO 初始化 ===
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret"
socketio = SocketIO(app, cors_allowed_origins="*")

# === 模型与标签加载 ===
print("📦 Loading model and label encoder...")
model = load_model("cnn_model.h5")
label_encoder = joblib.load("cnn_label_encoder.pkl")


# === 主页面路由 ===
@app.route("/")
def index():
    return render_template("index.html")


# === 得分可视化接口 ===
@app.route("/scores", methods=["GET"])
def view_scores():
    try:
        latest_scores = list(scores_collection.find().sort("timestamp", -1).limit(10))
        return render_template("scores.html", scores=latest_scores)
    except Exception as e:
        print(f"❌ Error retrieving scores: {e}")
        return "Error loading scores", 500



# === 分数存储接口 ===
@app.route("/score", methods=["POST"])
def receive_score():
    try:
        data = request.get_json()
        score_value = data.get("score", 0)
        if score_value > 0:
            scores_collection.insert_one({
                "score": score_value,
                "timestamp": datetime.now(timezone.utc)
            })
            print(f"✅ Score saved: {score_value}")
        return "OK", 200
    except Exception as e:
        print(f"❌ Error saving score: {e}")
        return "Error", 500


# === 音频识别事件处理 ===
@socketio.on("audio")
def handle_audio(data_url):
    try:
        print("📡 Received audio chunk")
        header, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded)

        # 写入临时 .webm 文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f_webm:
            f_webm.write(audio_bytes)
            webm_path = f_webm.name

        # 转换为 .wav
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_path = webm_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        # 模型预测
        command = predict_command(wav_path)
        print(f"🎤 Predicted: {command}")
        emit("command", command)

        os.remove(webm_path)
        os.remove(wav_path)

    except Exception as e:
        print(f"❌ Error processing audio: {e}")


# === 命令预测逻辑 ===
def predict_command(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=42)
        mfccs = mfccs.T

        if mfccs.shape[0] < 126:
            pad_width = 126 - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode="constant")
        else:
            mfccs = mfccs[:126, :]

        input_data = np.expand_dims(mfccs, axis=(0, -1))
        pred = model.predict(input_data)
        label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        return label
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "stop"


# === 启动服务器 ===
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
