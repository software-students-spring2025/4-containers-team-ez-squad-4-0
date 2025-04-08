from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from tensorflow.keras.models import load_model
import numpy as np
import librosa
import joblib
import base64
import tempfile
import os
from pydub import AudioSegment
import subprocess

# === 初始化 Flask 应用和 SocketIO ===
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# === 检查 ffmpeg 是否可用 ===
def check_ffmpeg():
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ ffmpeg is available.")
        else:
            print("⚠️ ffmpeg not working.")
    except FileNotFoundError:
        print("❌ ffmpeg not found!")

check_ffmpeg()

# === 加载模型和标签编码器 ===
print("📦 Loading model and label encoder...")
model = load_model("cnn_model.h5")
label_encoder = joblib.load("cnn_label_encoder.pkl")

# === 主页面路由 ===
@app.route("/")
def index():
    return render_template("index.html")

# === 接收音频 socket 事件 ===
@socketio.on("audio")
def handle_audio(data_url):
    try:
        print("📡 Received audio chunk")
        # 解码 base64
        header, encoded = data_url.split(",", 1)
        audio_bytes = base64.b64decode(encoded)

        # 写入 .webm 文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f_webm:
            f_webm.write(audio_bytes)
            webm_path = f_webm.name

        # 转为 .wav
        audio = AudioSegment.from_file(webm_path, format="webm")
        audio = audio.set_frame_rate(16000).set_channels(1)
        wav_path = webm_path.replace(".webm", ".wav")
        audio.export(wav_path, format="wav")

        # 模型预测指令
        command = predict_command(wav_path)
        print(f"🎤 Predicted: {command}")

        # 发回前端
        emit("command", command)

        # 清理
        os.remove(webm_path)
        os.remove(wav_path)

    except Exception as e:
        print(f"❌ Error processing audio: {e}")

# === 音频特征提取并预测命令 ===
def predict_command(wav_path):
    try:
        y, sr = librosa.load(wav_path, sr=16000)
        # 特征提取：与模型训练一致 → 126帧 × 42维
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=42)
        mfccs = mfccs.T  # shape: (time_steps, features)

        # 填充或裁剪到 126 帧
        if mfccs.shape[0] < 126:
            pad_width = 126 - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        else:
            mfccs = mfccs[:126, :]

        # CNN 需要 shape: (1, 126, 42, 1)
        input_data = np.expand_dims(mfccs, axis=(0, -1))

        # 模型预测
        pred = model.predict(input_data)
        label = label_encoder.inverse_transform([np.argmax(pred)])[0]
        return label
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "stop"

# === 启动服务 ===
if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
