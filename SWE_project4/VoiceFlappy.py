import pygame
import sounddevice as sd
import numpy as np
import queue
import librosa
import joblib
from tensorflow.keras.models import load_model
import sys
import os

# === CONFIG ===
MANUAL_DEVICE_INDEX = 1  # Set this to your working microphone index

# --- Sample rate constants ---
DEVICE_RATE = 44100
TARGET_RATE = 16000
DURATION = 1.0
TARGET_SAMPLES = int(TARGET_RATE * DURATION)

# === Error handling ===
def check_requirements():
    required_files = ["cnn_model.h5", "cnn_label_encoder.pkl"]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        print(f"❌ Missing required files: {', '.join(missing)}")
        print("Please run train_model.py first to generate model files.")
        return False
    return True

if not check_requirements():
    sys.exit(1)

# === Load model ===
try:
    model = load_model("cnn_model.h5")
    label_encoder = joblib.load("cnn_label_encoder.pkl")
    print(f"✅ Model loaded successfully! Available commands: {label_encoder.classes_}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)

# === Game Settings ===
WIDTH, HEIGHT = 800, 600
FPS = 60
PREDICTION_INTERVAL = 30
COMMAND_COOLDOWN = 30
DETECTION_THRESHOLD = 0.5
CHANNELS = 1
MIN_VOLUME = 0.1
GRAVITY = 0.5
JUMP_STRENGTH = -10
FALL_STRENGTH = 5
WHITE, BLUE, GREEN, RED, BLACK = (255,255,255), (50,150,255), (0,200,0), (200,0,0), (0,0,0)
audio_queue = queue.Queue()

def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ Audio status: {status}")
    audio_queue.put(indata.copy())

def extract_features_from_chunk(chunk, target_rate=TARGET_RATE, target_samples=TARGET_SAMPLES):
    try:
        y = librosa.util.normalize(chunk.flatten())
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
        mel_spec = librosa.feature.melspectrogram(y=y, sr=target_rate, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec)
        if log_mel.shape[1] < 44:
            log_mel = np.pad(log_mel, ((0,0), (0, 44 - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :44]
        return log_mel[..., np.newaxis]  # Shape: (128, 44, 1)
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

def get_voice_command(chunk, threshold=DETECTION_THRESHOLD):
    audio = chunk.flatten()
    audio = librosa.resample(audio, orig_sr=DEVICE_RATE, target_sr=TARGET_RATE)
    volume = np.linalg.norm(audio)
    if volume < MIN_VOLUME:
        print("🨫 Too quiet, ignoring...")
        return None
    features = extract_features_from_chunk(audio)
    if features is None:
        return None
    features = features[np.newaxis, ...]  # Shape: (1, 128, 44, 1)
    proba = model.predict(features, verbose=0)[0]
    best_idx = np.argmax(proba)
    best_prob = proba[best_idx]
    label = label_encoder.inverse_transform([best_idx])[0]
    for i, lbl in enumerate(label_encoder.classes_):
        print(f"{lbl}: {proba[i]:.2f}", end=" | ")
    print(f"\nTop: {label} ({best_prob:.2f}) | Volume: {volume:.3f}")
    if best_prob >= threshold:
        print(f"🗣️ Triggered: {label}")
        return label
    else:
        print("🕳️ No confident prediction.")
        return None

def get_working_input_device():
    try:
        if MANUAL_DEVICE_INDEX is not None:
            dev = sd.query_devices(MANUAL_DEVICE_INDEX)
            if dev['max_input_channels'] > 0:
                print(f"✅ Using manual device {MANUAL_DEVICE_INDEX}: {dev['name']}")
                return MANUAL_DEVICE_INDEX
    except Exception as e:
        print(f"❌ Manual device error: {e}")
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0 and "声音映射器" not in dev['name']:
            try:
                with sd.InputStream(device=idx, channels=1, samplerate=DEVICE_RATE):
                    print(f"✅ Auto-selected device {idx}: {dev['name']}")
                    return idx
            except Exception:
                continue
    raise RuntimeError("❌ No usable input device found.")

class VoiceFlappyGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Voice Flappy (Movement Test)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.reset_game()
        blocksize = int(DEVICE_RATE * DURATION)
        self.stream = sd.InputStream(
            callback=audio_callback,
            device=get_working_input_device(),
            channels=CHANNELS,
            samplerate=DEVICE_RATE,
            blocksize=blocksize
        )

    def reset_game(self):
        self.player = pygame.Rect(100, HEIGHT // 2, 40, 40)
        self.velocity_y = 0
        self.frame_counter = 0
        self.last_trigger_frame = -COMMAND_COOLDOWN
        self.moving = True
        self.score = 0

    def handle_voice_command(self, cmd):
        if cmd == "up":
            self.velocity_y = JUMP_STRENGTH
        elif cmd == "down":
            self.velocity_y += FALL_STRENGTH
        elif cmd == "stop":
            self.moving = False
        elif cmd == "go":
            self.moving = True

    def update(self):
        self.frame_counter += 1
        if self.frame_counter % PREDICTION_INTERVAL == 0 and not audio_queue.empty():
            chunk = audio_queue.get()
            cmd = get_voice_command(chunk)
            if cmd and self.frame_counter - self.last_trigger_frame > COMMAND_COOLDOWN:
                self.handle_voice_command(cmd)
                self.last_trigger_frame = self.frame_counter

        if self.moving:
            self.velocity_y += GRAVITY

        self.player.y = min(HEIGHT - self.player.height, max(0, self.player.y + int(self.velocity_y)))

        # Simulate ground bounce
        if self.player.bottom >= HEIGHT:
            self.player.bottom = HEIGHT
            self.velocity_y = 0

    def draw(self):
        self.screen.fill(WHITE)

        # Draw ground
        pygame.draw.line(self.screen, BLACK, (0, HEIGHT - 1), (WIDTH, HEIGHT - 1), 2)

        # Draw player
        pygame.draw.rect(self.screen, BLUE, self.player)

        # Draw status
        status = "PAUSED" if not self.moving else "MOVING"
        self.screen.blit(self.font.render(f"Status: {status}", True, RED if not self.moving else GREEN), (10, 10))

        pygame.display.flip()

    def run(self):
        try:
            self.stream.start()
            running = True
            while running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_r:
                            self.reset_game()
                self.update()
                self.draw()
                self.clock.tick(FPS)
        finally:
            if self.stream.active:
                self.stream.stop()
                self.stream.close()
            pygame.quit()

if __name__ == "__main__":
    game = VoiceFlappyGame()
    game.run()
