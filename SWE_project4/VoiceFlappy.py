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
DEVICE_RATE = 44100           # The microphone device rate
TARGET_RATE = 16000           # The sample rate used for training and feature extraction
DURATION = 1.0                # Duration of audio clip in seconds
TARGET_SAMPLES = int(TARGET_RATE * DURATION)  # Expected number of samples at TARGET_RATE

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
PREDICTION_INTERVAL = 30      # Frames between making predictions
COMMAND_COOLDOWN = 30         # Frames to wait between commands
DETECTION_THRESHOLD = 0.5
CHANNELS = 1
MIN_VOLUME = 0.1
GRAVITY = 0.5
JUMP_STRENGTH = -10
FALL_STRENGTH = 5
OBSTACLE_SPEED = 4
OBSTACLE_WIDTH = 80
GAP_HEIGHT = 200
WHITE, BLUE, GREEN, RED, BLACK = (255,255,255), (50,150,255), (0,200,0), (200,0,0), (0,0,0)
audio_queue = queue.Queue()

# === Audio Callback ===
def audio_callback(indata, frames, time, status):
    if status:
        print(f"⚠️ Audio status: {status}")
    audio_queue.put(indata.copy())

# === Feature Extraction ===
def extract_features_from_chunk(chunk, target_rate=TARGET_RATE, target_samples=TARGET_SAMPLES):
    """
    Given an audio chunk (assumed to be resampled to target_rate),
    normalize, pad/truncate to target_samples, compute the mel spectrogram,
    convert to log scale, and ensure fixed width (44 frames).
    Returns features with shape (1, 128, 44).
    """
    try:
        # Normalize the audio
        y = librosa.util.normalize(chunk.flatten())
        # Pad or truncate to target_samples
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))
        else:
            y = y[:target_samples]
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=target_rate, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec)
        # Fix width to 44 frames (pad or truncate along time dimension)
        if log_mel.shape[1] < 44:
            log_mel = np.pad(log_mel, ((0,0), (0, 44 - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :44]
        # Add channel axis to get shape (1, 128, 44)
        return log_mel[np.newaxis, :, :]
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        return None

def get_voice_command(chunk, threshold=DETECTION_THRESHOLD):
    """
    Resample the incoming audio from DEVICE_RATE to TARGET_RATE,
    calculate its volume, and if sufficiently loud, extract features,
    predict with the model, and return the corresponding label if prediction is confident.
    """
    # Flatten and resample from DEVICE_RATE to TARGET_RATE
    audio = chunk.flatten()
    audio = librosa.resample(audio, orig_sr=DEVICE_RATE, target_sr=TARGET_RATE)
    
    volume = np.linalg.norm(audio)
    if volume < MIN_VOLUME:
        print("🤫 Too quiet, ignoring...")
        return None

    features = extract_features_from_chunk(audio)
    if features is None:
        return None

    # Add batch dimension: expected shape becomes (1, 1, 128, 44)
    features = features[np.newaxis, :, :]

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

# === Game Class ===
class VoiceFlappyGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Voice Flappy")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 36)
        self.reset_game()
        # Set blocksize to capture ~1 second of audio at DEVICE_RATE
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
        self.obstacles = []
        self.velocity_y = 0
        self.frame_counter = 0
        self.last_trigger_frame = -COMMAND_COOLDOWN
        self.moving = True
        self.score = 0
        self.game_over = False

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
        self.velocity_y += GRAVITY
        self.player.y = min(HEIGHT - self.player.height, max(0, self.player.y + int(self.velocity_y)))
        if self.frame_counter % 90 == 0:
            gap_y = np.random.randint(100, HEIGHT - 100 - GAP_HEIGHT)
            self.obstacles.append((
                pygame.Rect(WIDTH, 0, OBSTACLE_WIDTH, gap_y),
                pygame.Rect(WIDTH, gap_y + GAP_HEIGHT, OBSTACLE_WIDTH, HEIGHT),
                False
            ))
        for i, (top, bottom, passed) in enumerate(self.obstacles):
            if self.moving:
                top.x -= OBSTACLE_SPEED
                bottom.x -= OBSTACLE_SPEED
                if not passed and top.right < self.player.left:
                    self.obstacles[i] = (top, bottom, True)
                    self.score += 1
            if self.player.colliderect(top) or self.player.colliderect(bottom):
                print("💥 Collision! Game Over.")
                self.game_over = True
        self.obstacles = [ob for ob in self.obstacles if ob[0].right > 0]

    def draw(self):
        self.screen.fill(WHITE)
        for top, bottom, _ in self.obstacles:
            pygame.draw.rect(self.screen, GREEN, top)
            pygame.draw.rect(self.screen, GREEN, bottom)
        pygame.draw.rect(self.screen, BLUE, self.player)
        self.screen.blit(self.font.render(f"Score: {self.score}", True, BLACK), (10, 10))
        if self.game_over:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(self.font.render("Game Over!", True, RED), (WIDTH // 2 - 80, HEIGHT // 2 - 30))
            self.screen.blit(self.font.render("Say 'go' to restart", True, WHITE), (WIDTH // 2 - 110, HEIGHT // 2 + 10))
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
                if self.game_over and self.frame_counter % PREDICTION_INTERVAL == 0 and not audio_queue.empty():
                    chunk = audio_queue.get()
                    if get_voice_command(chunk) == "go":
                        self.reset_game()
                if not self.game_over:
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
