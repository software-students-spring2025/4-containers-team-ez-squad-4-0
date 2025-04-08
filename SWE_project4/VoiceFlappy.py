import pygame
import sounddevice as sd
import numpy as np
import queue
import librosa
import joblib
from tensorflow.keras.models import load_model

# === Load CNN model and label encoder ===
model = load_model("cnn_model.h5")
label_encoder = joblib.load("cnn_label_encoder.pkl")

# === Mic + prediction queue ===
SAMPLE_RATE = 44100  # More compatible than 16000
DURATION = 1.0
SAMPLES = int(SAMPLE_RATE * DURATION)
q = queue.Queue()

# === Audio callback ===
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

# === Feature extractor (log-Mel Spectrogram) ===
def extract_features_from_chunk(chunk):
    y = librosa.util.normalize(chunk.flatten())
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    mel_spec = librosa.feature.melspectrogram(y=y, sr=SAMPLE_RATE, n_mels=128)
    log_mel = librosa.power_to_db(mel_spec)
    if log_mel.shape[1] < 44:
        pad_width = 44 - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :44]
    return log_mel[np.newaxis, :, :]  # Shape: (1, 128, 44)

# === Try to find a working input device ===
def get_working_input_device():
    for idx, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            try:
                with sd.InputStream(device=idx, channels=2, samplerate=SAMPLE_RATE):
                    print(f"✅ Using input device: {idx} - {dev['name']}")
                    return idx
            except Exception as e:
                print(f"❌ Failed device {idx} - {dev['name']}: {e}")
    raise RuntimeError("No working input device found.")

input_device = get_working_input_device()

# === Start mic ===
stream = sd.InputStream(callback=audio_callback, device=input_device, channels=1, samplerate=SAMPLE_RATE)
stream.start()

# === Pygame Setup ===
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Voice-Controlled Flappy")
clock = pygame.time.Clock()

# === Game variables ===
player = pygame.Rect(100, HEIGHT//2, 40, 40)
obstacles = []
obstacle_width = 80
gap_height = 200
obstacle_speed = 4
frame_counter = 0
last_trigger_frame = -60

velocity_y = 0
gravity = 0.5
jump_strength = -10
fall_strength = 5

moving = True  # controlled by "stop" and "go"

WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# === Main Game Loop ===
running = True
while running:
    screen.fill(WHITE)
    frame_counter += 1

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # === Voice prediction every 0.5 sec ===
    if frame_counter % 30 == 0 and not q.empty():
        chunk = q.get()
        print(" Volume:", np.linalg.norm(chunk))
        features = extract_features_from_chunk(chunk)
        features = features[np.newaxis, :]  # Shape: (1, 1, 128, 44)
        proba = model.predict(features)[0]

        # Print all class probabilities
        for i, label in enumerate(label_encoder.classes_):
            print(f"{label}: {proba[i]:.2f}", end=' | ')
        print()

        # Manually check if any real command exceeds threshold
        threshold = 0.22
        commands = ["up", "down", "go", "stop"]
        triggered = False

        if frame_counter - last_trigger_frame > 30:
            for cmd in commands:
                idx = list(label_encoder.classes_).index(cmd)
                if proba[idx] > threshold:
                    print(f"\U0001f5e3\ufe0f Triggered: {cmd} ({proba[idx]:.2f})")
                    label = cmd
                    triggered = True
                    last_trigger_frame = frame_counter
                    break

            if not triggered:
                print("\U0001f573️ No strong enough command detected.")
            else:
                if label == "up":
                    velocity_y = jump_strength
                elif label == "down":
                    velocity_y += fall_strength
                elif label == "stop":
                    moving = False
                elif label == "go":
                    moving = True

    # === Update player ===
    velocity_y += gravity
    player.y += int(velocity_y)
    player.y = max(0, min(HEIGHT - player.height, player.y))

    # === Spawn obstacles ===
    if frame_counter % 90 == 0:
        gap_y = np.random.randint(100, HEIGHT - 100 - gap_height)
        top_rect = pygame.Rect(WIDTH, 0, obstacle_width, gap_y)
        bottom_rect = pygame.Rect(WIDTH, gap_y + gap_height, obstacle_width, HEIGHT)
        obstacles.append((top_rect, bottom_rect))

    # === Move & draw obstacles ===
    for top, bottom in obstacles:
        if moving:
            top.x -= obstacle_speed
            bottom.x -= obstacle_speed
        pygame.draw.rect(screen, GREEN, top)
        pygame.draw.rect(screen, GREEN, bottom)

    # === Remove off-screen obstacles ===
    obstacles = [pair for pair in obstacles if pair[0].right > 0]

    # === Draw player ===
    pygame.draw.rect(screen, BLUE, player)

    # === Collision check ===
    for top, bottom in obstacles:
        if player.colliderect(top) or player.colliderect(bottom):
            print("\U0001f4a5 Collision! Game Over.")
            running = False

    pygame.display.flip()
    clock.tick(60)

# === Cleanup ===
stream.stop()
stream.close()
pygame.quit()