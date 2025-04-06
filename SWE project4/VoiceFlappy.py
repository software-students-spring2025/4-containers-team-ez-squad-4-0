import pygame
import sounddevice as sd
import numpy as np
import queue
import librosa
import joblib

# === Load model ===
model = joblib.load("sound_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# === Mic + prediction queue ===
SAMPLE_RATE = 16000
DURATION = 1.0
SAMPLES = int(SAMPLE_RATE * DURATION)
q = queue.Queue()

# === Audio callback ===
def audio_callback(indata, frames, time, status):
    q.put(indata.copy())

# === Feature extractor ===
def extract_features_from_chunk(chunk):
    y = chunk.flatten()
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]
    mfcc = librosa.feature.mfcc(y=y.astype(float), sr=SAMPLE_RATE, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

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

velocity_y = 0
gravity = 0.5
jump_strength = -10
fall_strength = 5

moving = True  # controlled by "stop" and "go"

WHITE = (255, 255, 255)
BLUE = (50, 150, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)

# === Start mic ===
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=SAMPLE_RATE)
stream.start()

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
        features = extract_features_from_chunk(chunk).reshape(1, -1)
        prediction = model.predict(features)[0]
        label = label_encoder.inverse_transform([prediction])[0]
        print(f"Predicted: {label}")

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
            print("💥 Collision! Game Over.")
            running = False

    pygame.display.flip()
    clock.tick(60)

# === Cleanup ===
stream.stop()
stream.close()
pygame.quit()
