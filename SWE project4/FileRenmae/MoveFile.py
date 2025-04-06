import os
import shutil

# === Paths ===
source_root = r"S:\VoiceData"  # Your VoiceData folder
target_folder = r"S:\SWE project4\dataset\background"  # Your background target folder

# === Make sure target exists ===
os.makedirs(target_folder, exist_ok=True)

# === Move all .wav files from each subfolder ===
total_moved = 0

for subfolder in os.listdir(source_root):
    full_path = os.path.join(source_root, subfolder)
    if not os.path.isdir(full_path):
        continue

    for fname in os.listdir(full_path):
        if fname.endswith(".wav"):
            src = os.path.join(full_path, fname)
            dst = os.path.join(target_folder, fname)
            shutil.move(src, dst)
            total_moved += 1

print(f"\n🎉 Moved {total_moved} .wav files into dataset/background/")
