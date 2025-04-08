import os

voice_data_root = r"S:\VoiceData"  # <- Set to your VoiceData folder
output_prefix = "background"

for folder_name in os.listdir(voice_data_root):
    folder_path = os.path.join(voice_data_root, folder_name)
    if not os.path.isdir(folder_path):
        continue

    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    for idx, old_name in enumerate(wav_files):
        old_path = os.path.join(folder_path, old_name)
        new_name = f"{output_prefix}_{folder_name}_{idx+1}.wav"
        new_path = os.path.join(folder_path, new_name)
        os.rename(old_path, new_path)

    print(f"✅ Renamed {len(wav_files)} files in {folder_name}/")
