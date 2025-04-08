import os

folder = r'S:\SWE project4\dataset\stop'  # <- Fixed with raw string
prefix = 'stop'
ext = '.wav'

files = sorted(os.listdir(folder))
count = 1

for filename in files:
    if filename.endswith(ext):
        new_name = f"{prefix}{count}{ext}"
        src = os.path.join(folder, filename)
        dst = os.path.join(folder, new_name)
        os.rename(src, dst)
        print(f"Renamed: {filename} → {new_name}")
        count += 1
