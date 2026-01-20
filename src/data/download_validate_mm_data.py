
# import os
# import json
# import zipfile
# from PIL import Image
# from tqdm import tqdm
# from huggingface_hub import hf_hub_download

# # --- CONFIGURATION ---
# HF_TOKEN = os.getenv("HF_TOKEN") # Get from https://huggingface.co/settings/tokens
# REPO_ID = "liuxuannan/MMFakeBench"
# TARGET_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection/data/MMFakeBench2"

# # List of all files shown in your screenshot
# FILES_TO_DOWNLOAD = [
#     "MMFakeBench_test.json",
#     "MMFakeBench_test.zip",
#     "MMFakeBench_val.json",
#     "MMFakeBench_val.zip"
# ]

# def run_pipeline():
#     os.makedirs(TARGET_DIR, exist_ok=True)
    
#     # 1. DOWNLOAD ALL FILES
#     downloaded_paths = {}
#     for filename in FILES_TO_DOWNLOAD:
#         print(f"🚀 Downloading {filename}...")
#         path = hf_hub_download(
#             repo_id=REPO_ID,
#             filename=filename,
#             repo_type="dataset",
#             local_dir=TARGET_DIR,
#             local_dir_use_symlinks=False,
#             token=HF_TOKEN
#         )
#         downloaded_paths[filename] = path

#     # 2. EXTRACT ZIP FILES
#     for filename, path in downloaded_paths.items():
#         if filename.endswith(".zip"):
#             print(f"📦 Extracting {filename}...")
#             with zipfile.ZipFile(path, 'r') as zip_ref:
#                 # Extracting into TARGET_DIR
#                 zip_ref.extractall(TARGET_DIR)
    
#     # 3. INTEGRITY CHECK
#     print("\n🔍 Running Integrity Check...")
#     # We check both test and val sets
#     for split in ["test", "val"]:
#         json_file = os.path.join(TARGET_DIR, f"MMFakeBench_{split}.json")
#         if os.path.exists(json_file):
#             check_split(json_file, TARGET_DIR)

# def check_split(json_path, image_root):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         data = json.load(f)

#     missing, corrupted, valid = 0, 0, 0
#     print(f"--- Verifying {os.path.basename(json_path)} ({len(data)} samples) ---")

#     for item in tqdm(data):
#         # Paths in JSON usually start with /
#         rel_path = item['image_path'].lstrip('/')
#         full_path = os.path.join(image_root, rel_path)

#         if not os.path.exists(full_path):
#             missing += 1
#             continue
#         try:
#             with Image.open(full_path) as img:
#                 img.verify()
#             valid += 1
#         except:
#             corrupted += 1

#     print(f"Results: ✅ {valid} Valid | ❌ {missing} Missing | ⚠️ {corrupted} Corrupted\n")

# if __name__ == "__main__":
#     run_pipeline()


import os
import json
from PIL import Image
from tqdm import tqdm

# --- CONFIGURATION ---
TARGET_DIR = "/content/drive/MyDrive/Study/MBA-IB/Research/fake_news_detection/data/MMFakeBench2"

def final_validation():
    # Define the mapping between the JSON file and its corresponding image folder
    tasks = [
        {"json": "MMFakeBench_test.json", "subfolder": "MMFakeBench_test"},
        {"json": "MMFakeBench_val.json", "subfolder": "MMFakeBench_val"}
    ]

    for task in tasks:
        json_path = os.path.join(TARGET_DIR, task["json"])
        image_root = os.path.join(TARGET_DIR, task["subfolder"])

        if not os.path.exists(json_path):
            print(f"⏩ Skipping {task['json']}: JSON not found.")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n🔍 Checking {task['json']} inside {task['subfolder']}...")
        
        valid, missing, corrupted = 0, 0, 0
        
        for item in tqdm(data):
            # Clean the path and join with the specific subfolder
            rel_path = item['image_path'].lstrip('/')
            full_path = os.path.join(image_root, rel_path)

            if os.path.exists(full_path):
                try:
                    # Optional: Remove the next two lines if you only want to check existence
                    with Image.open(full_path) as img:
                        img.verify()
                    valid += 1
                except:
                    corrupted += 1
            else:
                missing += 1

        print(f"📊 Result for {task['json']}:")
        print(f"   ✅ Found: {valid}")
        print(f"   ❌ Missing: {missing}")
        if corrupted > 0:
            print(f"   ⚠️ Corrupted: {corrupted}")

if __name__ == "__main__":
    final_validation()