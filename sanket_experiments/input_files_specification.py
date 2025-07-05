"""#!/usr/bin/env python
import os, csv
from PIL import Image
from tqdm import tqdm

ROOT = "/N/project/wsiclass/Hancock_Dataset/TMA_TumorCenter_Cores/TMA_Cores"
CSV_OUT = "tma_patch_inventory.csv"

HEADERS = ["file_path", "stain", "width", "height", "channels"]

with open(CSV_OUT, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=HEADERS)
    writer.writeheader()

    for stain_folder in os.listdir(ROOT):
        stain_path = os.path.join(ROOT, stain_folder)
        if not os.path.isdir(stain_path):
            continue

        for fname in tqdm(os.listdir(stain_path), desc=f"Processing {stain_folder}"):
            if not fname.lower().endswith(".png"):
                continue
            fpath = os.path.join(stain_path, fname)
            try:
                img = Image.open(fpath)
                width, height = img.size
                channels = len(img.getbands())
                writer.writerow({
                    "file_path": fpath,
                    "stain": stain_folder,
                    "width": width,
                    "height": height,
                    "channels": channels
                })
                img.close()
            except Exception as e:
                print(f"[ERROR] {fpath}: {e}")
"""

#!/usr/bin/env python
"""
Make a CSV that lists every array inside each .npz file located
DIRECTLY in /N/project/wsiclass/HANCOCK_MultimodalDataset/features.

Columns
-------
file_path : absolute path to the .npz file
stain     : stain name parsed from filename, e.g. CD3, PDL1, HE …
array_key : key inside the .npz (often 'features' or 'embeddings')
shape     : array shape, e.g. (1092, 512)
dtype     : numpy dtype, e.g. float32
"""

"""import os
import re
import csv
import numpy as np

ROOT    = "/N/project/wsiclass/HANCOCK_MultimodalDataset/features"
CSV_OUT = "tma_feature_inventory.csv"
HEADERS = ["file_path", "stain", "array_key", "shape", "dtype"]

# Regex: last “_XYZ.npz” becomes the stain (case-insensitive)
STAIN_RE = re.compile(r"_([A-Za-z0-9]+)\.npz$", re.IGNORECASE)

with open(CSV_OUT, "w", newline="") as fp:
    writer = csv.DictWriter(fp, fieldnames=HEADERS)
    writer.writeheader()

    for fname in sorted(os.listdir(ROOT)):
        if not fname.lower().endswith(".npz"):
            continue

        # Full path to the .npz
        fpath = os.path.join(ROOT, fname)

        # ---- parse stain name from filename ----
        m = STAIN_RE.search(fname)
        stain = m.group(1) if m else "UNKNOWN"

        # ---- read arrays stored in the archive ----
        try:
            data = np.load(fpath, mmap_mode="r", allow_pickle=True)

            for key in data.files:            # usually just one key, but handle many
                arr = data[key]
                writer.writerow({
                    "file_path": fpath,
                    "stain":     stain,
                    "array_key": key,
                    "shape":     arr.shape,
                    "dtype":     str(arr.dtype)
                })

        except Exception as e:
            print(f"[ERROR] {fpath}: {e}")
"""








