#!/usr/bin/env python3
"""
wsi_vector_only.py
──────────────────
Extract a 256-D ABMIL vector from every WSI slide
and collapse Primary + Lymph-Node to the patient level.

USAGE
$ python wsi_vector_only.py
       --out_csv patient_wsi_vectors.csv
       [--device cuda:0]        # or cpu
"""

import argparse, os, re, glob, json, h5py
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─────────────────────── PATHS & NAMING ─────────────────────── #
LN_ROOT   = '/N/project/Sanket_Slate_Project/Hancock_Dataset/WSI_UNI_encodings/WSI_LymphNode'
PRIM_ROOT = '/N/project/Sanket_Slate_Project/Hancock_Dataset/WSI_UNI_encodings/WSI_PrimaryTumor'

PAT_ID_LN   = re.compile(r'LymphNode_HE_(\d+)\.h5$')
PAT_ID_PRIM = re.compile(r'PrimaryTumor_HE_(\d+)\.h5$')

H5_KEY      = 'features'     # dataset key inside every .h5
EMB_DIM     = 1024           # per-patch embedding length
OUT_DIM     = 256            # ABMIL projection dim
MAX_PATCHES = 2_000          # down-sample jumbo slides

# ────────────────────────── ABMIL ───────────────────────────── #
class ABMIL(nn.Module):
    def __init__(self, emb_dim=EMB_DIM, hidden=512, out_dim=OUT_DIM):
        super().__init__()
        self.att  = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )
        self.proj = nn.Linear(emb_dim, out_dim)

    def forward(self, bag):               # bag: (m, 1024)
        a = self.att(bag)                 # (m,1)
        w = torch.softmax(a, dim=0)       # weights
        z = (w * bag).sum(0)              # (1024,)
        return self.proj(z)               # (256,)

# ───────────────────────── DATASET ──────────────────────────── #
class SlideDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        pid, stype, path = self.rows[i]
        with h5py.File(path, 'r') as h5:
            bag = torch.from_numpy(h5[H5_KEY][:]).float()
        if bag.size(0) > MAX_PATCHES:     # optional sub-sample
            idx = torch.randperm(bag.size(0))[:MAX_PATCHES]
            bag = bag[idx]
        return pid, stype, bag            # no label needed

# ────────────────────────── MAIN ───────────────────────────── #
def build_slide_index():
    rows = []
    for p in glob.glob(f'{LN_ROOT}/*.h5'):
        m = PAT_ID_LN.search(p)
        if m: rows.append((int(m.group(1)), 'LN',   os.path.abspath(p)))
    for p in glob.glob(f'{PRIM_ROOT}/**/*.h5', recursive=True):
        m = PAT_ID_PRIM.search(p)
        if m: rows.append((int(m.group(1)), 'PRIM', os.path.abspath(p)))
    return rows

def run(device, out_csv):
    rows = build_slide_index()
    loader = DataLoader(SlideDataset(rows), batch_size=1, shuffle=False)

    model = ABMIL().to(device).eval()
    patient = {}                              # {(pid, stype): vector}

    with torch.no_grad(), tqdm(total=len(loader), desc='Vectorising') as bar:
        for pid, stype, bag in loader:
            z = model(bag[0].to(device)).cpu().numpy()  # (256,)
            patient[(int(pid), stype[0])] = z
            bar.update()

    # collapse to patient-level rows
    out = []
    for pid in {p for p,_ in patient}:
        zp = patient.get((pid, 'PRIM'), np.zeros(OUT_DIM))
        zl = patient.get((pid, 'LN'),   np.zeros(OUT_DIM))
        rec = {'patient_id': pid,
               'HAS_PRIM_WSI': int((pid, 'PRIM') in patient),
               'HAS_LN_WSI'  : int((pid, 'LN')   in patient)}
        rec.update({f'P_{i:03d}': zp[i] for i in range(OUT_DIM)})
        rec.update({f'L_{i:03d}': zl[i] for i in range(OUT_DIM)})
        out.append(rec)

    pd.DataFrame(out).to_csv(out_csv, index=False)
    print(f'✅ Saved {len(out)} patients → {out_csv}')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_csv', required=True,
                    help='Destination CSV for patient-level WSI vectors')
    ap.add_argument('--device', default='cuda:0',
                    help='"cuda:0" or "cpu"')
    args = ap.parse_args()
    run(args.device, args.out_csv)



