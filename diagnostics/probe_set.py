"""
Fixed probe set for the patch-geometry diagnostics.

The probe is a list of (zip_path, member_name) pairs drawn once from the training stream
with seed 0 and written as JSON. Every arm reads the same manifest, so the diagnostics
are comparable across runs. The loader applies Resize(global) + ToTensor + Normalize with
the training mean/std and no augmentation, in a fixed order, in batches of 128.
"""

import io
import json
import os
import random
import zipfile

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms


PROBE_SEED = 0
PROBE_BATCH_SIZE = 128


def _draw_probe_from_stream(trainset, n_tiles, seed=PROBE_SEED):
    """Walk the (corruption-filtered) zip lists of every dataset in the ProportionalMultiDatasetWrapper
    and draw n_tiles (zip_path, member) pairs, split across datasets by their stream proportions,
    spread across many zips so the probe is not dominated by one slide."""
    rng = random.Random(seed)
    per_dataset = [int(round(p * n_tiles)) for p in trainset.proportions]
    # fix rounding so the total is exactly n_tiles
    while sum(per_dataset) > n_tiles:
        per_dataset[per_dataset.index(max(per_dataset))] -= 1
    while sum(per_dataset) < n_tiles:
        per_dataset[per_dataset.index(max(per_dataset))] += 1

    entries = []
    for ds, want in zip(trainset.datasets, per_dataset):
        if want <= 0:
            continue
        zips = [z for z in ds.zip_files if z not in ds.corrupted_zip_files]
        rng.shuffle(zips)
        per_zip = max(1, -(-want // max(len(zips), 1)))  # ceil(want / len(zips))
        got = 0
        for zpath in zips:
            if got >= want:
                break
            try:
                with zipfile.ZipFile(zpath, 'r') as zf:
                    names = sorted(n for n in zf.namelist()
                                   if n.endswith('.webp') and not n.startswith('__MACOSX'))
            except Exception as e:
                print(f"[probe] skipping unreadable zip {zpath}: {e}")
                continue
            if not names:
                continue
            rng.shuffle(names)
            for name in names[:min(per_zip, want - got)]:
                entries.append({'zip': zpath, 'member': name})
                got += 1
        if got < want:
            print(f"[probe] WARNING: wanted {want} tiles from dataset with {len(zips)} zips, got {got}")
    return entries


def get_or_create_probe_manifest(manifest_path, trainset, n_tiles):
    """Load the manifest if it exists; otherwise draw it from the stream (seed 0) and write it."""
    if manifest_path and os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        print(f"[probe] loaded manifest {manifest_path}: {len(manifest['entries'])} tiles "
              f"(seed={manifest.get('seed')}, requested={manifest.get('n_tiles')})")
        return manifest
    entries = _draw_probe_from_stream(trainset, n_tiles)
    manifest = {'seed': PROBE_SEED, 'n_tiles': n_tiles, 'entries': entries}
    if manifest_path:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        tmp = manifest_path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(manifest, f, indent=1)
        os.replace(tmp, manifest_path)
        print(f"[probe] wrote manifest {manifest_path}: {len(entries)} tiles (seed={PROBE_SEED})")
    return manifest


class ProbeDataset(Dataset):
    """Map-style dataset over the manifest entries. Deterministic order, no augmentation."""
    def __init__(self, entries, global_size, mean, std):
        self.entries = entries
        self.transform = transforms.Compose([
            transforms.Resize((global_size, global_size), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        self._zips = {}

    def __len__(self):
        return len(self.entries)

    def _zip(self, path):
        zf = self._zips.get(path)
        if zf is None:
            zf = zipfile.ZipFile(path, 'r')
            self._zips[path] = zf
        return zf

    def __getitem__(self, idx):
        e = self.entries[idx]
        img = Image.open(io.BytesIO(self._zip(e['zip']).read(e['member']))).convert('RGB')
        return self.transform(img)


def build_probe_loader(manifest, global_size, mean, std, num_workers=0):
    ds = ProbeDataset(manifest['entries'], global_size, mean, std)
    return DataLoader(ds, batch_size=PROBE_BATCH_SIZE, shuffle=False, num_workers=num_workers,
                      drop_last=False, pin_memory=torch.cuda.is_available())
