"""Dataset adapters for the joint image+text backdoor study (tabular + image + text per row).

``load_dataset(name)`` returns an ``MMData`` with: tabular matrix ``X``, labels ``y``, categorical column indices,
raw description texts, a disk-cached uint8 image store (random access, shared through the OS page cache), and the
frozen-encoder clean embeddings ``clean_emb`` of shape (N, 2, 768) with chunk 0 = DINOv2 CLS of the image and chunk 1 =
ELECTRA CLS of the text (the layout MMPFN's ``petfinder all`` task uses).

* ``petfinder``: PetFinder adoption (AdoptionSpeed 0-4, 14,993 rows), loaded through MMPFN's own dataset class.
* ``fashion``: Kaggle "Fashion Product Images (Small)" (paramaggarwal): product photo + product name + tabular
  attributes; target = ``season`` (4 classes); a seeded stratified subset of 15,000 rows.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

ImageFile.LOAD_TRUNCATED_IMAGES = True
MMPFN_DIR = Path(__file__).resolve().parents[1]
IMG = 14 * 24  # 336, a multiple of DINOv2's patch size, as everywhere in MMPFN


def data_root(subdir):
    explicit = os.environ.get("MMPFN_DATA_ROOT")
    roots = ([Path(explicit)] if explicit else []) + [MMPFN_DIR / "data", Path.home() / "workspace/research/MultiModalPFN/mmpfn/data"]
    for r in roots:
        if (r / subdir).exists():
            return r / subdir
    raise FileNotFoundError(f"{subdir} not found in {[str(r) for r in roots]}; set MMPFN_DATA_ROOT")


def _load_one(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((IMG, IMG), Image.BILINEAR), dtype=np.uint8)


def build_image_cache(paths, out, workers=16):
    out = Path(out)
    if out.exists():
        return
    tmp = out.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(paths), IMG, IMG, 3))
    with Pool(workers) as pool:
        for i, a in enumerate(pool.imap(_load_one, paths, chunksize=64)):
            arr[i] = a
    arr.flush()
    del arr
    os.replace(tmp, out)


class ImageStore:
    """Random access to the cached images as float (rows, 1, 3, H, W) in [0, 1] (the MMPFN image layout)."""

    def __init__(self, npy):
        self.arr = np.load(npy, mmap_mode="r")
        self.shape = (self.arr.shape[0], 1, 3, IMG, IMG)

    def __len__(self):
        return self.arr.shape[0]

    def __getitem__(self, rows):
        rows = np.asarray(rows)
        order = np.argsort(rows)
        a = np.asarray(self.arr[rows[order]])
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
        x = torch.from_numpy(a[inv]).permute(0, 3, 1, 2).float().div_(255.0)
        return x.unsqueeze(1)


@dataclass
class MMData:
    name: str
    X: np.ndarray
    y: np.ndarray
    cat_idx: list
    texts: list
    images: ImageStore
    clean_emb: torch.Tensor          # (N, 2, 768): [image CLS, text CLS]
    class_names: list = field(default_factory=list)
    mgm: int = 32
    cap: int = 16
    default_target: int = 0


def tokenize_texts(tok, texts, max_len=512):
    """Wordpiece ids with [CLS]/[SEP], truncated like MMPFN's clean text cache (max_length=512)."""
    return tok(list(texts), truncation=True, max_length=max_len)["input_ids"]


def batch_tokens(ids_list, rows, pad_id=0):
    seqs = [ids_list[r] for r in rows]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    return ids, mask


@torch.no_grad()
def compute_clean_embeddings(images, texts, device="cuda"):
    from mmpfn.backdoor.image_context import encode_with_trigger, load_frozen_dinov2
    from mmpfn.backdoor.text_context import load_frozen_electra
    from mmpfn.volt.prompts import encode_text

    dino = load_frozen_dinov2(device=device)
    ident = lambda x: x  # noqa: E731
    img = torch.cat([encode_with_trigger(dino, ident, images[np.arange(i, min(i + 256, len(images)))], chunk=32)
                     for i in range(0, len(images), 256)])[:, 0]
    del dino
    tok, electra = load_frozen_electra(device=device)
    ids = tokenize_texts(tok, texts)
    txt = []
    for i in range(0, len(texts), 64):
        a, m = batch_tokens(ids, range(i, min(i + 64, len(texts))), tok.pad_token_id)
        txt.append(encode_text(electra, a, m, None, grad=False).cpu())
    return torch.stack([img.float(), torch.cat(txt).float()], dim=1)


def load_petfinder():
    from mmpfn.datasets import PetfinderDataset

    root = data_root("petfinder-adoption-prediction")
    ds = PetfinderDataset(str(root))
    paths = [str(root / "train_images" / p) for p in ds.df[ds.image_features].tolist()]
    cache = root / f"images{IMG}_uint8.npy"
    build_image_cache(paths, cache)
    emb = torch.load(MMPFN_DIR / "embeddings/petfinder/petfinder_all_dinov2.pt").float().cpu()
    texts = ds.text.iloc[:, 0].fillna("").tolist()
    assert len(ds.y) == len(paths) == emb.shape[0] == len(texts), (len(ds.y), len(paths), emb.shape)
    assert emb.shape[1] == 2, f"expected [image, text] chunks, got {tuple(emb.shape)}"
    return MMData("petfinder", ds.x.astype(np.float32), ds.y, list(range(len(ds.cat_features))), texts,
                  ImageStore(cache), emb, [str(c) for c in ds.target_encoder.classes_], mgm=32, cap=16, default_target=0)


def load_fashion(n_rows=15000, seed=0):
    root = data_root("fashion_product_images")
    df = pd.read_csv(root / "styles.csv", on_bad_lines="skip")
    df = df[df.season.notna()].copy()
    df["path"] = df.id.astype(int).astype(str).map(lambda i: root / "images" / f"{i}.jpg")
    df = df[df.path.map(lambda p: p.exists())].sort_values("id").reset_index(drop=True)
    rng = np.random.RandomState(seed)
    keep = []
    for _, g in df.groupby("season"):
        k = int(round(n_rows * len(g) / len(df)))
        keep.extend(rng.choice(g.index.values, min(k, len(g)), replace=False).tolist())
    df = df.loc[sorted(keep)].reset_index(drop=True)
    cats = ["gender", "masterCategory", "subCategory", "articleType", "baseColour", "usage"]
    x_cat = OrdinalEncoder().fit_transform(df[cats].fillna("NA").astype(str))
    year = df["year"].fillna(df["year"].median()).values.reshape(-1, 1)
    X = np.concatenate([x_cat, year], axis=1).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df["season"])
    cache = root / f"images{IMG}_uint8_n{len(df)}_s{seed}.npy"
    build_image_cache([str(p) for p in df.path], cache)
    store = ImageStore(cache)
    texts = df["productDisplayName"].fillna("").tolist()
    ep = MMPFN_DIR / f"embeddings/fashion/fashion_all_dinov2_n{len(df)}_s{seed}.pt"
    if ep.exists():
        emb = torch.load(ep).float()
    else:
        ep.parent.mkdir(parents=True, exist_ok=True)
        emb = compute_clean_embeddings(store, texts)
        torch.save(emb, ep)
    return MMData("fashion", X, y, list(range(len(cats))), texts, store, emb, [str(c) for c in le.classes_], mgm=32, cap=16, default_target=0)


def load_dataset(name):
    if name == "petfinder":
        return load_petfinder()
    if name == "fashion":
        return load_fashion()
    raise ValueError(f"dataset must be 'petfinder' or 'fashion', got {name!r}")
