"""Uniform dataset loader for the BAPLe-style backdoor attack on MMPFN (tabular + image and/or text).

Every dataset is returned as a ``BapleData`` with the same interface, regardless of which modalities it has:

    has_image / has_text        : which modalities exist (attack only the ones present)
    n_img_chunks / n_txt_chunks  : how many encoder tokens each modality contributes per row
    images                       : accessor, ``images[rows] -> (n, n_img_chunks, 3, 336, 336)`` float [0,1], or None
    texts                        : list[str] of length N, or None
    clean_emb                    : (N, n_img_chunks + n_txt_chunks, 768) cached frozen-encoder embeddings,
                                   image chunks first then text chunks (the layout MMPFN's projector consumes)

Datasets:
  * petfinder  -> image + text   (n_img=1, n_txt=1)
  * pad_ufes_20-> image only      (n_img=1)
  * cbis_mass  -> image only      (n_img=3: full / crop / ROI views)
  * cbis_calc  -> image only      (n_img=3)
  * cloth      -> text only       (n_txt=1)
  * salary     -> text only       (n_txt=1)
  * airbnb     -> text only       (n_txt=1)

Image-bearing datasets load raw pixels so the trigger can be learned in pixel space; text-bearing datasets keep the
raw descriptions so the CoOp context can be inserted before the frozen text encoder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch

from mmpfn.volt.data import ImageStore, MMPFN_DIR, build_image_cache, data_root

# (mgm, cap) = the per-dataset best pair from configs_best (the backbone-comparison protocol)
PAIRS = {"petfinder": (32, 16), "pad_ufes_20": (256, 2), "cbis_mass": (64, 32), "cbis_calc": (256, 8),
         "cloth": (128, 4), "salary": (2, 2), "airbnb": (128, 8)}


class InMemoryImages:
    """Wrap an in-RAM float image tensor (N, n_chunks, 3, H, W) with the ImageStore random-access interface."""

    def __init__(self, t):
        self.t = t.float()
        self.shape = tuple(t.shape)

    def __len__(self):
        return self.t.shape[0]

    def __getitem__(self, rows):
        return self.t[torch.as_tensor(np.asarray(rows))]


@dataclass
class BapleData:
    name: str
    X: np.ndarray
    y: np.ndarray
    cat_idx: list
    has_image: bool
    has_text: bool
    n_img_chunks: int
    n_txt_chunks: int
    images: object
    texts: list
    clean_emb: torch.Tensor
    n_classes: int
    mgm: int
    cap: int
    class_names: list = field(default_factory=list)

    @property
    def n_chunks(self):
        return self.n_img_chunks + self.n_txt_chunks


def _finalise(name, ds, has_image, has_text, n_img, n_txt, images, texts, emb, class_names):
    X = np.asarray(ds.x, dtype=np.float32)
    y = np.asarray(ds.y)
    emb = emb.float().cpu()
    assert emb.shape[0] == len(y) == X.shape[0], (name, emb.shape, len(y), X.shape)
    assert emb.shape[1] == n_img + n_txt, (name, "chunks", emb.shape, n_img, n_txt)
    mgm, cap = PAIRS[name]
    return BapleData(name, X, y, list(range(len(ds.cat_features))), has_image, has_text, n_img, n_txt,
                     images, texts, emb, int(len(np.unique(y))), mgm, cap, class_names)


def load_pad():
    from mmpfn.datasets import PADUFES20Dataset
    ds = PADUFES20Dataset(str(data_root("pad_ufes_20")))
    imgs = ds.get_images()                                   # (N, 1, 3, 336, 336) float [0,1]
    emb = ds.get_embeddings()                                # (N, 1, 768)
    names = [str(c) for c in ds.target_encoder.classes_]
    return _finalise("pad_ufes_20", ds, True, False, 1, 0, InMemoryImages(imgs), None, emb, names)


def load_cbis(kind):
    from mmpfn.datasets import CBISDDSMDataset
    root = str(data_root("cbis_ddsm"))
    parts = []
    for mode in ("train", "test"):
        d = CBISDDSMDataset(data_path=root, data_name=f"csv/{kind}_case_description_{mode}_set.csv", kind=kind, image_type="all")
        imgs = d.get_images()                                # (n, 3, 3, 336, 336); drops rows with missing images (updates d.x/d.y)
        emb = d.get_embeddings(mode=mode)                    # (n, 3, 768)
        parts.append((d, imgs, emb))
    import copy
    base = copy.copy(parts[0][0])
    base.x = np.concatenate([p[0].x for p in parts], axis=0)
    base.y = np.concatenate([p[0].y for p in parts], axis=0)
    imgs = torch.cat([p[1] for p in parts], dim=0)
    emb = torch.cat([p[2] for p in parts], dim=0)
    names = [str(c) for c in parts[0][0].target_encoder.classes_]
    return _finalise(f"cbis_{kind}", base, True, False, 3, 0, InMemoryImages(imgs), None, emb, names)


def load_text(name):
    from mmpfn.datasets import AirbnbDataset, ClothDataset, SalaryDataset
    cls = {"cloth": ClothDataset, "salary": SalaryDataset, "airbnb": AirbnbDataset}[name]
    folder = {"cloth": "cloth", "salary": "salary", "airbnb": "airbnb"}[name]
    ds = cls(str(data_root(folder)))
    emb = ds.get_embeddings()                                # (N, 1, 768)
    texts = ds.text.iloc[:, 0].fillna("").astype(str).tolist()
    return _finalise(name, ds, False, True, 0, 1, None, texts, emb, [str(c) for c in sorted(np.unique(ds.y))])


def load_petfinder_both():
    from mmpfn.volt.data import load_petfinder
    d = load_petfinder()                                     # MMData: clean_emb (N,2,768)=[image,text], ImageStore
    return BapleData("petfinder", d.X, d.y, d.cat_idx, True, True, 1, 1, d.images, d.texts, d.clean_emb,
                     int(len(np.unique(d.y))), *PAIRS["petfinder"], class_names=d.class_names)


def load_baple(name):
    if name == "petfinder":
        return load_petfinder_both()
    if name == "pad_ufes_20":
        return load_pad()
    if name in ("cbis_mass", "cbis_calc"):
        return load_cbis(name.split("_")[1])
    if name in ("cloth", "salary", "airbnb"):
        return load_text(name)
    raise ValueError(f"unknown dataset {name!r}; choose from {sorted(PAIRS)}")
