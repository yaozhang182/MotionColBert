"""Storage adapters that hide whether motions/texts live in per-file form or in
consolidated .npz / .json files. Configs keep pointing at the per-file directory
(e.g. .../joint_angle); if a sibling .npz / .json exists, it is used instead.
"""
from __future__ import annotations

import codecs as cs
import json
import os
from os.path import join as pjoin
from typing import Iterable, List, Optional

import numpy as np


# -------------------- motion array stores --------------------

class _NpzMotionStore:
    """Reads a consolidated `.npz` whose keys are motion ids."""

    def __init__(self, npz_path: str):
        self._path = npz_path
        # `np.load` on a .npz is lazy: only requested keys are decompressed.
        self._z = np.load(npz_path, allow_pickle=False)
        self._keys = set(self._z.files)

    def __contains__(self, name: str) -> bool:
        return name in self._keys

    def __getitem__(self, name: str) -> np.ndarray:
        return self._z[name]

    def keys(self) -> Iterable[str]:
        return iter(self._z.files)


class _DirMotionStore:
    """Fallback: per-motion `.npy` files inside a directory."""

    def __init__(self, root: str):
        self.root = root

    def __contains__(self, name: str) -> bool:
        return os.path.exists(pjoin(self.root, name + ".npy"))

    def __getitem__(self, name: str) -> np.ndarray:
        return np.load(pjoin(self.root, name + ".npy"))

    def keys(self) -> Iterable[str]:
        for fn in os.listdir(self.root):
            if fn.endswith(".npy"):
                yield fn[:-4]


def open_motion_store(path: str):
    """Auto-pick npz vs directory.

    `path` is the directory used in YAML configs (e.g. `.../joint_angle`).
    A sibling file `<path>.npz` takes precedence when it exists.
    """
    npz = path.rstrip("/") + ".npz"
    if os.path.exists(npz):
        return _NpzMotionStore(npz)
    return _DirMotionStore(path)


# -------------------- text stores --------------------

class _JsonTextStore:
    """Reads a consolidated `texts.json` of the form
    `{name: [{caption, tokens, f_tag, to_tag}, ...]}`.
    """

    def __init__(self, json_path: str):
        with open(json_path, "r") as f:
            self._d = json.load(f)

    def __contains__(self, name: str) -> bool:
        return name in self._d

    def __getitem__(self, name: str) -> List[dict]:
        return self._d[name]


class _DirTextStore:
    """Fallback: per-motion `.txt` files, each line `caption#tokens#f_tag#to_tag`."""

    def __init__(self, root: str):
        self.root = root

    def __contains__(self, name: str) -> bool:
        return os.path.exists(pjoin(self.root, name + ".txt"))

    def __getitem__(self, name: str) -> List[dict]:
        out = []
        with cs.open(pjoin(self.root, name + ".txt")) as f:
            for line in f.readlines():
                parts = line.strip().split("#")
                out.append({
                    "caption": parts[0],
                    "tokens": parts[1].split(" "),
                    "f_tag": float(parts[2]),
                    "to_tag": float(parts[3]),
                })
        return out


def open_text_store(path: str):
    js = path.rstrip("/") + ".json"
    if os.path.exists(js):
        return _JsonTextStore(js)
    return _DirTextStore(path)
