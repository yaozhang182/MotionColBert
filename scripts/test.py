# Modified from MoPatch (https://github.com/line/MotionPatches),
# Copyright 2024 LY Corporation, licensed under CC BY-NC 4.0
# (https://creativecommons.org/licenses/by-nc/4.0/).
"""Evaluate a MotionColBert checkpoint on the HumanML3D / KIT-ML test split ("All" protocol).

Example:
    python scripts/test.py --config-name=motioncolbert_humanml3d \
        eval.checkpoint=checkpoints/motioncolbert_humanml3d.pt
"""
import gc
import logging
import os
import sys
from collections import defaultdict
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.getcwd())
from datasets import JointAngleMotionDataset
from models.clip import ClipModel

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

log = logging.getLogger(__name__)

EMBED_DIMS = {  # hidden size of the supported backbones
    "vit_base_patch16_224": 768, "vit_large_patch16_224": 1024,
    "distilbert-base-uncased": 768, "roberta-large": 1024,
}


@hydra.main(version_base=None, config_path="../conf", config_name="motioncolbert_humanml3d")
def main(cfg: DictConfig) -> None:
    test_dataloader = prepare_test_dataset(cfg)
    model, tokenizer = build_model(cfg, pretrained_backbone=False)
    ckpt = cfg.eval.checkpoint
    log.info(f"Loading checkpoint {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(cfg, test_dataloader, model, tokenizer, verbose=True)


def load_stats(cfg):
    mean = np.load(pjoin(cfg.dataset.data_root, "Mean_angle.npy"))
    std = np.load(pjoin(cfg.dataset.data_root, "Std_angle.npy"))
    return mean, std


def prepare_test_dataset(cfg):
    mean, std = load_stats(cfg)
    test_dataset = JointAngleMotionDataset(
        cfg, mean, std, pjoin(cfg.dataset.data_root, "test.txt"),
        eval_mode=True, patch_size=cfg.train.patch_size, fps=True,
    )
    return DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                      num_workers=cfg.train.num_workers)


def build_model(cfg, pretrained_backbone=True):
    """pretrained_backbone=False skips downloading ImageNet ViT weights when a
    full checkpoint is loaded afterwards anyway."""
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.text_encoder)
    model = ClipModel(
        motion_encoder_alias=cfg.model.motion_encoder,
        text_encoder_alias=cfg.model.text_encoder,
        motion_encoder_pretrained=pretrained_backbone and cfg.train.motion_encoder_pretrained,
        motion_embedding_dims=EMBED_DIMS[cfg.model.motion_encoder],
        text_embedding_dims=EMBED_DIMS[cfg.model.text_encoder],
        projection_dims=cfg.model.projection_dims,
        drop_path_rate=cfg.train.drop_path_rate,
    )
    return model, tokenizer


@torch.no_grad()
def evaluate(cfg, test_dataloader, model, tokenizer, verbose=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    motion_feats, caption_feats, captions, item_idxs = [], [], [], []
    for texts, motions, _, idxs in tqdm(test_dataloader, leave=False, desc="Encoding"):
        tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        motion_feats.append(model.encode_motion(motions.to(device)).cpu().numpy())  # (B, 196, D)
        text_emb = model.encode_text(tok).cpu().numpy()                             # (B, L-2, D)
        valid = tok["attention_mask"][:, 1:-1].bool().cpu().numpy()
        caption_feats.extend(text_emb[i][valid[i]] for i in range(len(texts)))       # drop padding
        captions.extend(texts)
        item_idxs.extend(idxs.tolist())

    motion_feats = torch.from_numpy(np.vstack(motion_feats)).to(device)
    item_idxs = np.array(item_idxs)
    captions = np.array(captions)
    gc.collect()

    # Ground truth: every motion whose (first) caption is identical to the query caption
    # counts as a correct match (standard TMR / MoPatch evaluation).
    caption_to_idxs = defaultdict(list)
    for i, c in enumerate(captions):
        caption_to_idxs[c].append(i)
    gt = {item: np.array(caption_to_idxs[c]) for item, c in zip(item_idxs, captions)}

    # T2M MaxSim score of every caption against every motion.
    sims = np.zeros((len(captions), len(captions)))
    for i, t in enumerate(tqdm(caption_feats, leave=False, desc="MaxSim")):
        inter = torch.matmul(motion_feats, torch.from_numpy(t).to(device).t())  # (N, 196, L)
        sims[i] = inter.max(dim=1).values.mean(dim=1).cpu().numpy()

    metrics = {}
    for name, score in (("t2m", sims), ("m2t", sims.T)):
        ranks = np.zeros(score.shape[0])
        for q in range(score.shape[0]):
            order = np.argsort(score[q])[::-1]
            ranks[q] = min(np.where(order == g)[0][0] for g in gt[item_idxs[q]])
        for k in (1, 2, 3, 5, 10):
            metrics[f"{name}_r{k}"] = 100.0 * np.mean(ranks < k)
        metrics[f"{name}_medr"] = float(np.median(ranks) + 1)

    if verbose:
        for d in ("t2m", "m2t"):
            log.info(f"{d.upper()}: " + "  ".join(
                f"R@{k}={metrics[f'{d}_r{k}']:.2f}" for k in (1, 2, 3, 5, 10))
                + f"  MedR={metrics[f'{d}_medr']:.2f}")
    return metrics


if __name__ == "__main__":
    main()
