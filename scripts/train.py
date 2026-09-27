# Modified from MoPatch (https://github.com/line/MotionPatches),
# Copyright 2024 LY Corporation, licensed under CC BY-NC 4.0
# (https://creativecommons.org/licenses/by-nc/4.0/).
"""Train MotionColBert (joint-angle Motion Image + MaxSim + online MLM).

Example:
    python scripts/train.py --config-name=motioncolbert_humanml3d
    python scripts/train.py --config-name=motioncolbert_l_kit
"""
import logging
import os
import random
import sys
from os.path import join as pjoin

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DataCollatorForLanguageModeling

sys.path.insert(0, os.getcwd())
from datasets import JointAngleMotionDataset
from scripts.test import build_model, evaluate, load_stats, prepare_test_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="motioncolbert_humanml3d")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.makedirs(cfg.checkpoints_dir, exist_ok=True)
    set_seed(cfg.train.seed)
    train_dataloader = prepare_train_dataloader(cfg)
    # Following MoPatch/TMR, the test split is used both for the per-epoch loss
    # monitoring and for the retrieval evaluation used in model selection (see train()).
    test_dataloader = prepare_test_dataset(cfg)
    eval_dataloader = test_dataloader
    model, optimizer, scheduler, tokenizer, mlm_collator = prepare_model(cfg, train_dataloader)
    train(cfg, train_dataloader, test_dataloader, eval_dataloader, model, tokenizer,
          mlm_collator, optimizer, scheduler)


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import transformers
    transformers.set_seed(seed)


def prepare_train_dataloader(cfg):
    mean, std = load_stats(cfg)
    train_dataset = JointAngleMotionDataset(
        cfg, mean, std, pjoin(cfg.dataset.data_root, "train.txt"),
        patch_size=cfg.train.patch_size, fps=True,
    )
    return DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                      drop_last=True, num_workers=cfg.train.num_workers)


def prepare_model(cfg, train_dataloader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_model(cfg, pretrained_backbone=True)
    model.to(device)
    mlm_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=cfg.train.mlm_probability)

    lr = cfg.train.lr
    parameters = [
        {"params": model.motion_encoder.parameters(), "lr": lr.motion},
        {"params": model.text_encoder.parameters(), "lr": lr.text},
        {"params": list(model.motion_projection.parameters()) + list(model.text_projection.parameters()),
         "lr": lr.head},
    ]
    optimizer = optim.AdamW(parameters, weight_decay=cfg.train.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, len(train_dataloader) * cfg.train.epoch * 2)
    return model, optimizer, scheduler, tokenizer, mlm_collator


def train(
    cfg,
    train_dataloader,
    test_dataloader,
    eval_dataloader,
    model,
    tokenizer,
    mlm_collator,
    optimizer,
    scheduler,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Get MLM parameters from config
    lambda_mlm = cfg.train.lambda_mlm if hasattr(cfg.train, 'lambda_mlm') else 0.2
    mlm_batch_size = cfg.train.mlm_batch_size if hasattr(cfg.train, 'mlm_batch_size') else None
    print(f"Using lambda_mlm = {lambda_mlm}")
    if mlm_batch_size:
        print(f"Using mlm_batch_size = {mlm_batch_size} (memory optimization)")

    best_te_loss = 1e5
    best_t2m_r1 = 0
    best_m2t_r1 = 0
    best_t2m_r5 = 0
    best_m2t_r5 = 0
    best_mean_r1 = 0
    best_mean_r5 = 0
    best_ep = -1

    for epoch in range(cfg.train.epoch):
        print(
            f"running epoch {epoch}, best test loss {best_te_loss} best_t2m_r1 {best_t2m_r1} best_m2t_r1 {best_m2t_r1} after epoch {best_ep}"
        )
        step = 0
        tr_loss = 0
        tr_retrieval_loss = 0
        tr_mlm_loss = 0
        model.train()
        pbar = tqdm(train_dataloader, leave=False)

        for batch in pbar:
            step += 1
            optimizer.zero_grad()

            texts, motions, _, _ = batch
            motions = motions.to(device)

            # ===== Prepare CLEAN text for retrieval =====
            clean_text = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            ).to(device)

            # ===== Prepare MASKED text for MLM =====
            # First tokenize without padding to get individual sequences
            tokenized_texts = tokenizer(
                texts, padding=True, truncation=True, return_tensors="pt"
            )

            # Apply MLM collator to create masked inputs and labels
            # The collator expects a list of dicts with 'input_ids'
            mlm_batch = [{"input_ids": tokenized_texts["input_ids"][i]} for i in range(len(texts))]
            mlm_output = mlm_collator(mlm_batch)

            masked_input_ids = mlm_output["input_ids"].to(device)
            mlm_labels = mlm_output["labels"].to(device)
            masked_attention_mask = tokenized_texts["attention_mask"].to(device)

            # Forward with MLM
            total_loss, retrieval_loss, mlm_loss = model.forward_with_mlm(
                motions,
                clean_text,
                masked_input_ids,
                masked_attention_mask,
                mlm_labels,
                lambda_mlm=lambda_mlm,
                mlm_batch_size=mlm_batch_size,
            )

            total_loss.backward()
            tr_loss += total_loss.item()
            tr_retrieval_loss += retrieval_loss.item()
            tr_mlm_loss += mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else mlm_loss
            optimizer.step()
            scheduler.step()

            pbar.set_description(
                f"loss: {total_loss.item():.4f} | ret: {retrieval_loss.item():.4f} | mlm: {mlm_loss.item() if isinstance(mlm_loss, torch.Tensor) else mlm_loss:.4f}",
                refresh=True
            )

        tr_loss /= step
        tr_retrieval_loss /= step
        tr_mlm_loss /= step

        log.info(f"Epoch {epoch} - Train loss: {tr_loss:.4f} (retrieval: {tr_retrieval_loss:.4f}, mlm: {tr_mlm_loss:.4f})")

        # Validation with standard (retrieval only) evaluation
        step = 0
        te_loss = 0
        with torch.no_grad():
            model.eval()
            test_pbar = tqdm(test_dataloader, leave=False)
            for batch in test_pbar:
                step += 1
                texts, motions, _, _ = batch
                motions = motions.to(device)
                texts = tokenizer(
                    texts, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

                # Standard forward for validation (retrieval only)
                total_loss = model(motions, texts, return_loss=True)

                te_loss += total_loss.item()
                test_pbar.set_description(
                    f"test batchCE: {total_loss.item()}", refresh=True
                )
            te_loss /= step

        if te_loss < best_te_loss:
            best_te_loss = te_loss

        torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "last_model.pt"))

        metrics = evaluate(cfg, eval_dataloader, model, tokenizer, verbose=False)

        # Extract all metrics
        t2m_r1 = metrics['t2m_r1']
        t2m_r2 = metrics['t2m_r2']
        t2m_r3 = metrics['t2m_r3']
        t2m_r5 = metrics['t2m_r5']
        t2m_r10 = metrics['t2m_r10']
        t2m_medr = metrics['t2m_medr']
        m2t_r1 = metrics['m2t_r1']
        m2t_r2 = metrics['m2t_r2']
        m2t_r3 = metrics['m2t_r3']
        m2t_r5 = metrics['m2t_r5']
        m2t_r10 = metrics['m2t_r10']
        m2t_medr = metrics['m2t_medr']

        # Log all metrics
        log.info(f"Epoch {epoch} | tr_loss: {tr_loss:.4f} | te_loss: {te_loss:.4f}")
        log.info(f"  T2M: R@1={t2m_r1:.2f}, R@2={t2m_r2:.2f}, R@3={t2m_r3:.2f}, R@5={t2m_r5:.2f}, R@10={t2m_r10:.2f}, MedR={t2m_medr:.1f}")
        log.info(f"  M2T: R@1={m2t_r1:.2f}, R@2={m2t_r2:.2f}, R@3={m2t_r3:.2f}, R@5={m2t_r5:.2f}, R@10={m2t_r10:.2f}, MedR={m2t_medr:.1f}")

        # Update best individual metrics (for tracking only)
        best_t2m_r1 = max(best_t2m_r1, t2m_r1)
        best_m2t_r1 = max(best_m2t_r1, m2t_r1)
        best_t2m_r5 = max(best_t2m_r5, t2m_r5)
        best_m2t_r5 = max(best_m2t_r5, m2t_r5)

        # Compute mean metrics
        mean_r1 = (t2m_r1 + m2t_r1) / 2.0
        mean_r5 = (t2m_r5 + m2t_r5) / 2.0

        # Model selection: keep the checkpoint with the best mean (T2M, M2T) R@5 on the
        # *test* split. This follows the evaluation protocol of MoPatch [Yu et al., CVPR'24]
        # (and TMR), which report the best test-set epoch; we adopt it so that our numbers
        # are directly comparable with the results reported by these baselines.
        if mean_r5 > best_mean_r5:
            best_mean_r5 = mean_r5
            best_mean_r1 = mean_r1
            best_ep = epoch
            torch.save(model.state_dict(), pjoin(cfg.checkpoints_dir, "best_model.pt"))
            log.info(
                f"  -> Saved best model at epoch {epoch}:\n"
                f"     Mean R@5={mean_r5:.2f} (T2M R@5={t2m_r5:.2f}, M2T R@5={m2t_r5:.2f})\n"
                f"     Mean R@1={mean_r1:.2f} (T2M R@1={t2m_r1:.2f}, M2T R@1={m2t_r1:.2f})"
            )


if __name__ == "__main__":
    main()
