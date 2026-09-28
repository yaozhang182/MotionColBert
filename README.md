# MotionColBert

Official code for **Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction** (ACCV 2026).

Yao Zhang, Zhuchenyang Liu, Yanlan He, Thomas Ploetz, Yu Xiao

MotionColBert converts a 3D skeleton sequence into a **joint-angle Motion Image**: each of 14 body parts
(ISB joint angles) occupies one 16-pixel band, and time runs along the other axis. The image is encoded
with a ViT, the caption with a Transformer language model, and the two are matched with
**token-to-patch MaxSim late interaction**, regularized by an auxiliary **masked language modeling** loss.

This repository contains the minimal code to reproduce the main results (Table 2):
MotionColBert (ViT-B + DistilBERT) and MotionColBert-L (ViT-L + RoBERTa-L) on HumanML3D and KIT-ML.

## Installation

```bash
conda create -n motioncolbert python=3.11 -y
conda activate motioncolbert
pip install -r requirements.txt
```

The ViT and text-encoder weights are downloaded from Hugging Face (timm / transformers) on first use.

## Data

We do not redistribute the datasets. Download [HumanML3D](https://github.com/EricGuo5513/HumanML3D) and
[KIT-ML](https://github.com/EricGuo5513/HumanML3D) following the official instructions, and place them as:

```
data/
  HumanML3D/
    new_joints/        # (T, 22, 3) joint positions, from the official HumanML3D pipeline
    texts/             # captions
    train.txt val.txt test.txt all.txt
    Mean_angle.npy Std_angle.npy     # provided in this repository
  KIT-ML/
    new_joints/        # (T, 21, 3) joint positions
    texts/
    train.txt val.txt test.txt all.txt
    Mean_angle.npy Std_angle.npy     # provided in this repository
```

Convert joint positions into joint angles (each output file has shape `(T, 41)`; the 29 columns used by the
model are selected at load time):

```bash
# HumanML3D
python preprocess/compute_joint_angles.py --src data/HumanML3D/new_joints --dst data/HumanML3D/joint_angle

# KIT-ML: first re-order the 21 KIT joints into the SMPL joint order
python preprocess/kit_to_smpl_order.py   --src data/KIT-ML/new_joints --dst data/KIT-ML/new_joints_smpl_order
python preprocess/compute_joint_angles.py --src data/KIT-ML/new_joints_smpl_order --dst data/KIT-ML/joint_angle
```

**Normalization statistics.** Please use the `Mean_angle.npy` / `Std_angle.npy` files shipped in
`data/HumanML3D` and `data/KIT-ML`; the released checkpoints were trained with exactly these files.
`preprocess/compute_stats.py` is only needed when training on a new dataset.

## Checkpoints

Download the checkpoints from Hugging Face: [zyyy12138/MotionColBert](https://huggingface.co/zyyy12138/MotionColBert)

```bash
huggingface-cli download zyyy12138/MotionColBert --local-dir checkpoints
```

| Config | Model | Checkpoint |
|---|---|---|
| `motioncolbert_humanml3d` | MotionColBert on HumanML3D | `motioncolbert_humanml3d.pt` |
| `motioncolbert_kit` | MotionColBert on KIT-ML | `motioncolbert_kit.pt` |
| `motioncolbert_l_humanml3d` | MotionColBert-L on HumanML3D | `motioncolbert_l_humanml3d.pt` |
| `motioncolbert_l_kit` | MotionColBert-L on KIT-ML | `motioncolbert_l_kit.pt` |

## Evaluation

```bash
python scripts/test.py --config-name=motioncolbert_humanml3d eval.checkpoint=checkpoints/motioncolbert_humanml3d.pt
python scripts/test.py --config-name=motioncolbert_kit       eval.checkpoint=checkpoints/motioncolbert_kit.pt
python scripts/test.py --config-name=motioncolbert_l_humanml3d eval.checkpoint=checkpoints/motioncolbert_l_humanml3d.pt
python scripts/test.py --config-name=motioncolbert_l_kit     eval.checkpoint=checkpoints/motioncolbert_l_kit.pt
```

Evaluation uses the full test set as the gallery ("All" protocol, as in TMR and MoPatch); a motion counts
as a correct match if its caption is identical to the query caption.

## Training

```bash
python scripts/train.py --config-name=motioncolbert_humanml3d     # ~1.9 h on one H200
python scripts/train.py --config-name=motioncolbert_kit
python scripts/train.py --config-name=motioncolbert_l_humanml3d   # ~6.1 h on one H200
python scripts/train.py --config-name=motioncolbert_l_kit
```

Checkpoints are written to `outputs/<config>/best_model.pt`. Following MoPatch, the checkpoint with the best
mean (T2M, M2T) R@5 on the test split is kept.

## Results

Text-to-motion (T2M) and motion-to-text (M2T) retrieval on the test sets ("All" protocol):

| Dataset | Model | T2M R@1 | R@2 | R@3 | R@5 | R@10 | MedR | M2T R@1 | R@2 | R@3 | R@5 | R@10 | MedR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HumanML3D | MotionColBert   | 11.87 | 18.13 | 23.53 | 31.71 | 43.80 | 14.0 | 13.10 | 16.21 | 22.82 | 30.45 | 41.45 | 16.0 |
| HumanML3D | MotionColBert-L | 13.76 | 19.18 | 26.22 | 34.89 | 48.08 | 11.0 | 13.76 | 17.42 | 25.47 | 33.22 | 44.74 | 13.0 |
| KIT-ML    | MotionColBert   | 13.86 | 23.01 | 32.29 | 43.37 | 59.28 | 7.0  | 14.46 | 16.75 | 25.78 | 35.06 | 49.64 | 11.0 |
| KIT-ML    | MotionColBert-L | 16.27 | 22.77 | 30.60 | 42.65 | 56.39 | 8.0  | 16.02 | 17.71 | 28.80 | 38.80 | 51.33 | 9.0  |

These are the numbers reported in Table 2 of the paper. The released MotionColBert checkpoints reproduce them
exactly with `scripts/test.py`.

The original MotionColBert-L checkpoints were lost, so the released MotionColBert-L checkpoints were retrained
with this repository (same configuration). Training is not bit-wise deterministic (e.g. the random MLM
masking), so their results differ slightly from the paper:

| Dataset | Released checkpoint | T2M R@1 | R@2 | R@3 | R@5 | R@10 | MedR | M2T R@1 | R@2 | R@3 | R@5 | R@10 | MedR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HumanML3D | `motioncolbert_l_humanml3d.pt` | 13.63 | 19.59 | 26.15 | 34.57 | 47.28 | 12.0 | 14.27 | 17.86 | 25.42 | 33.08 | 44.90 | 13.0 |
| KIT-ML    | `motioncolbert_l_kit.pt` | 15.30 | 25.54 | 33.86 | 45.66 | 59.04 | 7.0 | 16.51 | 17.95 | 28.43 | 36.39 | 52.89 | 9.0 |

## Citation

```bibtex
@inproceedings{zhang2026motioncolbert,
  title     = {Fine-grained Motion Retrieval via Joint-Angle Motion Images and Token-Patch Late Interaction},
  author    = {Zhang, Yao and Liu, Zhuchenyang and He, Yanlan and Ploetz, Thomas and Xiao, Yu},
  booktitle = {Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year      = {2026}
}
```

## Acknowledgements

This work was a part of Finland's Ministry of Education and Culture's Doctoral Education Pilot under
Decision No. VN/3137/2024-OKM-6 (The Finnish Doctoral Program Network in Artificial Intelligence, AI-DOC).
Our code builds on [MoPatch](https://github.com/line/MotionPatches) and [TMR](https://github.com/Mathux/TMR).

## License

This code is released under the [CC BY-NC 4.0](LICENSE) license. Parts of it are modified from
[MoPatch](https://github.com/line/MotionPatches) (Copyright 2024 LY Corporation, CC BY-NC 4.0); these files
carry a corresponding notice in their header.
