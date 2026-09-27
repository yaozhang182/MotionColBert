# Modified from MoPatch (https://github.com/line/MotionPatches),
# Copyright 2024 LY Corporation, licensed under CC BY-NC 4.0
# (https://creativecommons.org/licenses/by-nc/4.0/).
"""MotionColBert: joint-angle Motion Image + token-to-patch MaxSim + MLM regularization.

Only the configuration used in the main results table (Table 2 of the paper) is kept.
Parameter names are unchanged from training, so released checkpoints load with strict=True.
"""
import numpy as np
import timm
import torch
import torch.nn.functional as F
import transformers
from torch import nn

# Degrees of freedom of the 14 body parts, in the order produced by
# datasets/joint_angle_dataset.py (ALL_USED_INDICES); 29 dims in total.
PART_NAMES = [
    "global", "pelvis", "r_hip", "l_hip", "r_knee", "l_knee", "r_ankle", "l_ankle",
    "lumbar", "neck", "r_shoulder", "l_shoulder", "r_elbow", "l_elbow",
]
PART_DIMS = [3, 3, 3, 3, 1, 1, 1, 1, 3, 2, 3, 3, 1, 1]
PART_WIDTH = 16  # one ViT patch row per body part -> 14 x 16 = 224 pixels
MOTION_LENGTH = 224


class TextEncoder(nn.Module):
    """Transformer language model returning per-token hidden states (MaxSim) and an MLM loss."""

    def __init__(self, model_name: str, trainable: bool = True) -> None:
        super().__init__()
        self.text_model = transformers.AutoModelForMaskedLM.from_pretrained(model_name)
        # Backbone without the LM head, used for retrieval features.
        for attr in ("distilbert", "bert", "roberta"):
            if hasattr(self.text_model, attr):
                self.base_model = getattr(self.text_model, attr)
                break
        else:
            self.base_model = self.text_model.base_model
        for param in self.text_model.parameters():
            param.requires_grad = trainable

    def forward(self, input_ids, attention_mask):
        hidden = self.base_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        # Drop the leading [CLS] and the trailing position ([SEP] or padding); padding is
        # masked out again in ClipModel.compute_maxsim_scores.
        return hidden[:, 1:-1, :]

    def compute_mlm_loss(self, input_ids, attention_mask, labels):
        return self.text_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss


class MotionEncoder(nn.Module):
    """Builds the 224x224 joint-angle Motion Image and encodes it with a ViT."""

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        trainable: bool = True,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            img_size=(224, 224),
            in_chans=3,
            drop_path_rate=drop_path_rate,
        )
        # One learnable linear projection per body part: DoF -> 16-pixel band.
        for name, dim in zip(PART_NAMES, PART_DIMS):
            setattr(self, f"embed_{name}", nn.Linear(dim, PART_WIDTH))
        self.part_dims = PART_DIMS
        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        # x: (B, T, 29) normalized joint angles
        T = x.shape[1]
        parts = torch.split(x, self.part_dims, dim=2)
        bands = [getattr(self, f"embed_{name}")(p) for name, p in zip(PART_NAMES, parts)]
        image = torch.cat(bands, dim=2)  # (B, T, 224)
        if T != MOTION_LENGTH:
            image = F.interpolate(image.permute(0, 2, 1), size=MOTION_LENGTH,
                                  mode="linear", align_corners=False).permute(0, 2, 1)
        image = image.unsqueeze(1).repeat(1, 3, 1, 1)  # grey image -> 3 channels
        # Patch tokens of the last ViT block: (B, 196, D)
        return self.model.get_intermediate_layers(image, n=1, reshape=False)[0]


class ClipModel(nn.Module):
    def __init__(
        self,
        motion_encoder_alias: str = "vit_base_patch16_224",
        text_encoder_alias: str = "distilbert-base-uncased",
        motion_encoder_pretrained: bool = True,
        motion_encoder_trainable: bool = True,
        text_encoder_trainable: bool = True,
        motion_embedding_dims: int = 768,
        text_embedding_dims: int = 768,
        projection_dims: int = 256,
        logit: float = 0.07,
        drop_path_rate: float = 0.0,
        lambda_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.lambda_weight = lambda_weight  # weight of T2M vs. M2T in the contrastive loss
        self.motion_encoder = MotionEncoder(
            motion_encoder_alias,
            pretrained=motion_encoder_pretrained,
            trainable=motion_encoder_trainable,
            drop_path_rate=drop_path_rate,
        )
        self.text_encoder = TextEncoder(text_encoder_alias, trainable=text_encoder_trainable)
        self.motion_projection = nn.Linear(motion_embedding_dims, projection_dims)
        self.text_projection = nn.Linear(text_embedding_dims, projection_dims)
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1 / logit)))

    def encode_motion(self, motion):
        """(B, T, 29) -> L2-normalized patch embeddings (B, 196, projection_dims)."""
        return F.normalize(self.motion_projection(self.motion_encoder(motion)), p=2, dim=-1)

    def encode_text(self, text):
        """Tokenized text -> L2-normalized token embeddings (B, L-2, projection_dims)."""
        feats = self.text_encoder(input_ids=text["input_ids"], attention_mask=text["attention_mask"])
        return F.normalize(self.text_projection(feats), p=2, dim=-1)

    def compute_maxsim_scores(self, text_embeds, motion_embeds, attention_mask):
        """T2M MaxSim: for every text token take the best-matching motion patch, then
        average over the valid text tokens. Returns (B_text, B_motion) logits."""
        interaction = torch.einsum("itd,jmd->ijtm", text_embeds, motion_embeds)
        max_sim = interaction.max(dim=-1).values
        mask = attention_mask[:, 1:-1].unsqueeze(1).float()
        scores = (max_sim * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1e-9)
        return scores * self.logit_scale.exp()

    def retrieval_loss(self, scores):
        labels = torch.arange(len(scores), device=scores.device)
        t2m = F.cross_entropy(scores, labels)
        m2t = F.cross_entropy(scores.t(), labels)
        return self.lambda_weight * t2m + (1.0 - self.lambda_weight) * m2t

    def forward(self, motion, text, return_loss=False):
        self.logit_scale.data.clamp_(max=4.6052)
        scores = self.compute_maxsim_scores(self.encode_text(text), self.encode_motion(motion),
                                            text["attention_mask"])
        return self.retrieval_loss(scores) if return_loss else scores

    def forward_with_mlm(self, motion, clean_text, masked_input_ids, masked_attention_mask,
                         mlm_labels, lambda_mlm=0.2, mlm_batch_size=None):
        """Retrieval loss on clean text + lambda_mlm * MLM loss on masked text."""
        self.logit_scale.data.clamp_(max=4.6052)
        scores = self.compute_maxsim_scores(self.encode_text(clean_text), self.encode_motion(motion),
                                            clean_text["attention_mask"])
        retrieval_loss = self.retrieval_loss(scores)
        if lambda_mlm > 0:
            n = mlm_batch_size if mlm_batch_size else masked_input_ids.size(0)
            mlm_loss = self.text_encoder.compute_mlm_loss(
                masked_input_ids[:n], masked_attention_mask[:n], mlm_labels[:n])
        else:
            mlm_loss = torch.tensor(0.0, device=motion.device)
        return retrieval_loss + lambda_mlm * mlm_loss, retrieval_loss, mlm_loss
