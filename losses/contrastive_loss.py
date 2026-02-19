from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (NT-Xent style) with optional feature queue.

    Args:
        temperature: contrastive temperature.
        base_temperature: scaling factor used in original SupCon formulation.
        queue_size: if > 0, stores past embeddings/labels for additional negatives/positives.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        base_temperature: float = 0.1,
        queue_size: int = 0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.queue_size = queue_size
        self._queue_features: Optional[torch.Tensor] = None
        self._queue_labels: Optional[torch.Tensor] = None

    @torch.no_grad()
    def _update_queue(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        if self.queue_size <= 0:
            return

        features = features.detach()
        labels = labels.detach()

        if self._queue_features is None:
            self._queue_features = features
            self._queue_labels = labels
        else:
            self._queue_features = torch.cat([self._queue_features, features], dim=0)
            self._queue_labels = torch.cat([self._queue_labels, labels], dim=0)

        if self._queue_features.size(0) > self.queue_size:
            self._queue_features = self._queue_features[-self.queue_size :]
            self._queue_labels = self._queue_labels[-self.queue_size :]

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2:
            raise ValueError("`features` must be at least 2D: (B, D) or (B, V, D)")
        if labels.ndim != 1:
            raise ValueError("`labels` must be 1D tensor of class indices")

        if features.ndim == 3:
            b, v, d = features.shape
            features = features.reshape(b * v, d)
            labels = labels.repeat_interleave(v)
        elif features.ndim > 3:
            features = features.view(features.size(0), -1)

        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1)
        batch_size = features.size(0)

        if batch_size != labels.size(0):
            raise ValueError("Batch size mismatch between features and labels")

        if self.queue_size > 0 and self._queue_features is not None and self._queue_labels is not None:
            queue_features = F.normalize(self._queue_features.to(features.device), dim=1)
            queue_labels = self._queue_labels.to(labels.device)
            contrast_features = torch.cat([features, queue_features], dim=0)
            contrast_labels = torch.cat([labels, queue_labels], dim=0)
        else:
            contrast_features = features
            contrast_labels = labels

        logits = torch.matmul(features, contrast_features.T) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()

        positive_mask = labels.view(-1, 1).eq(contrast_labels.view(1, -1)).float()

        logits_mask = torch.ones_like(positive_mask)
        if contrast_features.size(0) >= batch_size:
            diag_idx = torch.arange(batch_size, device=features.device)
            logits_mask[diag_idx, diag_idx] = 0.0
            positive_mask[diag_idx, diag_idx] = 0.0

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = positive_mask.sum(dim=1)
        valid = pos_count > 0

        mean_log_prob_pos = torch.zeros_like(pos_count)
        mean_log_prob_pos[valid] = (positive_mask[valid] * log_prob[valid]).sum(dim=1) / pos_count[valid]

        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos[valid]
        loss = loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=features.device)

        self._update_queue(features, labels)
        return loss
