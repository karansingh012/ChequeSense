"""Neural network architectures for ChequeSense character and digit recognition."""

from __future__ import annotations

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.recognition.dataset import NUM_CLASSES


class ChequeDigitCNN(nn.Module):
    """Deep Convolutional Network for robust handwritten/printed cheque digit classification."""

    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES, dropout_rate: float = 0.25):
        super().__init__()
        self.num_classes = num_classes

        # Block 1: 32x32 -> 16x16
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.5),
        )

        # Block 2: 16x16 -> 8x8
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate * 0.75),
        )

        # Block 3: 8x8 -> 4x4
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(dropout_rate),
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout_rate * 1.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass computing unnormalized class logits."""
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = torch.flatten(out, 1)
        logits = self.classifier(out)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Computes softmax probabilities."""
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_glyph(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (predicted_class_indices, confidence_scores)."""
        probs = self.predict_proba(x)
        confs, preds = torch.max(probs, dim=-1)
        return preds, confs
