import torch
import torch.nn as nn


class RewardModel(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()
        self.image_enc = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256), nn.ReLU(),
        )
        self.text_enc = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, x, emb):
        img_feat  = self.image_enc(x)
        text_feat = self.text_enc(emb)
        return self.head(torch.cat([img_feat, text_feat], dim=1)).squeeze(1)
