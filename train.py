import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from CNN import CNN_FM
from dataset import EmojiDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

dataset    = EmojiDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model      = CNN_FM().to(device)
opt        = torch.optim.Adam(model.parameters(), lr=1e-4)


def flow_matching_loss(model, x1, emb):
    """x1: (B, 1, H, W) normalized to [0, 1]; emb: (B, 512) CLIP embedding"""
    B  = x1.shape[0]
    x0 = torch.randn_like(x1)
    t  = torch.rand(B, device=x1.device)

    t_  = t.view(B, 1, 1, 1)
    xt  = (1 - t_) * x0 + t_ * x1
    target = x1 - x0
    pred   = model(xt, t, emb)

    return nn.functional.mse_loss(pred, target)


EPOCHS      = 500
n_batches   = len(dataloader)
n_samples   = len(dataset)

lowest_loss = np.inf

for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch_idx, (x1, emb) in enumerate(dataloader):
        x1, emb = x1.to(device), emb.to(device)
        loss = flow_matching_loss(model, x1, emb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        if loss.item() < lowest_loss:
            torch.save(model.state_dict(), "emoji_fm.pt")
    print(f"\rEpoch {epoch+1}/{EPOCHS}  loss={total_loss/n_batches:.4f}  "
          f"[all {n_samples} images seen]")

torch.save(model.state_dict(), "emoji_fm.pt")
