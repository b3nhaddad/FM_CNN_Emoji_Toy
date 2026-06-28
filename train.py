import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from CNN import CNN_FM
from dataset import EmojiDataset

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

dataset    = EmojiDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model      = CNN_FM().to(device)
opt        = torch.optim.Adam(model.parameters(), lr=1e-4)


def flow_matching_loss(model, x1, emb):
    """x1: (B, 1, H, W) normalized to [0, 1]; emb: (B, 512) CLIP embedding"""
    B  = x1.shape[0]
    x0 = torch.randn_like(x1)
    # stratified sampling: divide [0,1] into B equal strata, one sample each
    t  = (torch.arange(B, device=x1.device).float() + torch.rand(B, device=x1.device)) / B
    t  = t[torch.randperm(B, device=x1.device)]

    t_  = t.view(B, 1, 1, 1)
    xt  = (1 - t_) * x0 + t_ * x1
    target = x1 - x0
    pred   = model(xt, t, emb)

    return nn.functional.mse_loss(pred, target)


EPOCHS      = 25
n_batches   = len(dataloader)
n_samples   = len(dataset)

lowest_loss = np.inf

for epoch in range(EPOCHS):
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    for x1, emb in pbar:
        x1, emb = x1.to(device), emb.to(device)
        loss = flow_matching_loss(model, x1, emb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item()
        if loss.item() < lowest_loss:
            lowest_loss = loss.item()
            torch.save(model.state_dict(), "emoji_fm.pt")
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    print(f"Epoch {epoch+1}/{EPOCHS}  avg_loss={total_loss/n_batches:.4f}")

torch.save(model.state_dict(), "emoji_fm_2.pt")
