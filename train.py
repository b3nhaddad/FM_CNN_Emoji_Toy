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


EPOCHS      = 500
scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-6)
n_batches   = len(dataloader)
n_samples   = len(dataset)

best_loss = np.inf

for epoch in range(EPOCHS):
    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
    for x1, emb in pbar:
        x1, emb = x1.to(device, dtype=torch.float32), emb.to(device, dtype=torch.float32)
        # CFG: randomly drop conditioning
        mask = torch.rand(emb.shape[0], device=emb.device) < 0.15
        emb = emb.clone()
        emb[mask] = 0.0
        loss = flow_matching_loss(model, x1, emb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    scheduler.step()
    avg_loss = total_loss / n_batches
    current_lr = scheduler.get_last_lr()[0]
    tag = ""
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "emoji_fm_best.pt")
        tag = "  ✓ saved"
    print(f"Epoch {epoch+1}/{EPOCHS}  avg_loss={avg_loss:.4f}  lr={current_lr:.2e}{tag}")

torch.save(model.state_dict(), "emoji_fm_last.pt")
