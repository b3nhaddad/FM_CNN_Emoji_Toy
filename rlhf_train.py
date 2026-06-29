"""
Two-phase RLHF:
  Phase 1 — train the reward model on feedback.json labels,
             conditioned on both image and text embedding.
  Phase 2 — fine-tune the flow model by maximising reward via
             differentiable ODE rollout.
"""
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from CNN import CNN_FM
from reward_model import RewardModel

FEEDBACK_FILE  = 'feedback.json'
FLOW_CKPT      = 'emoji_fm_2.pt'
DEVICE         = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RLHF_STEPS     = 20
REWARD_EPOCHS  = 20
RLHF_EPOCHS    = 50
KL_COEF        = 0.1


# ---------------------------------------------------------------------------
# Phase 1 — reward model training
# ---------------------------------------------------------------------------

class FeedbackDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            records = json.load(f)
        images, embeddings, ratings = [], [], []
        for r in records:
            arr = np.array(Image.open(r['image_path']).convert('RGB'), dtype=np.float32) / 255.0
            images.append(arr)
            embeddings.append(r['embedding'])
            ratings.append(r['rating'])

        self.images     = torch.tensor(np.stack(images)).permute(0, 3, 1, 2)  # (N, 3, 64, 64)
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)         # (N, 512)
        self.ratings    = torch.tensor(ratings, dtype=torch.float32)            # (N,)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.images[idx], self.embeddings[idx], self.ratings[idx]


def train_reward_model():
    fb_ds  = FeedbackDataset(FEEDBACK_FILE)
    fb_dl  = DataLoader(fb_ds, batch_size=16, shuffle=True)

    reward  = RewardModel().to(DEVICE)
    opt     = torch.optim.Adam(reward.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    print("=== Phase 1: training reward model ===")
    for epoch in range(REWARD_EPOCHS):
        total = 0.0
        pbar  = tqdm(fb_dl, desc=f"Reward epoch {epoch+1}/{REWARD_EPOCHS}", leave=True)
        for img, emb, rating in pbar:
            img, emb, rating = img.to(DEVICE), emb.to(DEVICE), rating.to(DEVICE)
            pred = reward(img, emb)
            loss = loss_fn(pred, rating)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            pbar.set_postfix(bce=f"{loss.item():.4f}")
        print(f"  Epoch {epoch+1}/{REWARD_EPOCHS}  avg_bce={total/len(fb_dl):.4f}")

    torch.save(reward.state_dict(), "reward_model.pt")
    print("  Saved → reward_model.pt\n")
    return reward


# ---------------------------------------------------------------------------
# Phase 2 — RLHF fine-tuning of the flow model
# ---------------------------------------------------------------------------

def rlhf_finetune(reward: RewardModel):
    flow = CNN_FM().to(DEVICE)
    flow.load_state_dict(torch.load(FLOW_CKPT, map_location=DEVICE))
    flow.train()
    reward.eval()

    flow_ref = CNN_FM().to(DEVICE)
    flow_ref.load_state_dict(torch.load(FLOW_CKPT, map_location=DEVICE))
    flow_ref.eval()
    for p in flow_ref.parameters():
        p.requires_grad_(False)

    with open(FEEDBACK_FILE) as f:
        records = json.load(f)
    all_embs = torch.tensor(
        [r['embedding'] for r in records], dtype=torch.float32
    ).to(DEVICE)

    opt = torch.optim.Adam(flow.parameters(), lr=1e-5)

    print("=== Phase 2: RLHF fine-tuning ===")
    pbar = tqdm(range(RLHF_EPOCHS), desc="RLHF")
    for epoch in pbar:
        idx  = torch.randint(0, len(all_embs), (16,))
        emb  = all_embs[idx]

        img  = _rollout(flow, emb)
        r    = reward(img, emb)

        # KL penalty: compare velocity fields at random (x, t) points
        x0_kl  = torch.randn(emb.shape[0], 3, 64, 64, device=DEVICE)
        t_kl   = torch.rand(emb.shape[0], device=DEVICE)
        t_kl_  = t_kl.view(-1, 1, 1, 1)
        x1_kl  = img.detach()
        xt_kl  = (1 - t_kl_) * x0_kl + t_kl_ * x1_kl
        with torch.no_grad():
            v_ref = flow_ref(xt_kl, t_kl, emb)
        v_new  = flow(xt_kl, t_kl, emb)
        kl_loss = torch.nn.functional.mse_loss(v_new, v_ref)

        loss = -r.mean() + KL_COEF * kl_loss

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
        opt.step()

        pbar.set_postfix(reward=f"{r.mean().item():.4f}", kl=f"{kl_loss.item():.4f}")

    torch.save(flow.state_dict(), "emoji_fm_rlhf.pt")
    print("  Saved → emoji_fm_rlhf.pt")


# ---------------------------------------------------------------------------
# Differentiable ODE rollout — no @torch.no_grad so grads flow through
# ---------------------------------------------------------------------------

def _rollout(model, emb):
    x  = torch.randn(emb.shape[0], 3, 64, 64, device=DEVICE)
    dt = 1.0 / RLHF_STEPS
    for i in range(RLHF_STEPS):
        t      = torch.full((emb.shape[0],), i * dt, device=DEVICE)
        t_next = torch.clamp(torch.full((emb.shape[0],), (i + 1) * dt, device=DEVICE), max=1.0)
        v1     = model(x, t, emb)
        x_pred = x + dt * v1
        v2     = model(x_pred, t_next, emb)
        x      = x + dt * (v1 + v2) / 2
    return x


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    reward = train_reward_model()
    rlhf_finetune(reward)
