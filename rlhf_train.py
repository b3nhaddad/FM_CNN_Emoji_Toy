"""
Two-phase RLHF:
  Phase 1 — train the reward model on your feedback.json labels,
             using the exact images you rated (saved in feedback_images/)
  Phase 2 — fine-tune the flow model by maximising the reward via
             differentiable ODE rollout (short, 20-step version)
"""
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from CNN import CNN_FM
from reward_model import RewardModel

FEEDBACK_FILE = 'feedback.json'
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RLHF_STEPS    = 20
REWARD_EPOCHS = 20
RLHF_EPOCHS   = 50


# ---------------------------------------------------------------------------
# Phase 1 — reward model training
# ---------------------------------------------------------------------------

class FeedbackDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            records = json.load(f)
        # load the exact images the user rated
        images  = []
        for r in records:
            arr = np.array(Image.open(r['image_path']).convert('L'), dtype=np.float32) / 255.0
            images.append(arr)

        self.images  = torch.tensor(np.stack(images)).unsqueeze(1)   # (N, 1, 64, 64)
        self.ratings = torch.tensor([r['rating'] for r in records], dtype=torch.float32)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.images[idx], self.ratings[idx]


def train_reward_model():
    fb_ds  = FeedbackDataset(FEEDBACK_FILE)
    fb_dl  = DataLoader(fb_ds, batch_size=16, shuffle=True)

    reward  = RewardModel().to(DEVICE)
    opt     = torch.optim.Adam(reward.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    print("=== Phase 1: training reward model ===")
    for epoch in range(REWARD_EPOCHS):
        total = 0.0
        for img, rating in fb_dl:
            img, rating = img.to(DEVICE), rating.to(DEVICE)
            pred = reward(img)
            loss = loss_fn(pred, rating)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  Epoch {epoch+1}/{REWARD_EPOCHS}  bce={total/len(fb_dl):.4f}")

    torch.save(reward.state_dict(), "reward_model.pt")
    print("  Saved → reward_model.pt\n")
    return reward


# ---------------------------------------------------------------------------
# Phase 2 — RLHF fine-tuning of the flow model
# ---------------------------------------------------------------------------

def rlhf_finetune(reward: RewardModel):
    flow = CNN_FM().to(DEVICE)
    flow.load_state_dict(torch.load("emoji_fm.pt", map_location=DEVICE))
    flow.train()
    reward.eval()

    with open(FEEDBACK_FILE) as f:
        records = json.load(f)
    all_embs = torch.tensor(
        [r['embedding'] for r in records], dtype=torch.float32
    ).to(DEVICE)

    opt = torch.optim.Adam(flow.parameters(), lr=1e-5)

    print("=== Phase 2: RLHF fine-tuning ===")
    for epoch in range(RLHF_EPOCHS):
        idx  = torch.randint(0, len(all_embs), (16,))
        emb  = all_embs[idx]

        img  = _rollout(flow, emb)      # differentiable — grads flow back
        r    = reward(img)
        loss = -r.mean()               # maximise reward

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"  Epoch {epoch+1}/{RLHF_EPOCHS}  mean_reward={r.mean().item():.4f}")

    torch.save(flow.state_dict(), "emoji_fm_rlhf.pt")
    print("  Saved → emoji_fm_rlhf.pt")


# ---------------------------------------------------------------------------
# Differentiable ODE rollout — no @torch.no_grad so grads flow through
# ---------------------------------------------------------------------------

def _rollout(model, emb):
    x  = torch.randn(emb.shape[0], 1, 64, 64, device=DEVICE)
    dt = 1.0 / RLHF_STEPS
    for i in range(RLHF_STEPS):
        t = torch.full((emb.shape[0],), i * dt, device=DEVICE)
        x = x + model(x, t, emb) * dt
    return x


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    reward = train_reward_model()
    rlhf_finetune(reward)
