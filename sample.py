import torch
from transformers import CLIPTokenizer, CLIPTextModel

CLIP_MODEL = 'openai/clip-vit-base-patch32'

def load_clip(device):
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
    clip      = CLIPTextModel.from_pretrained(CLIP_MODEL).eval().to(device)
    return tokenizer, clip

def encode_text(text, tokenizer, clip, device):
    inputs = tokenizer([text], padding=True, truncation=True, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        return clip(**inputs).pooler_output   # (1, 512)

@torch.no_grad()
def sample(model, emb, device, H=64, W=64, steps=20, guidance_scale=7.5):
    x        = torch.randn(1, 3, H, W, device=device)
    null_emb = torch.zeros_like(emb)
    dt       = 1.0 / steps

    for i in range(steps):
        t      = torch.full((1,), i * dt, device=device)
        t_next = torch.clamp(torch.full((1,), (i + 1) * dt, device=device), max=1.0)

        # CFG velocity at current step
        v_cond   = model(x, t, emb)
        v_uncond = model(x, t, null_emb)
        v        = v_uncond + guidance_scale * (v_cond - v_uncond)

        # Heun predictor step
        x_pred = x + dt * v

        # CFG velocity at predicted next step
        v_cond_next   = model(x_pred, t_next, emb)
        v_uncond_next = model(x_pred, t_next, null_emb)
        v_next        = v_uncond_next + guidance_scale * (v_cond_next - v_uncond_next)

        # Heun corrector
        x = x + dt * (v + v_next) / 2

    return x


if __name__ == '__main__':
    from CNN import CNN_FM

    device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model     = CNN_FM().to(device)
    model.load_state_dict(torch.load("emoji_fm.pt", map_location=device))
    model.eval()

    tokenizer, clip = load_clip(device)

    prompt = ""
    emb    = encode_text(prompt, tokenizer, clip, device)
    img    = sample(model, emb, device)

    # tensor (1,3,64,64) -> uint8 RGB PIL image
    import os
    from PIL import Image as PILImage
    img_np  = (img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
    pil_img = PILImage.fromarray(img_np, mode='RGB')
    out     = os.path.join('/Users/oliverhaddad/PycharmProjects/BensNN', 'output.png')
    pil_img.save(out)
    print(f"Saved → {out}")
