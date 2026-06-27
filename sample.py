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
def sample(model, emb, device, H=64, W=64, steps=100):
    x  = torch.randn(1, 1, H, W, device=device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((1,), i * dt, device=device)
        x = x + model(x, t, emb) * dt
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

    # tensor (1,1,64,64) -> uint8 grayscale PIL image
    import os
    from PIL import Image as PILImage
    img_np  = (img.squeeze().cpu().numpy() * 255).clip(0, 255).astype('uint8')
    pil_img = PILImage.fromarray(img_np, mode='L')
    out     = os.path.join('/Users/oliverhaddad/PycharmProjects/BensNN', 'output.png')
    pil_img.save(out)
    print(f"Saved → {out}")
