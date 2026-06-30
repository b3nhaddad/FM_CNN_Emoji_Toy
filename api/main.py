import io
import base64
import logging
import os
import sys

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image as PILImage

# Add repo root to path so model/ and data/ are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.cnn import CNN_FM
from data.sample import load_clip, sample, encode_text

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

DEVICE = torch.device("cpu")  # Cloud Run has no GPU
CKPT = os.environ.get("MODEL_CKPT", "emoji_fm_best.pt")
GUIDANCE_SCALE = float(os.environ.get("GUIDANCE_SCALE", "7.5"))

app = Flask(__name__)
CORS(app, origins=os.environ.get("CORS_ORIGINS", "*"))

# ── Load model once at startup ────────────────────────────────────────────────

log.info("Loading CLIP...")
tokenizer, clip_model = load_clip(DEVICE)

log.info("Loading FM model from %s...", CKPT)
model = CNN_FM().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
model.eval()
log.info("Models ready.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/generate", methods=["POST"])
def generate():
    body = request.get_json(silent=True) or {}
    prompt = (body.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "prompt is required"}), 400
    if len(prompt) > 300:
        return jsonify({"error": "prompt too long (max 300 chars)"}), 400

    log.info("Generating for prompt: %r", prompt)

    emb = encode_text(prompt, tokenizer, clip_model, DEVICE)
    img_tensor = sample(model, emb, DEVICE, guidance_scale=GUIDANCE_SCALE)

    img_np = (
        img_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255
    ).clip(0, 255).astype("uint8")
    pil_img = PILImage.fromarray(img_np, mode="RGB")

    # Scale up 64×64 → 256×256 with nearest-neighbor to keep the pixel-art look
    pil_img = pil_img.resize((256, 256), PILImage.NEAREST)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return jsonify({"url": f"data:image/png;base64,{b64}"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
