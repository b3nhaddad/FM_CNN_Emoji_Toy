FROM python:3.12-slim

WORKDIR /app

# Install deps first (layer-cached until requirements change)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download CLIP so cold starts don't hit HuggingFace
RUN python -c "from transformers import CLIPTokenizer, CLIPTextModel; \
    CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')"

# Copy source code
COPY model/ ./model/
COPY data/ ./data/
COPY api/main.py ./api/main.py

# Copy model weights (must exist in build context — not gitignored here)
COPY emoji_fm_best.pt ./emoji_fm_best.pt

ENV MODEL_CKPT=emoji_fm_best.pt
ENV PORT=8080

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--timeout", "120", "api.main:app"]
