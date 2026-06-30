import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPTokenizer, CLIPTextModel

EMOJI_PATH = '/Users/oliverhaddad/node_modules/emoji-datasource-apple/img/apple/64'
EMOJI_JSON = '/Users/oliverhaddad/node_modules/emoji-datasource-apple/emoji.json'
CLIP_MODEL   = 'openai/clip-vit-base-patch32'
EMBED_CACHE  = 'emoji_embeddings_v2.pt'

SKIN_TONE_LABELS = {
    '1F3FB': 'light skin tone',
    '1F3FC': 'medium-light skin tone',
    '1F3FD': 'medium skin tone',
    '1F3FE': 'medium-dark skin tone',
    '1F3FF': 'dark skin tone',
}

class EmojiDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.moji_data = []
        self.build_image_pairs()

        images = [np.array(self._to_rgb(img)) for img, _ in self.moji_data]
        names  = [name for _, name in self.moji_data]

        self.features   = torch.tensor(np.stack(images), dtype=torch.bfloat16).permute(0, 3, 1, 2) / 255.0
        self.embeddings = self._load_or_encode(names)

    def _to_rgb(self, img):
        bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
        bg.paste(img.convert('RGBA'), mask=img.convert('RGBA').split()[3])
        return bg.convert('RGB').resize((64, 64))

    def _load_or_encode(self, names):
        if os.path.exists(EMBED_CACHE):
            print(f"Loading embeddings from {EMBED_CACHE}")
            return torch.load(EMBED_CACHE)

        print("Computing CLIP embeddings...")
        tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)
        clip      = CLIPTextModel.from_pretrained(CLIP_MODEL).eval()

        chunks = []
        for i in range(0, len(names), 64):
            batch  = names[i:i + 64]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
            with torch.no_grad():
                chunks.append(clip(**inputs).pooler_output)
        embeddings = torch.cat(chunks, dim=0)
        torch.save(embeddings, EMBED_CACHE)
        print(f"Saved embeddings to {EMBED_CACHE}")
        return embeddings

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.embeddings[idx]

    def build_image_pairs(self):
        with open(EMOJI_JSON) as f:
            data = json.load(f)
        for emoji in data:
            base_name = emoji['short_name'].replace('_', ' ')
            if emoji.get('has_img_apple'):
                image = Image.open(os.path.join(EMOJI_PATH, emoji['image']))
                self.moji_data.append((image, base_name))
            for tone_key, variant in emoji.get('skin_variations', {}).items():
                if variant.get('has_img_apple'):
                    image = Image.open(os.path.join(EMOJI_PATH, variant['image']))
                    label = SKIN_TONE_LABELS.get(tone_key, tone_key)
                    name  = f"{base_name} {label}"
                    self.moji_data.append((image, name))

#moji_dataset = EmojiDataset()
#print(len(moji_dataset))
