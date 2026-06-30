"""
Interactive feedback collector.
Run after training: python scripts/collect_feedback.py
Press y = good, n = bad, q = quit.
Feedback is appended to feedback.json.
Each rated image is saved to feedback_images/ so the reward model
trains on the exact images you saw.
"""
import json
import os
import torch
from PIL import Image

from model.cnn import CNN_FM
from data.sample import load_clip, encode_text, sample

FEEDBACK_FILE = 'feedback.json'
IMG_DIR       = 'feedback_images'
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

PROMPTS = [
    "smiling face",
    "fire",
    "heart",
    "thumbs up",
    "crying face",
    "star",
    "rocket",
    "cat face",
    "pizza",
    "rainbow",
    "lightning bolt",
    "sun",
    "moon",
    "flower",
    "skull",
    "trophy",
    "musical note",
    "ghost",
    "dragon",
    "snowflake",
]


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE) as f:
            return json.load(f)
    return []


def save_feedback(records):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(records, f, indent=2)


def tensor_to_pil(img_tensor):
    arr = (img_tensor.squeeze().cpu().numpy() * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(arr, mode='L')


def main():
    os.makedirs(IMG_DIR, exist_ok=True)

    model = CNN_FM().to(DEVICE)
    model.load_state_dict(torch.load("emoji_fm_2.pt", map_location=DEVICE))
    model.eval()

    tokenizer, clip_model = load_clip(DEVICE)
    records = load_feedback()

    print("Rate each image: y = good, n = bad, q = quit\n")

    for prompt in PROMPTS:
        emb     = encode_text(prompt, tokenizer, clip_model, DEVICE)
        img     = sample(model, emb, DEVICE)    # (1, 1, 64, 64)
        pil_img = tensor_to_pil(img)
        pil_img.show()

        rating = input(f'  "{prompt}" → good? (y/n/q): ').strip().lower()
        if rating == 'q':
            break
        if rating not in ('y', 'n'):
            print("  skipped\n")
            continue

        # save the exact image the user rated
        img_path = os.path.join(IMG_DIR, f'{len(records):04d}_{prompt.replace(" ", "_")}.png')
        pil_img.save(img_path)

        records.append({
            'prompt':    prompt,
            'embedding': emb.squeeze().cpu().tolist(),
            'image_path': img_path,
            'rating':    1 if rating == 'y' else 0,
        })
        save_feedback(records)
        print(f"  saved ({len(records)} total)\n")

    print(f"Done. {len(records)} feedback records in {FEEDBACK_FILE}.")


if __name__ == '__main__':
    main()
