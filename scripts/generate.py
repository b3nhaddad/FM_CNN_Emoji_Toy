import os
import torch
from PIL import Image as PILImage

from model.cnn import CNN_FM
from data.sample import load_clip, sample, encode_text

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('Using device:', DEVICE)

IMG_DIR        = './generated_images/'
CKPT           = "/Users/couchroomkid/PycharmProjects/FM_CNN_Emoji_Toy/emoji_fm_best.pt"   # RLHF-finetuned model; fall back to "emoji_fm_2.pt" if needed
GUIDANCE_SCALE = 7.5




def main():
    tokenizer, clip = load_clip(DEVICE)
    os.makedirs(IMG_DIR, exist_ok=True)

    model = CNN_FM().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.eval()

    while True:
        prompt = input("Enter prompt: ").strip()

        emb = encode_text(prompt, tokenizer, clip, DEVICE)
        img = sample(model, emb, DEVICE, guidance_scale=GUIDANCE_SCALE)

        img_np  = (img.squeeze().permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype('uint8')
        pil_img = PILImage.fromarray(img_np, mode='RGB')

        filename = prompt.replace(' ', '_') + '.png' if prompt else 'output.png'
        out_path = os.path.join(IMG_DIR, filename)
        pil_img.save(out_path)
        print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
