import PIL
import numpy as np
from PIL import Image

def white_bg(img: PIL.Image.Image) -> PIL.Image.Image:

    if img.mode != "RGBA":
        img = img.convert("RGBA")

    data = np.array(img)

    r, g, b, a = data[:, :, 0], data[:, :, 1], data[:, :, 2], data[:, :, 3]

    white_mask = (r == 255) & (g == 255) & (b == 255)

    data[white_mask, 3] = 0

    result = Image.fromarray(data)
    return result