from PIL import Image, ImageOps
from numpy import ndarray, zeros, asarray
from typing import Tuple


def load_image(image_path: str, size: Tuple[int, int]) -> ndarray:
    try:
        with Image.open(image_path, "r") as img:
            img = ImageOps.grayscale(img)
            img: Image = img.resize(size, Image.BILINEAR)
            return asarray(img) / 255.
    except Exception as e:
        print(e)
        return zeros(size)
