from config import device
from PIL import Image
from transformers import pipeline
import numpy as np
import torch


def background_whitening(image, w, h):
    background = Image.new('RGBA', (w, h), (255, 255, 255))
    background.paste(image, (0, 0), mask=image)
    return background


def background_removal(image):
    pipe = pipeline(
        "image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device
    )
    return pipe(image)


def mask_generation(image, processor, model, category):
    inputs = processor(images=image, return_tensors="pt").to(device if device == 'cuda' else 'cpu')
    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    predicted_mask = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
    if category == "Tops":
        predicted_mask_1 = predicted_mask == 4
        predicted_mask_2 = predicted_mask == 7
    elif category == "Bottoms":
        predicted_mask_1 = predicted_mask == 5
        predicted_mask_2 = predicted_mask == 6
    else:
        raise NotImplementedError

    predicted_mask = predicted_mask_1 + predicted_mask_2
    mask_image = Image.fromarray((predicted_mask * 255).astype(np.uint8))
    return mask_image


def prepare_image(image):
    return image.unsqueeze(0).to(dtype=torch.float32)


def prepare_mask_image(mask_image):
    mask_image = mask_image.unsqueeze(0)
    mask_image[mask_image < 0.5] = 0
    mask_image[mask_image >= 0.5] = 1
    return mask_image


def convert_to_pil(images):
    images = (images * 255).round().astype("uint8")
    return [Image.fromarray(image) for image in images]
