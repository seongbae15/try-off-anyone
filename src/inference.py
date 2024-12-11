from config import device
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from diffusers.image_processor import VaeImageProcessor
from src.model.pipeline import TryOffAnyone
from src.preprocessing import background_removal, background_whitening, mask_generation
from PIL import Image
from io import BytesIO
import argparse
import torch
import os
import requests


def inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--url', type=str, default="https://cdn11.bigcommerce.com/s-405b0/images/stencil/590x590/products/97/20409/8000-gildan-tee-t-shirt.ca-model__66081.1724276210.jpg")
    return parser.parse_known_args()[0]


def get_image_file(image_url):
    try:
        image_response = requests.get(image_url, timeout=20)
    except Exception:
        raise "Provide valid url!"
    image_response.raise_for_status()
    return BytesIO(image_response.content)


def generate_laydown(pipeline, cloth_image, mask, args):
    result = pipeline(
        cloth_image, mask, inference_steps=args.steps, scale=args.scale, height=args.height,
        width=args.width, generator=torch.Generator(device=device).manual_seed(args.seed)
    )
    return result[0]


def test_image():
    args = inference_args()
    pipeline = TryOffAnyone()
    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("sayeed99/segformer_b3_clothes")
    model.to(device)
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )
    vae_processor = VaeImageProcessor(vae_scale_factor=8)

    image = Image.open(get_image_file(args.url))
    image = image.convert("RGB").resize((args.width, args.height))

    mask_image = mask_generation(image, processor, model, "Tops")

    mask = mask_processor.preprocess(mask_image, args.height, args.width)[0]
    image = vae_processor.preprocess(image, args.height, args.width)[0]

    laydown_image = generate_laydown(pipeline, image, mask, args)
    laydown_image = background_whitening(
        background_removal(laydown_image), args.width, args.height
    )
    laydown_image.save(os.path.join('data', f"{args.url.split('/')[-1][:-4]}.png"))
