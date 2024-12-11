from config import TEST_DATA_PATH, device
from diffusers.image_processor import VaeImageProcessor
from src.model.pipeline import TryOffAnyone
from src.preprocessing import background_removal, background_whitening
from tqdm import tqdm
from PIL import Image
from src.stats.calculate_statistics import statistics
import argparse
import torch
import os
import pathlib
import torch.multiprocessing as multiprocessing


def inference_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument('--gpu_id', type=int, default=0)
    return parser.parse_known_args()[0]


def generate_laydown(pipeline, cloth_image, mask, args):
    result = pipeline(
        cloth_image, mask, inference_steps=args.steps, scale=args.scale, height=args.height,
        width=args.width, generator=torch.Generator(device=device).manual_seed(args.seed)
    )
    return result[0]


def test_vton():
    args = inference_args()
    pipeline = TryOffAnyone()
    mask_processor = VaeImageProcessor(
        vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )
    vae_processor = VaeImageProcessor(vae_scale_factor=8)
    data = os.listdir(os.path.join(TEST_DATA_PATH, 'image'))
    save_model_dir = os.path.join('data', 'results')
    pathlib.Path(save_model_dir).mkdir(parents=True, exist_ok=True)
    for image_name in tqdm(data):
        model_image = Image.open(
            os.path.join(TEST_DATA_PATH, 'image', image_name)
        ).convert("RGB").resize((args.width, args.height))

        mask_image = Image.open(
            os.path.join(TEST_DATA_PATH, 'masks', image_name)
        ).resize((args.width, args.height))

        mask = mask_processor.preprocess(mask_image, args.height, args.width)[0]
        model_image = vae_processor.preprocess(model_image, args.height, args.width)[0]

        laydown_image = generate_laydown(pipeline, model_image, mask, args)
        laydown_image = background_whitening(
            background_removal(laydown_image), args.width, args.height
        )
        laydown_image.save(os.path.join('data', 'results', image_name[:-4] + ".png"))
    statistics()
