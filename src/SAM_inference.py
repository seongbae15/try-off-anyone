from config import device
from transformers import SamModel, SamProcessor
from diffusers.image_processor import VaeImageProcessor
from src.model.pipeline import TryOffAnyone
from src.preprocessing import background_removal, background_whitening, mask_generation
from PIL import Image
from io import BytesIO
import argparse
import numpy as np
import torch
import os
import requests


from transformers import SamModel, SamProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")


def inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--url', type=str, default="https://cdn11.bigcommerce.com/s-405b0/images/stencil/590x590/products/97/20409/8000-gildan-tee-t-shirt.ca-model__66081.1724276210.jpg")
    parser.add_argument('--mask_type', type=int, default="2")

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
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    mask_processor = VaeImageProcessor(
        vae_scale_factor = 8, do_normalize=False, do_binarize=True, do_convert_grayscale=True
    )
    vae_processor = VaeImageProcessor(vae_scale_factor=8)

    image = Image.open(get_image_file(args.url))
    image = image.convert("RGB").resize((args.width, args.height))

    # 좌표 지정하는 코드 필요.
    points = [[[500, 600]]]

    # mask_image = mask_generation(image, processor, model) #, "Tops")

    inputs = processor(image, input_points=points, return_tensors="pt").to(device)
    image_embeddings = model.get_image_embeddings(inputs["pixel_values"])

    inputs.pop("pixel_values", None)

    inputs.update({"image_embeddings": image_embeddings,})

    with torch.no_grad():
        outputs = model(**inputs)
    
    # mask = mask_processor.preprocess(outputs.numpy(), args.height, args.width)[0]
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )
    cloth = vae_processor.preprocess(image, args.height, args.width)[0]

    # bool ndarray → float tensor → [1,1,H,W]
    mask_np     = masks[0]                           # (H, W), bool
    mask_tensor = torch.from_numpy(mask_np.astype(np.float32))
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).to(device)
    # → generate_laydown(pipeline, cloth_tensor, mask_tensor, args)

    laydown_image = generate_laydown(pipeline, cloth, masks, args)
    laydown_image = background_whitening(
        background_removal(laydown_image), args.width, args.height
    )
    laydown_image.save(os.path.join('data', f"{args.url.split('/')[-1][:-4]}.png"))
