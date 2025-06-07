import argparse
import os
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation

from config import dtype, device, base_ckpt, TEST_DATA_PATH, concat_d, TRAIN_DATA_PATH
from src.model.attention import Skip


def skip_cross_attentions(unet):
    attn_processors = {
        name: unet.attn_processors[name] if name.endswith("attn1.processor") else Skip()
        for name in unet.attn_processors.keys()
    }
    return attn_processors


def fine_tuned_modules(unet):
    trainable_modules = torch.nn.ModuleList()
    for blocks in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
        if hasattr(blocks, "attentions"):
            trainable_modules.append(blocks.attentions)
        else:
            for block in blocks:
                if hasattr(block, "attentions"):
                    trainable_modules.append(block.attentions)
    return trainable_modules


class ClothDataset(Dataset):
    def __init__(
        self, image_root, mask_root, cloth_root, processor, mask_model, transform
    ):
        self.image_files = sorted(os.listdir(image_root))
        self.image_root = image_root
        self.mask_root = mask_root
        self.cloth_root = cloth_root
        self.processor = processor
        self.mask_model = mask_model
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image = Image.open(os.path.join(self.image_root, filename)).convert("RGB")
        cloth = Image.open(os.path.join(self.cloth_root, filename)).convert("RGB")
        mask_image = Image.open(
            os.path.join(self.mask_root, f"{Path(filename).stem}_mask.png")
        ).convert("L")

        image = image.resize((384, 512))
        cloth = cloth.resize((384, 512))
        mask_image = mask_image.resize((384, 512))

        return self.transform(image), self.transform(mask_image), self.transform(cloth)


def train():
    parser = argparse.ArgumentParser(
        description="Train a model with specified parameters."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
    args = parser.parse_known_args()[0]

    print(args)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Load model
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(
        device, dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(base_ckpt, subfolder="unet").to(
        device, dtype=dtype
    )
    unet.set_attn_processor(skip_cross_attentions(unet))

    trainable = fine_tuned_modules(unet)
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=args.lr)

    vae.eval()
    noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder="scheduler")

    processor = SegformerImageProcessor.from_pretrained("sayeed99/segformer_b3_clothes")
    mask_model = AutoModelForSemanticSegmentation.from_pretrained(
        "sayeed99/segformer_b3_clothes"
    ).to(device)

    transform = transforms.Compose(
        [
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
        ]
    )

    dataset = ClothDataset(
        os.path.join(TRAIN_DATA_PATH, "image"),
        os.path.join(TRAIN_DATA_PATH, "agnostic-mask"),
        os.path.join(TRAIN_DATA_PATH, "cloth"),
        processor,
        mask_model,
        transform,
    )
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    for epoch in range(args.epochs):
        unet.train()
        loop = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for i, (image, mask, cloth) in enumerate(loop):
            image = image.to(device, dtype=dtype)
            mask = mask.to(device, dtype)
            cloth = cloth.to(device, dtype)
            with torch.no_grad():
                latent_image = (
                    vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
                )
                latent_masked = (
                    vae.encode(image * (mask < 0.5)).latent_dist.sample()
                    * vae.config.scaling_factor
                )
                latent_target = (
                    vae.encode(cloth).latent_dist.sample() * vae.config.scaling_factor
                )

            x = torch.cat([latent_masked, latent_image], dim=concat_d)
            m = torch.cat([mask, torch.zeros_like(mask)], dim=concat_d)
            x_cm = torch.cat([latent_target, latent_image], dim=concat_d)

            t = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (image.size(0),),
                device=device,
            ).long()
            noise = torch.randn_like(x_cm)
            noisy_latent = noise_scheduler.add_noise(x_cm, noise, t)
            model_input = torch.cat([noisy_latent, m, x], dim=1)
            noise_pred = unet(model_input, t, return_dict=False)[0]
            loss = torch.nn.functional.mase_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())
            writer.add_scalar(
                "Loss/train_step", loss.item(), epoch * len(dataloader) + i
            )
            break

        writer.add_scalar("Loss/train_epoch", loss.item(), epoch)

        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(
                args.ckpt_dir, f"unet_transformers_epoch_{epoch + 1}.pt"
            )
            torch.save(
                {
                    name: module.state_dict()
                    for name, module in zip(
                        [f"block_{i}" for i in range(len(trainable))], trainable
                    )
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")
        break
    writer.close()
    print("Training Complete")


if __name__ == "__main__":
    train()
