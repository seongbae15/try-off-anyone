import argparse
from pathlib import Path
import os
import math

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from config import base_ckpt
from src.model.pipeline import fine_tuned_modules, skip_cross_attentions

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_path", type=str, default=base_ckpt)
    parser.add_argument(
        "--vae_model_path", type=str, default="stabilityai/sd-vae-ft-mse"
    )
    parser.add_argument("--dataset_path", type=str, default="./data/zalando-hd-resized")
    parser.add_argument("--output_dir", type=str, default="./ckpt_save/")
    parser.add_argument("--seed", type=int, default=36)
    parser.add_argument("--resolution_height", type=int, default=512)
    parser.add_argument("--resolution_width", type=int, default=384)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=20)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--checkpoining_steps", type=int, default=500)
    parser.add_argument("--checkpoints_toal_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    args = parser.parse_args()
    return args


class VTONDataset(Dataset):
    def __init__(self, dataset_path, split="train", height=512, width=384):
        self.dataset_path = dataset_path
        self.split = split
        self.height = height
        self.width = width
        self.image_person_dir = Path(self.dataset_path).joinpath(split, "image")
        self.image_cloth_dir = Path(self.dataset_path).joinpath(split, "cloth")
        self.mask_cloth_person_dir = Path(self.dataset_path).joinpath(
            split, "agnostic-mask"
        )

        # Check if directories exist
        if not self.image_person_dir.exists():
            raise FileNotFoundError(
                f"Image person directory not found: {self.image_person_dir}"
            )
        if not self.image_cloth_dir.exists():
            raise FileNotFoundError(
                f"Image cloth directory not found: {self.image_cloth_dir}"
            )
        if not self.mask_cloth_person_dir.exists():
            raise FileNotFoundError(
                f"Mask cloth person directory not found: {self.mask_cloth_person_dir}"
            )

        self.image_persion_files = sorted(
            [f for f in self.image_person_dir.glob("*.jpg")]
        )

        self.valid_files = []
        for person_img_path in self.image_persion_files:
            base_name = person_img_path.stem
            cloth_img_path = Path(self.image_cloth_dir).joinpath(f"{base_name}.jpg")
            mask_img_path = Path(self.mask_cloth_person_dir).joinpath(
                f"{base_name}_mask.png"
            )

            if cloth_img_path.exists() and mask_img_path.exists():
                self.valid_files.append(
                    {
                        "person": person_img_path,
                        "cloth": cloth_img_path,
                        "mask": mask_img_path,
                    }
                )
            else:
                logger.warning(f"Skip {base_name}")

        if not self.valid_files:
            raise ValueError(f"No valid files found in {self.split} split.")

        # Image preprocessor
        self.vae_image_processor = VaeImageProcessor(
            vae_scale_factor=8, do_normalize=True, do_binarize=True, do_convert_rgb=True
        )
        # Mask preprocessor
        self.mask_processor = VaeImageProcessor(
            vae_scale_factor=8,
            do_normalize=False,
            do_binarize=True,
            do_convert_grayscale=True,
        )

        self.transform_resize = transforms.Resize(
            (self.height, self.width),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        item = self.valid_files[idx]

        try:
            person_image_hc = Image.open(item["person"])
            cloth_image_c = Image.open(item["cloth"])
            garment_mask_m = Image.open(item["mask"])

            person_image_hc = self.transform_resize(person_image_hc)
            cloth_image_c = self.transform_resize(cloth_image_c)
            garment_mask_m = self.transform_resize(garment_mask_m)

            person_image_hc_tensor = self.vae_image_processor.preprocess(
                person_image_hc
            )
            cloth_image_c_tensor = self.vae_image_processor.preprocess(cloth_image_c)
            garment_mask_m_tensor = self.mask_processor.preprocess(garment_mask_m)

            if not isinstance(person_image_hc_tensor, torch.Tensor):
                person_image_hc_tensor = torch.tensor(person_image_hc_tensor)
            if not isinstance(cloth_image_c_tensor, torch.Tensor):
                cloth_image_c_tensor = torch.tensor(cloth_image_c_tensor)
            if not isinstance(garment_mask_m_tensor, torch.Tensor):
                garment_mask_m_tensor = torch.tensor(garment_mask_m_tensor)

            person_image_hc_tensor = person_image_hc_tensor.float()
            cloth_image_c_tensor = cloth_image_c_tensor.float()
            garment_mask_m_tensor = garment_mask_m_tensor.float()

            if person_image_hc_tensor.ndim == 4:
                person_image_hc_tensor = person_image_hc_tensor.squeeze(0)
            if cloth_image_c_tensor.ndim == 4:
                cloth_image_c_tensor = cloth_image_c_tensor.squeeze(0)
            if garment_mask_m_tensor.ndim == 4:
                garment_mask_m_tensor = garment_mask_m_tensor.squeeze(0)

            return {
                "person_hc": person_image_hc_tensor,
                "cloth_c": cloth_image_c_tensor,
                "mask_m": garment_mask_m_tensor,
            }
        except Exception as e:
            logger.error(
                f"Error processing item at index {idx}: ({item['person'].name}){e}"
            )

            if idx > 0:
                return self.__getitem__(idx - 1)
            else:
                dummy_hc = torch.zeros(
                    (3, self.height, self.width), dtype=torch.float32
                )
                dummy_c = torch.zeros((3, self.height, self.width), dtype=torch.float32)
                dummy_m = torch.zeros((1, self.height, self.width), dtype=torch.float32)
                print("It's dummy")
                return {"person_hc": dummy_hc, "cloth_c": dummy_c, "mask_m": dummy_m}


def train_g():
    args = parse_args()

    logging_dir = Path(args.logging_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="no",
        log_with="tensorboard",
        project_dir=logging_dir,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load Models
    noise_scheduler = DDIMScheduler.from_pretrained(
        args.pretrained_path, subfolder="scheduler"
    )
    vae = AutoencoderKL.from_pretrained(args.vae_model_path)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_path, subfolder="unet")

    # Freeze VAE
    vae.requires_grad_(False)

    # Freeze all U-Net parameters first
    unet.requires_grad_(False)

    # Unfreeze only transformer block
    trainable_unet_module = fine_tuned_modules(unet)
    params_to_optimize = unet.parameters()
    if not trainable_unet_module:
        logger.warning("No trainable modules identified by fine_tuned_modules.")
        unet.requires_grad_(True)
    else:
        for module_list in trainable_unet_module:
            for module in module_list:
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad = True
        params_to_optimize = []
        for module_list in trainable_unet_module:
            for module in module_list:
                if module is not None:
                    params_to_optimize.extend(list(module.parameters()))

        if not params_to_optimize:
            logger.error(
                "No parameters found to optimize even after attempting to unfreeze transformer blocks."
            )
            unet.requires_grad_(True)
            params_to_optimize = unet.parameters()

    # Skip cross-attentions as per
    unet.set_attn_processor(skip_cross_attentions(unet))

    # Optimizer
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Dataset and DataLoader
    train_dataset = VTONDataset(
        args.dataset_path,
        split="train",
        height=args.resolution_height,
        width=args.resolution_width,
    )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.train_batch_size,
        num_workers=4,
    )

    # Scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(
            args.max_train_steps
            if args.max_train_steps
            else len(train_dataloader) * args.num_train_epochs
        ),
    )

    # Prepare everything with our accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast vae to fp32
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(
            args.max_train_steps / num_update_steps_per_epoch
        )

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs in d.startswith("checkpoint-")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None

        else:
            accelerator.print(f"Resuming from chekcpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                latents_c = vae.encode(
                    batch["cloth_c"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents_c = latents_c * vae.config.scaling_factor

                with torch.no_grad():
                    latents_hc = vae.encode(
                        batch["person_hc"].to(dtype=weight_dtype)
                    ).latent_dist.sample()
                    latents_hc = latents_hc * vae.config.scaling_factor

                mask_m_resized = F.interpolate(
                    batch["mask_m"].to(dtype=weight_dtype),
                    size=latents_c.shape[-2:],
                    mode="nearest",
                )

                noise = torch.randn_like(latents_c)
                bsz = latents_c.shape[0]

                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents_c.device,
                ).long()

                # Add noise to the latents_c according to the noise magnitude at each timestep
                noisy_latents_c = noise_scheduler.add_noise(latents_c, noise, timesteps)

                # Prepare U-Net input
                model_input = torch.cat(
                    [noisy_latents_c, mask_m_resized, latents_hc], dim=1
                )

                # Predict the noise residual
                noise_pred = unet(
                    model_input, timesteps, encoder_hidden_states=None
                ).sample

                # Calcluate loss
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")

                # Gather the losses
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = [
                        p for p in params_to_optimize if p.grad is not None
                    ]
                    if params_to_clip:
                        accelerator.clip_grad_norm_(params_to_clip, 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpoining_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                        # Save U-Net
                        unwrapped_unet = accelerator.unwrap_model(unet)

                        trainable_state_dict = {}
                        for name, param in unwrapped_unet.named_parameters():
                            if param.requires_grad:
                                trainable_state_dict[name] = param.detach().cpu()

                        if trainable_state_dict:
                            torch.save(
                                trainable_state_dict,
                                os.path.join(save_path, "unet_transformer_block.pt"),
                            )
                            logger.info(
                                f"Saved fine-tuned U-Net transformer block to {os.path.join(save_path, 'unet_transformer_blocks.pt')}"
                            )
                        else:
                            logger.warning("No trainable parameters found.")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save the final trained_model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        # Save only the fine-tuned parameters
        final_trainable_state_dict = {}
        for name, param in unwrapped_unet.named_parameters():
            if param.requires_grad:
                final_trainable_state_dict[name] = param.cpu().clone()

        if final_trainable_state_dict:
            torch.save(
                final_trainable_state_dict,
                os.path.join(args.output_dir, "final_unet_transformer_block.pt"),
            )
            logger.info(
                f"Saved_final fine-tuned U-Net transformer block to {os.path.join(args.output_dir, 'final_unet_transformer_block.pt')}"
            )
        else:
            logger.warning("No trainable parameters found.")
            unwrapped_unet.save_pretrained(
                os.path.join(args.output_dir, "final_unet_full")
            )
    accelerator.end_training()


if __name__ == "__main__":
    train_g()
