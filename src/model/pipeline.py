from accelerate import load_checkpoint_in_model
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from config import dtype, device, concat_d, base_ckpt
from src.model.attention import Skip
from src.preprocessing import convert_to_pil, prepare_image, prepare_mask_image
import torch


def fine_tuned_modules(unet):
    trainable_modules = torch.nn.ModuleList()
    for blocks in [unet.down_blocks, unet.mid_block, unet.up_blocks]:
        if hasattr(blocks, 'attentions'):
            trainable_modules.append(blocks.attentions)
        else:
            for block in blocks:
                if hasattr(block, 'attentions'):
                    trainable_modules.append(block.attentions)
    return trainable_modules


def skip_cross_attentions(unet):
    attn_processors = {
        name: unet.attn_processors[name] if name.endswith('attn1.processor') else Skip()
        for name in unet.attn_processors.keys()
    }
    return attn_processors


def encode(image, vae):
    image = image.to(memory_format=torch.contiguous_format).float().to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        return vae.encode(image).latent_dist.sample() * vae.config.scaling_factor


class TryOffAnyone:
    def __init__(self):
        self.noise_scheduler = DDIMScheduler.from_pretrained(base_ckpt, subfolder='scheduler')
        vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse').to(device, dtype=dtype)
        unet = UNet2DConditionModel.from_pretrained(
            base_ckpt, subfolder='unet'
        ).to(device, dtype=dtype)

        unet.set_attn_processor(skip_cross_attentions(unet))
        load_checkpoint_in_model(fine_tuned_modules(unet), 'ckpt')

        self.unet = torch.compile(unet, backend='aot_eager' if device == 'mps' else 'inductor')
        self.vae = torch.compile(vae, mode='reduce-overhead')

        torch.set_float32_matmul_precision('high')
        torch.backends.cuda.matmul.allow_tf32 = True

    @torch.no_grad()
    def __call__(self, image, mask, inference_steps, scale, height, width, generator):

        image = prepare_image(image).to(device, dtype=dtype)
        mask = prepare_mask_image(mask).to(device, dtype=dtype)
        masked_image = image * (mask < 0.5)

        masked_latent = encode(masked_image, self.vae)
        image_latent = encode(image, self.vae)
        mask = torch.nn.functional.interpolate(mask, size=masked_latent.shape[-2:], mode='nearest')

        masked_latent_concat = torch.cat([masked_latent, image_latent], dim=concat_d)
        mask_concat = torch.cat([mask, torch.zeros_like(mask)], dim=concat_d)

        latents = randn_tensor(
            masked_latent_concat.shape,
            generator=generator,
            device=masked_latent_concat.device,
            dtype=dtype,
        )

        self.noise_scheduler.set_timesteps(inference_steps, device=device)
        timesteps = self.noise_scheduler.timesteps

        if do_classifier_free_guidance := (scale > 1.0):
            masked_latent_concat = torch.cat(
                [
                    torch.cat([masked_latent, torch.zeros_like(image_latent)], dim=concat_d),
                    masked_latent_concat,
                ]
            )

            mask_concat = torch.cat([mask_concat] * 2)

        extra_step = {'generator': generator, 'eta': 1.0}
        for i, t in enumerate(timesteps):
            input_latents = (torch.cat([latents] * 2) if do_classifier_free_guidance else latents)
            input_latents = self.noise_scheduler.scale_model_input(input_latents, t)

            input_latents = torch.cat([input_latents, mask_concat, masked_latent_concat], dim=1)

            noise_pred = self.unet(
                input_latents,
                t.to(device),
                encoder_hidden_states=None,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_unc, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_unc + scale * (noise_pred_text - noise_pred_unc)

            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step).prev_sample

        latents = latents.split(latents.shape[concat_d] // 2, dim=concat_d)[0]
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(device, dtype=dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = convert_to_pil(image)
        return image
