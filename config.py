import torch

TEST_DATA_PATH = 'data/zalando-hd-resized/test/'
device = 'mps'
concat_d = -2
dtype = torch.bfloat16 if device == 'cuda' else torch.float32
base_ckpt = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
