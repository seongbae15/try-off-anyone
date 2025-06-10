import torch

TEST_DATA_PATH = "data/zalando-hd-resized/test/"
device = "cuda"
concat_d = -2
dtype = torch.bfloat16 if device == "cuda" else torch.float16
base_ckpt = "stable-diffusion-v1-5/stable-diffusion-inpainting"
test_url = "https://img.shopcider.com/hermes/video/1747296659000-x2rzst.jpg"
