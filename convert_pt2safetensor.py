import torch
from safetensors.torch import save_file, load_file
import argparse
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path")
    args = parser.parse_args()

    state_dict = torch.load(args.ckpt_path)
    save_file(state_dict, Path(args.ckpt_path).parent.joinpath("model.safetensors"))
