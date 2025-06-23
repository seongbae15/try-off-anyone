import sys
from safetensors.torch import load_file

ckpt_path = "/Users/seongbae/workspace/ezpz-test/try-off-anyone/ckpt/model.safetensors"

# Load the checkpoint
state_dict = load_file(ckpt_path)

total_params = 0
for k, v in state_dict.items():
    num_params = v.numel()
    print(f"{k}: {v.shape} -> {num_params} params")
    total_params += num_params

print(f"\nTotal tensors: {len(state_dict)}")
print(f"Total parameters: {total_params}")
