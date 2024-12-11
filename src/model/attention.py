import torch


class Skip(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
            self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None
    ):
        return hidden_states
