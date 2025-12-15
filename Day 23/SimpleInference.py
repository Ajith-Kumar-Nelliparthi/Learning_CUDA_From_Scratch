import torch
import torch.nn as nn

def solve(input: torch.Tensor, model: nn.Module, output: torch.Tensor):
    output.copy_(model(input))