import argparse
import time
from typing import Tuple

import pandas as pd
import torch
import torchvision
from torch import nn


def get_inputs(size: Tuple[int, ...], device: str) -> torch.Tensor:
    return torch.randn(size).to(device)


def warmup(m: nn.Module, inputs: torch.Tensor, N: int = 10) -> None:
    m.eval()
    for _ in range(N):
        m(inputs)


@torch.inference_mode()
def measure_forward_inference(m: nn.Module, inputs: torch.Tensor, N: int = 100) -> float:
    m.eval()
    forward_time = 0
    for _ in range(N):
        time0 = time.time()
        m(inputs)
        forward_time += time.time() - time0

    return forward_time / N


def measure_forward_backward_training(m: nn.Module, inputs: torch.Tensor, N: int = 100) -> Tuple[float, float]:
    m.train()
    forward_time = 0
    backward_time = 0
    for _ in range(N):
        time0 = time.time()
        outputs = m(inputs)
        forward_time += time.time() - time0
        
        loss = outputs.abs().mean()
        time0 = time.time()
        loss.backward()
        backward_time += time.time() - time0

    return forward_time / N, backward_time / N


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="resnet50")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--jit", action="store_true")

    return parser


def main():
    args = get_parser().parse_args()
    model_name = args.model_name
    batch_size = args.batch_size
    img_size = args.img_size
    N = args.N
    jit = args.jit

    input_shape = (batch_size, 3, img_size, img_size)

    assert hasattr(torchvision.models, model_name)
    m: nn.Module = getattr(torchvision.models, model_name)()
    m.eval()
    if jit:
        m = torch.jit.script(m)
    
    print(f"Image size: {img_size}, {img_size}")
    print(f"Batch size: {batch_size}")
    print()

    indexes = [
        "Forward (inference)",
        "Forward (training)",
        "Backward (training)"
    ]
    data_dict = {x: {} for x in indexes}

    for device in ("cpu", "mps"):
        m.to(device)
        img = get_inputs(input_shape, device)
        warmup(m, img)

        forward_eval_time = measure_forward_inference(m, img, N=N)
        forward_train_time, backward_train_time = measure_forward_backward_training(m, img, N=N)

        data_dict[indexes[0]][device] = forward_eval_time
        data_dict[indexes[1]][device] = forward_train_time
        data_dict[indexes[2]][device] = backward_train_time

    for x in data_dict.values():
        x["speedup"] = x["cpu"] / x["mps"]


    print(pd.DataFrame.from_dict(data_dict, orient="index").to_markdown())


if __name__ == "__main__":
    main()
