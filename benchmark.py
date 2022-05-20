import argparse
import itertools
import time
from typing import Any, List, Tuple

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
def measure_forward_inference(
    m: nn.Module, inputs: torch.Tensor, N: int = 100
) -> float:
    m.eval()
    forward_time = 0
    for _ in range(N):
        time0 = time.time()
        m(inputs)
        forward_time += time.time() - time0

    return forward_time / N


def measure_forward_backward_training(
    m: nn.Module, inputs: torch.Tensor, N: int = 100
) -> Tuple[float, float]:
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
    parser.add_argument("--batch_sizes", default="1,4,16")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--jit", action="store_true")

    return parser


def format_data(data: List[List[Any]], columns: List[str], batch_sizes: List[int]):
    df = pd.DataFrame(data, columns=columns)
    df = df.set_index(["device", "batch size"]).sort_index().T
    df[[("speedup", b) for b in batch_sizes]] = df["cpu"] / df["mps"]
    df = df.T.unstack().T

    df.columns.name = None
    new_index = [(x, f"batch {b}") for x, b in df.index]
    df.index = pd.MultiIndex.from_tuples(new_index)
    return df


def main():
    args = get_parser().parse_args()
    model_name = args.model_name
    img_size = args.img_size
    N = args.N
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    jit = args.jit

    assert hasattr(torchvision.models, model_name)
    m: nn.Module = getattr(torchvision.models, model_name)()
    m.eval()
    if jit:
        m = torch.jit.script(m)

    print(f"Image size: {img_size}, {img_size}")
    print()

    columns = [
        "device",
        "batch size",
        "Forward (inference)",
        "Forward (training)",
        "Backward (training)",
    ]
    data = []

    devices = ("cpu", "mps")
    for device, batch_size in itertools.product(devices, batch_sizes):
        print(f"Measuring device={device}, batch_size={batch_size}")

        m.to(device)
        input_shape = (batch_size, 3, img_size, img_size)
        img = get_inputs(input_shape, device)
        warmup(m, img)

        forward_eval_time = measure_forward_inference(m, img, N=N)
        forward_train_time, backward_train_time = measure_forward_backward_training(
            m, img, N=N
        )

        sample = [
            device,
            batch_size,
            forward_eval_time,
            forward_train_time,
            backward_train_time,
        ]
        data.append(sample)

    df = format_data(data, columns, batch_sizes)
    print(df)
    print(df.to_html())


if __name__ == "__main__":
    main()
