import argparse
import itertools
import logging
import time
from typing import Any, List, Tuple

import pandas as pd
import torch
import torchvision
from torch import nn
from transformers import AutoConfig, AutoModel


def get_model_and_inputs(
    model_name: str, jit: bool, batch_size: int, size: int, device: str
) -> nn.Module:
    if hasattr(torchvision.models, model_name):
        m: nn.Module = getattr(torchvision.models, model_name)()
        inputs = torch.randn((batch_size, 3, size, size), device=device)
    else:
        try:
            config = AutoConfig.from_pretrained(model_name)
            m = AutoModel.from_config(config)
            inputs = torch.randint(1000, size=(batch_size, size), device=device)
        except Exception as e:
            logging.warning(e)

    m.eval()
    if jit:
        m = torch.jit.script(m)
    return m, inputs


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

    return N / forward_time


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

        if hasattr(outputs, "pooler_output"):
            outputs = outputs.pooler_output
        loss = outputs.abs().mean()

        time0 = time.time()
        loss.backward()
        backward_time += time.time() - time0

    return N / forward_time, N / backward_time


def format_data(data: List[List[Any]], columns: List[str], batch_sizes: List[int]):
    df = pd.DataFrame(data, columns=columns)
    df["device"] = df["device"].apply(lambda x: f"{x} (it/s)")

    df = df.set_index(["device", "batch size"]).sort_index().T
    df[[("speedup", b) for b in batch_sizes]] = df["mps (it/s)"] / df["cpu (it/s)"]
    df = df.T.unstack().T

    df.columns.name = None
    new_index = [(x, f"batch {b}") for x, b in df.index]
    df.index = pd.MultiIndex.from_tuples(new_index)
    return df


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="resnet50")
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--batch_sizes", default="1,4,16")
    parser.add_argument("--N", type=int, default=100)
    parser.add_argument("--jit", action="store_true")

    return parser


def main():
    args = get_parser().parse_args()
    model_name = args.model_name
    size = args.size
    N = args.N
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    jit = args.jit

    print(f"Input size: {size}")
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
        m, inputs = get_model_and_inputs(model_name, jit, batch_size, size, device)
        m.to(device)

        try:
            warmup(m, inputs)

            forward_eval_speed = measure_forward_inference(m, inputs, N=N)
            forward_train_speed, backward_train_speed = measure_forward_backward_training(
                m, inputs, N=N
            )

            sample = [
                device,
                batch_size,
                forward_eval_speed,
                forward_train_speed,
                backward_train_speed,
            ]
            data.append(sample)

            logging.info("Sleeping for 30s")
            time.sleep(30)
        
        except Exception as e:
            logging.warning(f"Failed to measure. Exception={e}")

        del m
        del inputs

    df = format_data(data, columns, batch_sizes)
    print(df)
    print(df.to_html())


if __name__ == "__main__":
    main()
