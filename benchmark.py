import argparse
import itertools
import logging
import subprocess
import sys
import time
from typing import Any, List, Tuple

import pandas as pd
import torch
import torchvision
import transformers
from torch import nn
from transformers import AutoConfig, AutoModel

logging.basicConfig(
    level="INFO",
    format="[%(asctime)s - %(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


class BertWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.output_hidden_states = True
        self.model = AutoModel.from_config(config, add_pooling_layer=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).last_hidden_state[:, 0]


def get_git_commit_hash():
    cmd = "git rev-parse --short HEAD"
    proc = subprocess.run(cmd, shell=True, capture_output=True)
    return proc.stdout.decode().rstrip()


def log_system_info() -> None:
    logging.info(f"Python {sys.version}")
    logging.info("Packages version:")
    logging.info(f"  - Git commit: {get_git_commit_hash()}")
    logging.info(f"  - torch={torch.__version__}")
    logging.info(f"  - torchvision={torchvision.__version__}")
    logging.info(f"  - transformers={transformers.__version__}")
    logging.info("")


def get_model_and_inputs(
    model_name: str, jit: bool, size: int
) -> Tuple[nn.Module, torch.Tensor]:

    if hasattr(torchvision.models, model_name):
        m = getattr(torchvision.models, model_name)()
        inputs = torch.randn((1, 3, size, size))
        model_source = "torchvision"

    elif model_name.startswith("bert"):
        config = AutoConfig.from_pretrained(model_name)
        m = BertWrapper(config)
        inputs = torch.randint(config.vocab_size, size=(1, size))
        model_source = "Hugging Face"

    else:
        raise ValueError(f"{model_name} is not supported")

    m.eval()
    num_params = sum(p.numel() for p in m.parameters())
    if jit:
        m = torch.jit.script(m)

    logging.info(f"Using {model_name} from {model_source}")
    logging.info(f"  - Num params: {num_params}")
    logging.info(f"  - Input shape: {tuple(inputs.shape)}")
    logging.info("")

    return m, inputs


def warmup(m: nn.Module, inputs: torch.Tensor, N: int = 10) -> None:
    m.eval()
    for _ in range(N):
        m(inputs)


@torch.inference_mode()
def measure_eval(m: nn.Module, inputs: torch.Tensor, N: int = 100) -> float:
    m.eval()
    forward_time = 0
    for _ in range(N):
        time0 = time.time()
        m(inputs)
        forward_time += time.time() - time0

    return N / forward_time


def measure_train(
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
    df = df.round(3)
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
    log_system_info()

    args = get_parser().parse_args()
    model_name = args.model_name
    size = args.size
    N = args.N
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    jit = args.jit

    m, inputs = get_model_and_inputs(model_name, jit, size)

    columns = [
        "device",
        "batch size",
        "Forward (inference)",
        "Forward (training)",
        "Backward (training)",
    ]
    data = []

    devices = ("mps", "cpu")
    for batch_size, device in itertools.product(batch_sizes, devices):
        logging.info(f"Measuring device={device}, batch_size={batch_size}")
        m.to(device)
        new_shape = (batch_size,) + inputs.shape[1:]
        inputs_batch = inputs.expand(new_shape).to(device)

        warmup(m, inputs_batch)

        f_eval_speed = measure_eval(m, inputs_batch, N=N)
        f_train_speed, b_train_speed = measure_train(m, inputs_batch, N=N)

        sample = [
            device,
            batch_size,
            f_eval_speed,
            f_train_speed,
            b_train_speed,
        ]
        data.append(sample)

    df = format_data(data, columns, batch_sizes)
    logging.info("Results in HTML table format\n" + df.to_html())
    logging.info("\n" + str(df))


if __name__ == "__main__":
    main()
