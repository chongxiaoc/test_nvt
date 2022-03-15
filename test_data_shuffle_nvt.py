import argparse
import cudf
import numpy as np
import nvtabular as nvt
import time
import torch

from dataloader_bench.data_generation import (generate_data, DATA_SPEC)
from nvtabular.loader.torch import TorchAsyncItr as NVTDataLoader

import rmm
rmm.reinitialize(pool_allocator=True,
                 initial_pool_size=None, # Use default size
                )

DEFAULT_DATA_DIR = "hdfs:///user/chongxiaoc/data_bench/tmp"

numpy_to_torch_dtype = {
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128
}

# Training settings
parser = argparse.ArgumentParser(description="Parquet File Generator")
parser.add_argument(
    "--batch-size",
    type=int,
    default=50000,
    metavar="N",
    help="input batch size for training (default: 50000)")
parser.add_argument(
    "--epochs",
    type=int,
    default=1,
    metavar="N",
    help="number of epochs (default: 1)")
# Synthetic training data generation settings.
parser.add_argument("--num-rows", type=int, default=2 * (10**7))
parser.add_argument("--num-files", type=int, default=25)
parser.add_argument("--max-row-group-skew", type=float, default=0.0)
parser.add_argument("--num-row-groups-per-file", type=int, default=5)
parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
parser.add_argument("--gpu-mem-frac", type=float, default=0.20)
parser.add_argument("--num-devices", type=int, default=1)
parser.add_argument("--skip", help="skip file generation", type=bool, default=False)


def human_readable_size(num, precision=1, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0 or unit == "Zi":
            break
        num /= 1024.0
    return f"{num:.{precision}f}{unit}{suffix}"


if __name__ == "__main__":
    args = parser.parse_args()

    num_rows = args.num_rows
    num_files = args.num_files
    num_row_groups_per_file = args.num_row_groups_per_file
    max_row_group_skew = args.max_row_group_skew
    data_dir = args.data_dir
    batch_size = args.batch_size
    epochs = args.epochs

    if not args.skip:
        filenames = None
        print(f"Generating {num_rows} rows over {num_files} files, with "
              f"{num_row_groups_per_file} row groups per file and at most "
              f"{100 * max_row_group_skew:.1f}% row group skew.")
        filenames, num_bytes = generate_data(num_rows, num_files,
                                             num_row_groups_per_file,
                                             max_row_group_skew, data_dir)
        print(f"Generated {len(filenames)} files containing {num_rows} rows "
              f"with {num_row_groups_per_file} row groups per file, totalling "
              f"{human_readable_size(num_bytes)}.")

    df = cudf.read_parquet(data_dir+"/*")
    #df = pd.read_parquet(data_dir)
    #train_set = nvt.Dataset(data_dir, engine="parquet", part_mem_fraction=float(args.gpu_mem_frac))
    ds = nvt.Dataset(df)

    cont_names = ["embeddings_name0", "embeddings_name1", "embeddings_name2", "embeddings_name3",
                   "one_hot0", "one_hot1"]
    label_name = "labels"

    kwargs = {
        "conts": cont_names,
        "labels": [label_name],
        "device": 0,
    }

    dl = NVTDataLoader(ds, batch_size=batch_size, shuffle=True, **kwargs)

    samples_seen = 0
    start_time = time.time()
    for _ in range(epochs):
        for X in dl:
            num_samples = X[-1].size()[0]
            samples_seen += num_samples
            throughput = samples_seen / (time.time() - start_time)
    print("NVT samples seen:", samples_seen, "throughput:", throughput)
