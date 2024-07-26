from typing import List
import torch
from icicl.data.era5 import BaseERA5DataGenerator

def adjust_num_batches(dask_workers: List, worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    num_batches = worker_info.dataset.num_batches

    adjusted_num_batches = max(1, num_batches // num_workers)
    if worker_id >= num_batches:
        adjusted_num_batches = 0

    elif num_workers < num_batches and worker_id < num_batches % num_workers:
        adjusted_num_batches += 1

    # print(f"Adjusting worker {worker_id} num_batches from {num_batches} to {adjusted_num_batches}.")
    dataset = worker_info.dataset
    dataset.num_batches = adjusted_num_batches

    if isinstance(dataset, BaseERA5DataGenerator) and dataset.distributed:
        args = dataset.get_data_loader_args(worker_id, num_workers)
        print(f"Worker {worker_id} has config {args}.")
        print(dask_workers[worker_id])
        dataset.load_data(dask_worker=dask_workers[worker_id], **args)
