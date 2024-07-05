import torch


def adjust_num_batches(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    num_workers = worker_info.num_workers
    num_batches = worker_info.dataset.num_batches

    adjusted_num_batches = max(1, num_batches // num_workers)
    if worker_id >= num_batches:
        adjusted_num_batches = 0

    elif num_workers < num_batches and worker_id < num_batches % num_workers:
        adjusted_num_batches += 1

    # print(f"Adjusting worker {worker_id} num_batches from {num_batches} to {adjusted_num_batches}.")
    worker_info.dataset.num_batches = adjusted_num_batches