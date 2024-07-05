import torch


def adjust_num_batches(worker_id: int):
    worker_info = torch.utils.data.get_worker_info()
    num_batches = worker_info.dataset.num_batches

    adjusted_num_batches = max(1, num_batches // worker_info.num_workers)
    if worker_id * adjusted_num_batches >= num_batches:
        adjusted_num_batches = 0
    elif worker_id == worker_info.num_workers - 1:
        # also add the remainder here
        adjusted_num_batches += num_batches % worker_info.num_workers

    print(
        f"Adjusting worker {worker_id} num_batches from {num_batches} to {adjusted_num_batches}."
    )
    worker_info.dataset.num_batches = adjusted_num_batches