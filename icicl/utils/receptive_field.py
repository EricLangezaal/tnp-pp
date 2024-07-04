from typing import Tuple


def compute_swintnp_receptive_field(
    coarsen_factors: Tuple[int, ...],
    window_sizes: Tuple[int, ...],
    shift_sizes: Tuple[int, ...],
    num_layers: int,
) -> Tuple[int, ...]:
    return tuple(
        cs * (w + 2 * num_layers * (w - s))
        for w, s, cs in zip(window_sizes, shift_sizes, coarsen_factors)
    )