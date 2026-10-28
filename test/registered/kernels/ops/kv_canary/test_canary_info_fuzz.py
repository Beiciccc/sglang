from __future__ import annotations

import random

import pytest
import torch

from sglang.kernels.ops.kv_canary.canary_info import launch_canary_info_kernel
from sglang.kernels.ops.kv_canary.canary_info_ref import (
    launch_canary_info_kernel_torch_reference,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.kv_canary._canary_helpers import (
    make_canary_buf,
    run_canary_info_kernel_pair,
    stamp_clean_chain,
)
from sglang.test.kernels.kv_canary._fuzz_driver import FUZZ_SEEDS_PR

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=30, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")

_FUZZ_ITER_PER_SEED = 20
_NUM_SLOTS = 2048


def _random_sampler(rng: random.Random, rows: int) -> dict:
    def f32(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=_DEVICE)

    has_seed = rng.random() < 0.5
    return dict(
        temperatures=f32(
            [rng.choice([0.0, 1.0, rng.uniform(0.01, 2.0)]) for _ in range(rows)]
        ),
        top_ps=f32([rng.choice([1.0, rng.uniform(0.01, 1.0)]) for _ in range(rows)]),
        top_ks=torch.tensor(
            [
                rng.choice([-1, 1, 1 << 30, rng.randint(-(1 << 31), (1 << 31) - 1)])
                for _ in range(rows)
            ],
            dtype=torch.int32,
            device=_DEVICE,
        ),
        min_ps=f32([rng.choice([0.0, rng.uniform(0.0, 1.0)]) for _ in range(rows)]),
        sampling_seed=(
            torch.tensor(
                [rng.randint(-(1 << 63), (1 << 63) - 1) for _ in range(rows)],
                dtype=torch.int64,
                device=_DEVICE,
            )
            if has_seed
            else None
        ),
    )


def _random_chain_buf(
    rng: random.Random, num_chains: int
) -> tuple[torch.Tensor, list[int]]:
    buf = make_canary_buf(num_slots=_NUM_SLOTS, slot_stride_bytes=32, device=_DEVICE)
    free_slots = list(range(1, _NUM_SLOTS))
    rng.shuffle(free_slots)
    last_slots: list[int] = []
    cursor = 0
    for _ in range(num_chains):
        chain_len = rng.randint(1, 3)
        if rng.random() < 0.1 or cursor + chain_len > len(free_slots):
            last_slots.append(-1)
            continue
        slots = free_slots[cursor : cursor + chain_len]
        cursor += chain_len
        start = rng.randint(0, 4096)
        stamp_clean_chain(
            cuda_buf=buf,
            ref_buf=buf,
            slot_indices=slots,
            tokens=[rng.randint(0, 200000) for _ in slots],
            positions=list(range(start, start + chain_len)),
        )
        last_slots.append(slots[-1])
    return buf, last_slots


def _run_one(rng: random.Random, *, wide: bool) -> None:
    batch_size = rng.randint(1, 128 if wide else 512)
    width = rng.randint(1, 8) if wide else 1
    num_nodes = rng.randint(1, batch_size * width) if wide else batch_size
    buf, node_slots = _random_chain_buf(rng, num_nodes)
    if wide:
        node_index = torch.tensor(
            [
                [rng.choice([-1, rng.randrange(num_nodes)]) for _ in range(width)]
                for _ in range(batch_size)
            ],
            dtype=torch.int64,
        ).to(_DEVICE)
    else:
        node_index = torch.arange(batch_size, dtype=torch.int64, device=_DEVICE)[
            :, None
        ]
    slots = torch.tensor(node_slots, dtype=torch.int64, device=_DEVICE)
    out_kernel, out_ref = run_canary_info_kernel_pair(
        launch=launch_canary_info_kernel,
        launch_ref=launch_canary_info_kernel_torch_reference,
        out_like=node_index,
        canary_buf=buf,
        node_index=node_index,
        node_slots=slots,
        node_tokens=torch.tensor(
            [rng.randint(0, 200000) for _ in range(num_nodes)],
            dtype=torch.int64,
            device=_DEVICE,
        ),
        **_random_sampler(rng, batch_size),
    )

    assert torch.equal(out_kernel, out_ref)
    invalid = (node_index < 0) | (slots[node_index.clamp(min=0)] < 0)
    assert bool((out_kernel[invalid] == 0).all())


@pytest.mark.parametrize("seed", FUZZ_SEEDS_PR)
def test_canary_info_kernel_per_row_matches_reference(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(_FUZZ_ITER_PER_SEED):
        _run_one(rng, wide=False)


@pytest.mark.parametrize("seed", FUZZ_SEEDS_PR)
def test_canary_info_kernel_wide_node_index_matches_reference(seed: int) -> None:
    rng = random.Random(seed)
    for _ in range(_FUZZ_ITER_PER_SEED):
        _run_one(rng, wide=True)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
