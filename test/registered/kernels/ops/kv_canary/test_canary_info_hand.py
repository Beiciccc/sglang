from __future__ import annotations

from typing import Optional

import pytest
import torch

from sglang.kernels.ops.kv_canary.canary_info import launch_canary_info_kernel
from sglang.kernels.ops.kv_canary.canary_info_ref import (
    launch_canary_info_kernel_torch_reference,
)
from sglang.kernels.ops.kv_canary.consts import (
    chain_digest_anchor,
    chain_digest_extend,
    sampling_digest_from_values,
    splitmix64_mix3,
)
from sglang.srt.kv_canary.canary_info.checker import CanaryInfoChecker
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kernels.kv_canary._canary_helpers import (
    make_canary_buf,
    run_canary_info_kernel_pair,
    stamp_clean_chain,
    to_signed_int64,
)

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")


_DEVICE = torch.device("cuda")

_PROMPT = [101, 102, 103]
_SLOTS = [5, 2, 9]
_TOKEN = 7
_PARAMS = SamplingParams(temperature=0.7, top_p=0.9, top_k=50, min_p=0.05)


def _golden(*, seed: Optional[int]) -> int:
    digest = chain_digest_anchor()
    for position, token in enumerate(_PROMPT):
        digest = chain_digest_extend(digest, token, position)
    sampling = sampling_digest_from_values(
        temperature=_PARAMS.temperature,
        top_p=_PARAMS.top_p,
        top_k=_PARAMS.top_k,
        min_p=_PARAMS.min_p,
        sampling_seed=0 if seed is None else seed,
        has_seed=seed is not None,
    )
    return splitmix64_mix3(digest, _TOKEN, sampling)


def _i64(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device=_DEVICE)


def _stamped_buf() -> torch.Tensor:
    buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_DEVICE)
    stamp_clean_chain(
        cuda_buf=buf,
        ref_buf=buf,
        slot_indices=_SLOTS,
        tokens=_PROMPT,
        positions=list(range(len(_PROMPT))),
    )
    return buf


def _sampler_tensors(
    num_rows: int,
    *,
    params: SamplingParams = _PARAMS,
    seed: Optional[int] = None,
) -> dict[str, Optional[torch.Tensor]]:
    def full(value, dtype: torch.dtype) -> torch.Tensor:
        return torch.full((num_rows,), value, dtype=dtype, device=_DEVICE)

    return dict(
        temperatures=full(params.temperature, torch.float32),
        top_ps=full(params.top_p, torch.float32),
        top_ks=full(params.top_k, torch.int32),
        min_ps=full(params.min_p, torch.float32),
        sampling_seed=None if seed is None else full(seed, torch.int64),
    )


def _run(
    *,
    canary_buf: torch.Tensor,
    node_index: torch.Tensor,
    node_slots: list[int],
    node_tokens: list[int],
    sampler: dict[str, Optional[torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    return run_canary_info_kernel_pair(
        launch=launch_canary_info_kernel,
        launch_ref=launch_canary_info_kernel_torch_reference,
        out_like=node_index,
        canary_buf=canary_buf,
        node_index=node_index,
        node_slots=_i64(node_slots),
        node_tokens=_i64(node_tokens),
        **sampler,
    )


def _run_per_row(
    *,
    canary_buf: torch.Tensor,
    last_slot_indices: list[int],
    next_token_ids: list[int],
    sampler: Optional[dict[str, Optional[torch.Tensor]]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows = len(last_slot_indices)
    return _run(
        canary_buf=canary_buf,
        node_index=torch.arange(num_rows, dtype=torch.int64, device=_DEVICE)[:, None],
        node_slots=last_slot_indices,
        node_tokens=next_token_ids,
        sampler=sampler if sampler is not None else _sampler_tensors(num_rows),
    )


@pytest.mark.parametrize("seed", [None, 1234])
def test_three_slot_chain_matches_host_golden(seed: Optional[int]) -> None:
    out_kernel, out_ref = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[_SLOTS[-1]],
        next_token_ids=[_TOKEN],
        sampler=_sampler_tensors(1, seed=seed),
    )
    expected = [[to_signed_int64(_golden(seed=seed))]]
    assert out_kernel.tolist() == expected
    assert out_ref.tolist() == expected


@pytest.mark.parametrize("deterministic", [False, True])
def test_kernel_output_is_accepted_by_checker(deterministic: bool) -> None:
    params = SamplingParams(
        temperature=0.7, top_p=0.9, top_k=50, min_p=0.05, sampling_seed=1234
    )
    out_kernel, _ = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[_SLOTS[-1]],
        next_token_ids=[_TOKEN],
        sampler=_sampler_tensors(
            1, params=params, seed=1234 if deterministic else None
        ),
    )
    checker = CanaryInfoChecker(
        rid="r", sampling_params=params, prompt_ids=_PROMPT, deterministic=deterministic
    )
    assert checker.advance(_TOKEN, int(out_kernel[0, 0].item())) is None


def test_intermediate_slot_gives_prefix_of_that_length() -> None:
    out_kernel, out_ref = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[_SLOTS[0]],
        next_token_ids=[_PROMPT[1]],
    )
    checker = CanaryInfoChecker(
        rid="r", sampling_params=_PARAMS, prompt_ids=_PROMPT[:1], deterministic=False
    )
    expected = [[to_signed_int64(checker.expected(_PROMPT[1]))]]
    assert out_kernel.tolist() == expected
    assert out_ref.tolist() == expected


def test_negative_last_slot_outputs_zero() -> None:
    out_kernel, out_ref = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[-1, _SLOTS[-1], -1],
        next_token_ids=[123, _TOKEN, 456],
    )
    expected = [[0], [to_signed_int64(_golden(seed=None))], [0]]
    assert out_kernel.tolist() == expected
    assert out_ref.tolist() == expected


def test_each_sampler_value_changes_output() -> None:
    num_variants = 6
    sampler = _sampler_tensors(num_variants, seed=1234)
    sampler["temperatures"].view(torch.int32)[1] ^= 1
    sampler["top_ps"].view(torch.int32)[2] ^= 1
    sampler["top_ks"][3] ^= 1
    sampler["min_ps"].view(torch.int32)[4] ^= 1
    sampler["sampling_seed"][5] ^= 1
    out_kernel, out_ref = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[_SLOTS[-1]] * num_variants,
        next_token_ids=[_TOKEN] * num_variants,
        sampler=sampler,
    )
    assert out_kernel.tolist() == out_ref.tolist()
    values = [row[0] for row in out_kernel.tolist()]
    assert len(set(values)) == len(values)


def test_seed_presence_changes_output() -> None:
    without_seed, _ = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[_SLOTS[-1]],
        next_token_ids=[_TOKEN],
        sampler=_sampler_tensors(1, seed=None),
    )
    zero_seed, _ = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[_SLOTS[-1]],
        next_token_ids=[_TOKEN],
        sampler=_sampler_tensors(1, seed=0),
    )
    assert without_seed.tolist() != zero_seed.tolist()


def test_token_and_prefix_change_output() -> None:
    buf = _stamped_buf()
    out_kernel, _ = _run_per_row(
        canary_buf=buf,
        last_slot_indices=[_SLOTS[-1], _SLOTS[-1]],
        next_token_ids=[_TOKEN, _TOKEN + 1],
    )
    assert out_kernel[0, 0].item() != out_kernel[1, 0].item()

    buf.view(torch.int64)[_SLOTS[-1], 0] += 1
    tampered, tampered_ref = _run_per_row(
        canary_buf=buf,
        last_slot_indices=[_SLOTS[-1]],
        next_token_ids=[_TOKEN],
    )
    assert tampered.tolist() == tampered_ref.tolist()
    assert tampered[0, 0].item() != out_kernel[0, 0].item()


def test_empty_batch_is_noop() -> None:
    out_kernel, out_ref = _run_per_row(
        canary_buf=_stamped_buf(),
        last_slot_indices=[],
        next_token_ids=[],
    )
    assert out_kernel.numel() == 0
    assert out_ref.numel() == 0


def test_wide_node_index_gathers_nodes_and_pads_with_zero() -> None:
    buf = _stamped_buf()
    node_slots = [_SLOTS[0], _SLOTS[1], _SLOTS[2], -1]
    node_tokens = [_PROMPT[1], _PROMPT[2], _TOKEN, 0]
    out_kernel, out_ref = _run(
        canary_buf=buf,
        node_index=_i64([[0, 1, 2], [2, -1, -1], [3, -1, -1]]),
        node_slots=node_slots,
        node_tokens=node_tokens,
        sampler=_sampler_tensors(3),
    )
    flat, _ = _run_per_row(
        canary_buf=buf,
        last_slot_indices=node_slots[:3],
        next_token_ids=node_tokens[:3],
    )
    flat_values = [row[0] for row in flat.tolist()]
    expected = [flat_values, [flat_values[2], 0, 0], [0, 0, 0]]
    assert out_kernel.tolist() == expected
    assert out_ref.tolist() == expected
    assert expected[0][2] == to_signed_int64(_golden(seed=None))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
