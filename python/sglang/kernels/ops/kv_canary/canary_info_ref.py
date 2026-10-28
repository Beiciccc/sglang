from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.ops.kv_canary.consts import (
    canary_info,
    sampling_digest_from_values,
    to_signed_int64,
)
from sglang.kernels.ops.kv_canary.verify_ref import compute_slot_hash


def launch_canary_info_kernel_torch_reference(
    *,
    canary_buf: torch.Tensor,
    node_index: torch.Tensor,
    node_slots: torch.Tensor,
    node_tokens: torch.Tensor,
    temperatures: torch.Tensor,
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    min_ps: torch.Tensor,
    sampling_seed: Optional[torch.Tensor],
    out: torch.Tensor,
) -> None:
    buf_i64 = canary_buf.detach().cpu().contiguous().view(torch.int64)
    slots = node_slots.detach().cpu().tolist()
    tokens = node_tokens.detach().cpu().tolist()
    temperature_list = temperatures.detach().cpu().tolist()
    top_p_list = top_ps.detach().cpu().tolist()
    top_k_list = top_ks.detach().cpu().tolist()
    min_p_list = min_ps.detach().cpu().tolist()
    seed_list = None if sampling_seed is None else sampling_seed.detach().cpu().tolist()
    values = []
    for row, nodes in enumerate(node_index.detach().cpu().tolist()):
        sampling = sampling_digest_from_values(
            temperature=temperature_list[row],
            top_p=top_p_list[row],
            top_k=top_k_list[row],
            min_p=min_p_list[row],
            sampling_seed=0 if seed_list is None else seed_list[row],
            has_seed=seed_list is not None,
        )
        values.append(
            [
                (
                    to_signed_int64(
                        canary_info(
                            compute_slot_hash(buf_i64, slots[node]),
                            tokens[node],
                            sampling,
                        )
                    )
                    if node >= 0 and slots[node] >= 0
                    else 0
                )
                for node in nodes
            ]
        )
    out.copy_(torch.tensor(values, dtype=torch.int64).view(out.shape).to(out.device))
