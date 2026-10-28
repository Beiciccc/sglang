from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.kernels.ops.kv_canary._dispatch import use_torch_reference

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def launch_canary_info_kernel(
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
    num_rows = int(node_index.shape[0])
    if tuple(out.shape) != tuple(node_index.shape):
        raise ValueError(
            f"kv-canary: out must match node_index shape {tuple(node_index.shape)}, "
            f"got {tuple(out.shape)}"
        )
    if int(node_slots.shape[0]) != int(node_tokens.shape[0]):
        raise ValueError(
            "kv-canary: node_slots and node_tokens must share their length"
        )
    for name, tensor in (
        ("temperatures", temperatures),
        ("top_ps", top_ps),
        ("top_ks", top_ks),
        ("min_ps", min_ps),
    ):
        if tensor.numel() != num_rows:
            raise ValueError(
                f"kv-canary: {name} must hold {num_rows} rows, got {tensor.numel()}"
            )

    node_index = _as_int64(node_index)
    node_slots = _as_int64(node_slots)
    node_tokens = _as_int64(node_tokens)
    temperatures = temperatures.reshape(-1).to(torch.float32).contiguous()
    top_ps = top_ps.reshape(-1).to(torch.float32).contiguous()
    top_ks = top_ks.reshape(-1).to(torch.int32).contiguous()
    min_ps = min_ps.reshape(-1).to(torch.float32).contiguous()
    has_seed = sampling_seed is not None
    seed = _as_int64(sampling_seed.reshape(-1)) if has_seed else node_tokens

    if use_torch_reference(canary_buf.device):
        from sglang.kernels.ops.kv_canary.canary_info_ref import (
            launch_canary_info_kernel_torch_reference,
        )

        launch_canary_info_kernel_torch_reference(
            canary_buf=canary_buf,
            node_index=node_index,
            node_slots=node_slots,
            node_tokens=node_tokens,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
            min_ps=min_ps,
            sampling_seed=seed if has_seed else None,
            out=out,
        )
        return

    _jit_canary_info_module().canary_info_cuda(
        canary_buf,
        node_index,
        node_slots,
        node_tokens,
        temperatures,
        top_ps,
        top_ks,
        min_ps,
        seed,
        int(has_seed),
        out,
    )


def _as_int64(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(torch.int64).contiguous()


@cache_once
def _jit_canary_info_module() -> Module:
    return load_jit(
        "kv_canary_info",
        cuda_files=["kv_canary/canary_info.cuh"],
        cuda_wrappers=[("canary_info_cuda", "canary::canary_info_cuda")],
    )
