from __future__ import annotations

from typing import TYPE_CHECKING, Final

from sglang.kernels.ops.kv_canary.consts import sampling_digest_from_values

if TYPE_CHECKING:
    from sglang.srt.sampling.sampling_params import SamplingParams

DEFAULT_SAMPLING_SEED: Final[int] = 42


def sampling_digest(params: SamplingParams, *, deterministic: bool) -> int:
    return sampling_digest_from_values(
        temperature=params.temperature,
        top_p=params.top_p,
        top_k=params.top_k,
        min_p=params.min_p,
        sampling_seed=(
            params.sampling_seed
            if params.sampling_seed is not None
            else DEFAULT_SAMPLING_SEED
        ),
        has_seed=deterministic,
    )
