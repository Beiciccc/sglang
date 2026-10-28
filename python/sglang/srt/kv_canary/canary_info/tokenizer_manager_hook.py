from __future__ import annotations

import logging
from array import array
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

from sglang.srt.kv_canary.canary_info.aux_output import CANARY_INFOS_KEY
from sglang.srt.kv_canary.canary_info.checker import (
    CANARY_INFO_MISMATCH,
    CanaryInfoChecker,
    CanaryInfoMismatch,
)
from sglang.srt.kv_canary.config import CanaryMode, kv_canary_mode
from sglang.srt.managers.io_struct import (
    BatchStrOutput,
    BatchTokenIDOutput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import FINISH_ABORT
from sglang.srt.runtime_context import get_exec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CanaryInfoReqState:
    checker: CanaryInfoChecker
    mismatch: Optional[CanaryInfoMismatch] = None


class CanaryInfoTokenizerManagerHook:
    def __init__(self, *, abort_on_mismatch: bool) -> None:
        self._abort_on_mismatch = abort_on_mismatch
        self._deterministic = get_exec().deterministic.enable_deterministic_inference

    @classmethod
    def maybe_create(cls) -> Optional[CanaryInfoTokenizerManagerHook]:
        mode = kv_canary_mode()
        if mode is CanaryMode.NONE:
            return None
        return cls(abort_on_mismatch=mode is CanaryMode.RAISE)

    def on_request_dispatched(self, *, rid_to_state: dict, tokenized_obj: Any) -> None:
        if not isinstance(tokenized_obj, TokenizedGenerateReqInput):
            return
        if (state := rid_to_state.get(tokenized_obj.rid)) is None:
            return
        checker = _build_canary_info_checker(
            tokenized_obj, deterministic=self._deterministic
        )
        state.canary_info = CanaryInfoReqState(checker=checker)

    def on_batch_output(
        self,
        *,
        state: Any,
        rid: str,
        recv_obj: Any,
        customized_info: Optional[dict],
        index: int,
        abort_request: Callable[[str], Any],
    ) -> None:
        canary_info = state.canary_info
        assert canary_info is not None
        if (mismatch := canary_info.mismatch) is None:
            mismatch = _check_canary_infos(
                checker=canary_info.checker,
                token_ids=recv_obj.output_ids[index],
                canary_infos=_canary_infos_at(customized_info, index),
            )
            if mismatch is None:
                return
            logger.error(mismatch.format())
            state.canary_info = CanaryInfoReqState(
                checker=canary_info.checker, mismatch=mismatch
            )
            if self._abort_on_mismatch:
                abort_request(rid)
        if self._abort_on_mismatch:
            _withhold_mismatched_output(
                recv_obj=recv_obj,
                customized_info=customized_info,
                index=index,
                mismatch=mismatch,
            )


def _build_canary_info_checker(
    tokenized_obj: TokenizedGenerateReqInput, *, deterministic: bool
) -> CanaryInfoChecker:
    return CanaryInfoChecker(
        rid=tokenized_obj.rid,
        sampling_params=tokenized_obj.sampling_params,
        prompt_ids=tokenized_obj.input_ids,
        deterministic=deterministic,
    )


def _check_canary_infos(
    *,
    checker: CanaryInfoChecker,
    token_ids: Sequence[int],
    canary_infos: Optional[Sequence[Optional[int]]],
) -> Optional[CanaryInfoMismatch]:
    if len(token_ids) == 0:
        return None
    if canary_infos is None or len(canary_infos) != len(token_ids):
        first_token = int(token_ids[0])
        return CanaryInfoMismatch(
            rid=checker.rid,
            position=checker.position,
            token_id=first_token,
            expected=checker.expected(first_token),
            actual=None,
        )
    return checker.advance_many(token_ids, canary_infos)


def _canary_infos_at(
    customized_info: Optional[dict], index: int
) -> Optional[Sequence[Optional[int]]]:
    if customized_info is None or CANARY_INFOS_KEY not in customized_info:
        return None
    return customized_info[CANARY_INFOS_KEY][index]


def _withhold_mismatched_output(
    *,
    recv_obj: Union[BatchStrOutput, BatchTokenIDOutput],
    customized_info: Optional[dict],
    index: int,
    mismatch: CanaryInfoMismatch,
) -> None:
    recv_obj.finished_reasons[index] = _canary_info_mismatch_finish_reason(mismatch)
    recv_obj.output_ids[index] = array("q")
    if customized_info is not None and CANARY_INFOS_KEY in customized_info:
        customized_info[CANARY_INFOS_KEY][index] = []
    if isinstance(recv_obj, BatchStrOutput):
        recv_obj.output_strs[index] = ""


def _canary_info_mismatch_finish_reason(mismatch: CanaryInfoMismatch) -> dict:
    return FINISH_ABORT(
        message=CANARY_INFO_MISMATCH, err_type="KvCanaryInfoMismatch"
    ).to_json() | {"position": mismatch.position}
