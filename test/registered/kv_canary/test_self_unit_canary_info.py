from __future__ import annotations

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.kernels.ops.kv_canary.consts import (
    RealKvHashMode,
    chain_digest_anchor,
    chain_digest_extend,
    sampling_digest_from_values,
    splitmix64_mix3,
    to_signed_int64,
)
from sglang.kernels.ops.kv_canary.verify import CanaryLaunchTag, VerifyOrWriteContext
from sglang.kernels.ops.kv_canary.verify_ref import compute_slot_hash
from sglang.kernels.ops.kv_canary.write_ref import (
    launch_canary_write_kernel_torch_reference,
)
from sglang.srt.kv_canary.canary_info.aux_output import (
    CANARY_INFOS_KEY,
    CanaryInfoHostOutput,
)
from sglang.srt.kv_canary.canary_info.checker import (
    CANARY_INFO_MISMATCH,
    CanaryInfoChecker,
    CanaryInfoMismatch,
)
from sglang.srt.kv_canary.canary_info.digest import (
    DEFAULT_SAMPLING_SEED,
    sampling_digest,
)
from sglang.srt.kv_canary.canary_info.tokenizer_manager_hook import (
    CanaryInfoReqState,
    CanaryInfoTokenizerManagerHook,
    _build_canary_info_checker,
    _check_canary_infos,
)
from sglang.srt.managers.auxiliary_output import CommittedTokens
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.managers.tokenizer_manager import ReqState
from sglang.srt.sampling.sampling_params import TOP_K_ALL, SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.kv_canary._canary_helpers import (
    FakeViolationLog,
    make_canary_buf,
    make_write_plan,
)
from sglang.test.kv_canary.canary_info_utils import reference_canary_info_stream
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_CPU = torch.device("cpu")

_GOLDEN_CASES: tuple[tuple[list[int], int, dict, bool], ...] = (
    (
        [1],
        2,
        dict(temperature=1.0, top_p=1.0, top_k=TOP_K_ALL, min_p=0.0),
        False,
    ),
    (
        [101, 102, 103],
        7,
        dict(temperature=0.7, top_p=0.9, top_k=50, min_p=0.05),
        False,
    ),
    (
        [5, 6, 7, 8, 9],
        151643,
        dict(temperature=1.0, top_p=1.0, top_k=1, min_p=0.0, sampling_seed=1234),
        True,
    ),
    (
        [42] * 16,
        0,
        dict(temperature=1.3, top_p=0.5, top_k=20, min_p=0.0, sampling_seed=0),
        True,
    ),
    (
        list(range(1000, 1033)),
        99999,
        dict(temperature=0.2, top_p=1.0, top_k=TOP_K_ALL, min_p=0.1),
        True,
    ),
)


def _golden_canary_info(
    *, prompt: list[int], token: int, params_kwargs: dict, deterministic: bool
) -> int:
    digest = chain_digest_anchor()
    for position, prompt_token in enumerate(prompt):
        digest = chain_digest_extend(digest, prompt_token, position)
    seed = params_kwargs.get("sampling_seed")
    sampling = sampling_digest_from_values(
        temperature=params_kwargs["temperature"],
        top_p=params_kwargs["top_p"],
        top_k=params_kwargs["top_k"],
        min_p=params_kwargs["min_p"],
        sampling_seed=(DEFAULT_SAMPLING_SEED if seed is None else seed)
        if deterministic
        else 0,
        has_seed=deterministic,
    )
    return splitmix64_mix3(digest, token, sampling)


def _make_checker(
    prompt: list[int], params: SamplingParams, *, deterministic: bool = False
) -> CanaryInfoChecker:
    return CanaryInfoChecker(
        rid="r0", sampling_params=params, prompt_ids=prompt, deterministic=deterministic
    )


class TestCanaryInfoDigest(CustomTestCase):
    def test_prefix_digest_matches_write_ref_chain(self) -> None:
        tokens = [11, 22, 33, 44, 55, 66, 77]
        slots = [3, 9, 1, 14, 6, 2, 12]
        buf = make_canary_buf(num_slots=16, slot_stride_bytes=32, device=_CPU)
        log = FakeViolationLog.allocate(device=_CPU)
        launch_canary_write_kernel_torch_reference(
            context=VerifyOrWriteContext(
                canary_buf=buf,
                kernel_kind=CanaryLaunchTag.TAIL_K_FULL,
                violation_ring=log.ring,
                violation_write_index=log.write_index,
                slot_run_counter=log.slot_run_counter,
                kernel_run_counter=log.kernel_run_counter,
                enable_chain_position_assert=log.enable_chain_position_assert,
                real_kv_sources=(),
                real_kv_hash_mode=RealKvHashMode.NONE,
            ),
            plan=make_write_plan(
                write_offsets=[0, len(tokens)],
                seed_slot_indices=[-1],
                num_valid_reqs=1,
                device=_CPU,
            ),
            input_ids=torch.tensor(tokens, dtype=torch.int64),
            positions=torch.arange(len(tokens), dtype=torch.int64),
            out_cache_loc=torch.tensor(slots, dtype=torch.int64),
            enable_write_input_assert=False,
            expected_input_tokens=None,
            expected_input_positions=None,
        )

        buf_i64 = buf.view(torch.int64)
        digest = chain_digest_anchor()
        self.assertEqual(compute_slot_hash(buf_i64, -1), digest)
        for position, (token, slot) in enumerate(zip(tokens, slots)):
            digest = chain_digest_extend(digest, token, position)
            self.assertEqual(compute_slot_hash(buf_i64, slot), digest)

    def test_golden_vectors(self) -> None:
        for prompt, token, params_kwargs, deterministic in _GOLDEN_CASES:
            with self.subTest(prompt_len=len(prompt), token=token):
                checker = _make_checker(
                    prompt, SamplingParams(**params_kwargs), deterministic=deterministic
                )
                expected = _golden_canary_info(
                    prompt=prompt,
                    token=token,
                    params_kwargs=params_kwargs,
                    deterministic=deterministic,
                )
                self.assertEqual(checker.expected(token), expected)

    def test_sampling_digest_ignores_seed_when_not_deterministic(self) -> None:
        unseeded = SamplingParams(temperature=0.8, top_p=0.9)
        seeded = SamplingParams(temperature=0.8, top_p=0.9, sampling_seed=1234)
        self.assertEqual(
            sampling_digest(unseeded, deterministic=False),
            sampling_digest(seeded, deterministic=False),
        )
        self.assertEqual(
            sampling_digest(seeded, deterministic=False),
            sampling_digest_from_values(
                temperature=0.8,
                top_p=0.9,
                top_k=seeded.top_k,
                min_p=seeded.min_p,
                sampling_seed=0,
                has_seed=False,
            ),
        )

    def test_sampling_digest_treats_missing_seed_as_default_when_deterministic(
        self,
    ) -> None:
        unseeded = SamplingParams(temperature=0.8, top_p=0.9)
        default_seeded = SamplingParams(
            temperature=0.8, top_p=0.9, sampling_seed=DEFAULT_SAMPLING_SEED
        )
        other_seeded = SamplingParams(temperature=0.8, top_p=0.9, sampling_seed=0)
        self.assertEqual(
            sampling_digest(unseeded, deterministic=True),
            sampling_digest(default_seeded, deterministic=True),
        )
        self.assertNotEqual(
            sampling_digest(unseeded, deterministic=True),
            sampling_digest(other_seeded, deterministic=True),
        )
        self.assertNotEqual(
            sampling_digest(unseeded, deterministic=True),
            sampling_digest(unseeded, deterministic=False),
        )

    def test_sampling_digest_matches_values_and_covers_every_field(self) -> None:
        params = SamplingParams(
            temperature=0.6, top_p=0.95, top_k=40, min_p=0.02, sampling_seed=99
        )
        values = dict(
            temperature=0.6,
            top_p=0.95,
            top_k=40,
            min_p=0.02,
            sampling_seed=99,
            has_seed=True,
        )
        base = sampling_digest_from_values(**values)
        self.assertEqual(sampling_digest(params, deterministic=True), base)
        for changed in (
            dict(temperature=0.61),
            dict(top_p=0.94),
            dict(top_k=41),
            dict(top_k=-1),
            dict(min_p=0.03),
            dict(sampling_seed=98),
            dict(has_seed=False),
        ):
            with self.subTest(changed=changed):
                self.assertNotEqual(
                    sampling_digest_from_values(**(values | changed)), base
                )


class TestCanaryInfoChecker(CustomTestCase):
    def setUp(self) -> None:
        self.prompt = [7, 8, 9, 10]
        self.outputs = [100, 101, 102, 103, 104]
        self.params = SamplingParams(temperature=0.9, top_k=30)
        self.canary_infos = reference_canary_info_stream(
            prompt=self.prompt,
            outputs=self.outputs,
            params=self.params,
            deterministic=False,
        )

    def test_checker_rejects_empty_prompt(self) -> None:
        with self.assertRaises(ValueError):
            _make_checker([], self.params)

    def test_checker_accepts_sequence_from_reference(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        for token, info in zip(self.outputs, self.canary_infos):
            self.assertIsNone(checker.advance(token, info))
        self.assertEqual(checker.position, len(self.prompt) + len(self.outputs))

    def test_checker_accepts_whole_chunk(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance_many(self.outputs, self.canary_infos))
        self.assertEqual(checker.position, len(self.prompt) + len(self.outputs))

    def test_checker_accepts_signed_canary_info(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        signed = [to_signed_int64(info) for info in self.canary_infos]
        self.assertIsNone(checker.advance_many(self.outputs, signed))

    def test_checker_rejects_wrong_token(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance(self.outputs[0], self.canary_infos[0]))
        mismatch = checker.advance(self.outputs[1] + 1, self.canary_infos[1])
        self.assertIsNotNone(mismatch)
        self.assertEqual(mismatch.position, len(self.prompt) + 1)
        self.assertEqual(mismatch.token_id, self.outputs[1] + 1)
        self.assertEqual(mismatch.rid, "r0")

    def test_checker_rejects_reordered_or_dropped_token(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance(self.outputs[0], self.canary_infos[0]))
        mismatch = checker.advance(self.outputs[2], self.canary_infos[2])
        self.assertEqual(mismatch.position, len(self.prompt) + 1)
        self.assertEqual(mismatch.actual, self.canary_infos[2])

    def test_checker_rejects_replayed_token(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        self.assertIsNone(checker.advance_many(self.outputs[:2], self.canary_infos[:2]))
        mismatch = checker.advance(self.outputs[1], self.canary_infos[1])
        self.assertEqual(mismatch.position, len(self.prompt) + 2)

    def test_checker_rejects_missing_value(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        mismatch = checker.advance(self.outputs[0], None)
        self.assertIsNone(mismatch.actual)
        self.assertEqual(mismatch.position, len(self.prompt))

    def test_checker_rejects_other_sampling_params(self) -> None:
        checker = _make_checker(self.prompt, SamplingParams(temperature=0.5))
        self.assertIsNotNone(checker.advance(self.outputs[0], self.canary_infos[0]))

    def test_checker_deterministic_flag_must_match_stream(self) -> None:
        deterministic_infos = reference_canary_info_stream(
            prompt=self.prompt,
            outputs=self.outputs,
            params=self.params,
            deterministic=True,
        )
        checker = _make_checker(self.prompt, self.params, deterministic=True)
        self.assertIsNone(checker.advance_many(self.outputs, deterministic_infos))
        checker = _make_checker(self.prompt, self.params, deterministic=True)
        self.assertIsNotNone(checker.advance(self.outputs[0], self.canary_infos[0]))

    def test_checker_keeps_accepted_prefix_after_mismatch(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        bad = list(self.canary_infos)
        bad[2] ^= 1
        mismatch = checker.advance_many(self.outputs, bad)
        self.assertEqual(mismatch.position, len(self.prompt) + 2)
        self.assertEqual(checker.position, len(self.prompt) + 2)
        self.assertIsNone(checker.advance_many(self.outputs[2:], self.canary_infos[2:]))

    def test_mismatch_log_round_trips(self) -> None:
        checker = _make_checker(self.prompt, self.params)
        mismatch = checker.advance(self.outputs[0], 0)
        found = CanaryInfoMismatch.find_all(f"prefix {mismatch.format()} suffix")
        self.assertEqual(found, [mismatch])


def _make_state(rid: str, prompt: list[int], params: SamplingParams) -> ReqState:
    state = ReqState([], False, MagicMock(), MagicMock(), MagicMock())
    state.canary_info = CanaryInfoReqState(
        checker=_build_canary_info_checker(
            SimpleNamespace(rid=rid, input_ids=prompt, sampling_params=params),
            deterministic=False,
        ),
    )
    return state


def _make_hook(
    *, deterministic: bool, abort_on_mismatch: bool = True
) -> CanaryInfoTokenizerManagerHook:
    exec_bag = SimpleNamespace(
        deterministic=SimpleNamespace(enable_deterministic_inference=deterministic)
    )
    with patch(
        "sglang.srt.kv_canary.canary_info.tokenizer_manager_hook.get_exec",
        return_value=exec_bag,
    ):
        return CanaryInfoTokenizerManagerHook(abort_on_mismatch=abort_on_mismatch)


def _make_manager() -> SimpleNamespace:
    return SimpleNamespace(abort_request=MagicMock())


def _make_recv(*, output_ids: list[list[int]]) -> SimpleNamespace:
    return SimpleNamespace(
        output_ids=[array("q", ids) for ids in output_ids],
        finished_reasons=[None] * len(output_ids),
    )


class TestCanaryInfoHookCheck(CustomTestCase):
    def setUp(self) -> None:
        self.params = SamplingParams(temperature=0.7, top_k=40)
        self.prompt_a = [11, 12, 13]
        self.prompt_b = [21, 22, 23, 24]
        self.outputs_a = [101, 102]
        self.outputs_b = [201, 202]
        self.infos_a = reference_canary_info_stream(
            prompt=self.prompt_a,
            outputs=self.outputs_a,
            params=self.params,
            deterministic=False,
        )
        self.infos_b = reference_canary_info_stream(
            prompt=self.prompt_b,
            outputs=self.outputs_b,
            params=self.params,
            deterministic=False,
        )

    def _run(
        self, manager, states, recv, customized_info, *, abort_on_mismatch=True
    ) -> None:
        for index, (rid, state) in enumerate(states.items()):
            _make_hook(
                deterministic=False, abort_on_mismatch=abort_on_mismatch
            ).on_batch_output(
                state=state,
                rid=rid,
                recv_obj=recv,
                customized_info=customized_info,
                index=index,
                abort_request=manager.abort_request,
            )

    def _two_states(self) -> dict[str, ReqState]:
        return {
            "a": _make_state("a", self.prompt_a, self.params),
            "b": _make_state("b", self.prompt_b, self.params),
        }

    def test_matching_chunk_is_delivered_unchanged(self) -> None:
        manager = _make_manager()
        recv = _make_recv(output_ids=[self.outputs_a, self.outputs_b])
        customized_info = {CANARY_INFOS_KEY: [self.infos_a, self.infos_b]}
        self._run(manager, self._two_states(), recv, customized_info)
        manager.abort_request.assert_not_called()
        self.assertEqual(list(recv.output_ids[0]), self.outputs_a)
        self.assertEqual(recv.finished_reasons, [None, None])

    def test_mismatch_aborts_only_that_rid_and_drops_its_tokens(self) -> None:
        manager = _make_manager()
        states = self._two_states()
        recv = _make_recv(
            output_ids=[[self.outputs_a[0], self.outputs_a[1] + 1], self.outputs_b]
        )
        customized_info = {CANARY_INFOS_KEY: [self.infos_a, self.infos_b]}
        self._run(manager, states, recv, customized_info)
        manager.abort_request.assert_called_once_with("a")
        self.assertEqual(len(recv.output_ids[0]), 0)
        self.assertEqual(customized_info[CANARY_INFOS_KEY][0], [])
        finish_reason = recv.finished_reasons[0]
        self.assertEqual(finish_reason["type"], "abort")
        self.assertEqual(finish_reason["message"], CANARY_INFO_MISMATCH)
        self.assertEqual(finish_reason["err_type"], "KvCanaryInfoMismatch")
        self.assertEqual(finish_reason["position"], len(self.prompt_a) + 1)
        self.assertIsNotNone(states["a"].canary_info.mismatch)
        self.assertEqual(list(recv.output_ids[1]), self.outputs_b)
        self.assertIsNone(recv.finished_reasons[1])

    def test_missing_canary_infos_abort(self) -> None:
        manager = _make_manager()
        states = {"a": _make_state("a", self.prompt_a, self.params)}
        recv = _make_recv(output_ids=[self.outputs_a])
        self._run(manager, states, recv, None)
        manager.abort_request.assert_called_once_with("a")
        self.assertEqual(len(recv.output_ids[0]), 0)

    def test_none_padded_canary_infos_abort(self) -> None:
        manager = _make_manager()
        states = {"a": _make_state("a", self.prompt_a, self.params)}
        recv = _make_recv(output_ids=[self.outputs_a])
        self._run(manager, states, recv, {CANARY_INFOS_KEY: [[None, None]]})
        manager.abort_request.assert_called_once_with("a")
        self.assertEqual(len(recv.output_ids[0]), 0)
        self.assertEqual(recv.finished_reasons[0]["message"], CANARY_INFO_MISMATCH)
        self.assertIsNotNone(states["a"].canary_info.mismatch)

    def test_length_mismatch_aborts(self) -> None:
        manager = _make_manager()
        states = {"a": _make_state("a", self.prompt_a, self.params)}
        recv = _make_recv(output_ids=[self.outputs_a])
        self._run(manager, states, recv, {CANARY_INFOS_KEY: [self.infos_a[:1]]})
        manager.abort_request.assert_called_once_with("a")
        self.assertEqual(len(recv.output_ids[0]), 0)
        self.assertEqual(recv.finished_reasons[0]["message"], CANARY_INFO_MISMATCH)
        self.assertIsNotNone(states["a"].canary_info.mismatch)

    def test_checker_spans_multiple_chunks(self) -> None:
        manager = _make_manager()
        states = {"a": _make_state("a", self.prompt_a, self.params)}
        for token, info in zip(self.outputs_a, self.infos_a):
            recv = _make_recv(output_ids=[[token]])
            self._run(manager, states, recv, {CANARY_INFOS_KEY: [[info]]})
            self.assertEqual(list(recv.output_ids[0]), [token])
            self.assertIsNone(recv.finished_reasons[0])
        manager.abort_request.assert_not_called()
        self.assertEqual(
            states["a"].canary_info.checker.position,
            len(self.prompt_a) + len(self.outputs_a),
        )

    def test_log_mode_records_mismatch_but_delivers_unchanged(self) -> None:
        manager = _make_manager()
        states = self._two_states()
        bad_outputs = [self.outputs_a[0], self.outputs_a[1] + 1]
        recv = _make_recv(output_ids=[bad_outputs, self.outputs_b])
        customized_info = {CANARY_INFOS_KEY: [self.infos_a, self.infos_b]}
        self._run(manager, states, recv, customized_info, abort_on_mismatch=False)
        manager.abort_request.assert_not_called()
        self.assertEqual(list(recv.output_ids[0]), bad_outputs)
        self.assertEqual(recv.finished_reasons, [None, None])
        mismatch = states["a"].canary_info.mismatch
        self.assertEqual(mismatch.position, len(self.prompt_a) + 1)
        self.assertIsNone(states["b"].canary_info.mismatch)

        later = _make_recv(output_ids=[[7]])
        self._run(manager, {"a": states["a"]}, later, None, abort_on_mismatch=False)
        self.assertEqual(list(later.output_ids[0]), [7])
        self.assertIs(states["a"].canary_info.mismatch, mismatch)

    def test_request_without_checker_is_rejected(self) -> None:
        manager = _make_manager()
        state = ReqState([], False, MagicMock(), MagicMock(), MagicMock())
        recv = _make_recv(output_ids=[self.outputs_a])
        with self.assertRaises(AssertionError):
            self._run(manager, {"a": state}, recv, None)

    def test_chunks_after_mismatch_are_withheld(self) -> None:
        manager = _make_manager()
        states = {"a": _make_state("a", self.prompt_a, self.params)}
        self._run(manager, states, _make_recv(output_ids=[self.outputs_a]), None)
        self.assertIsNotNone(states["a"].canary_info.mismatch)

        checker = states["a"].canary_info.checker
        recv = _make_recv(output_ids=[self.outputs_a[:1]])
        self._run(
            manager,
            states,
            recv,
            {CANARY_INFOS_KEY: [[checker.expected(self.outputs_a[0])]]},
        )
        self.assertEqual(len(recv.output_ids[0]), 0)
        self.assertEqual(recv.finished_reasons[0]["message"], CANARY_INFO_MISMATCH)
        manager.abort_request.assert_called_once_with("a")

    def test_empty_chunk_passes(self) -> None:
        checker = _make_state("a", self.prompt_a, self.params).canary_info.checker
        self.assertIsNone(
            _check_canary_infos(checker=checker, token_ids=[], canary_infos=None)
        )


class TestCanaryInfoHookDeterministic(CustomTestCase):
    def _dispatch(self, *, deterministic: bool, params: SamplingParams):
        state = ReqState([], False, MagicMock(), MagicMock(), MagicMock())
        tokenized = MagicMock(
            spec=TokenizedGenerateReqInput,
            rid="a",
            input_ids=[11, 12, 13],
            sampling_params=params,
        )

        _make_hook(deterministic=deterministic).on_request_dispatched(
            rid_to_state={"a": state}, tokenized_obj=tokenized
        )
        return state.canary_info.checker

    def test_non_deterministic_hook_ignores_seed(self) -> None:
        params = SamplingParams(temperature=0.7, sampling_seed=1234)
        checker = self._dispatch(deterministic=False, params=params)
        expected = reference_canary_info_stream(
            prompt=[11, 12, 13],
            outputs=[101],
            params=SamplingParams(temperature=0.7),
            deterministic=False,
        )
        self.assertIsNone(checker.advance(101, expected[0]))

    def test_deterministic_hook_uses_default_seed(self) -> None:
        params = SamplingParams(temperature=0.7)
        checker = self._dispatch(deterministic=True, params=params)
        expected = reference_canary_info_stream(
            prompt=[11, 12, 13],
            outputs=[101],
            params=SamplingParams(temperature=0.7, sampling_seed=42),
            deterministic=True,
        )
        self.assertIsNone(checker.advance(101, expected[0]))


class TestCanaryInfoHostOutput(CustomTestCase):
    def _req(self) -> SimpleNamespace:
        return SimpleNamespace(customized_info=None)

    def test_consume_records_values_for_committed_tokens(self) -> None:
        reqs = [self._req(), self._req()]
        output = CanaryInfoHostOutput(values=torch.tensor([7, 9]))
        output.consume(
            SimpleNamespace(reqs=reqs),
            [CommittedTokens(output_index=0, token_ids=(1,)), None],
        )
        self.assertEqual(reqs[0].customized_info[CANARY_INFOS_KEY], [7])
        self.assertIsNone(reqs[1].customized_info)

    def test_consume_truncates_speculative_rows_to_committed_tokens(self) -> None:
        req = self._req()
        req.customized_info = {CANARY_INFOS_KEY: [5]}
        output = CanaryInfoHostOutput(values=torch.tensor([[11, 12, 13]]))
        output.consume(
            SimpleNamespace(reqs=[req]),
            [CommittedTokens(output_index=1, token_ids=(4, 5))],
        )
        self.assertEqual(req.customized_info[CANARY_INFOS_KEY], [5, 11, 12])

    def test_consume_rejects_gap_before_committed_tokens(self) -> None:
        req = self._req()
        output = CanaryInfoHostOutput(values=torch.tensor([3]))
        with self.assertRaises(AssertionError):
            output.consume(
                SimpleNamespace(reqs=[req]),
                [CommittedTokens(output_index=2, token_ids=(9,))],
            )


if __name__ == "__main__":
    unittest.main()
