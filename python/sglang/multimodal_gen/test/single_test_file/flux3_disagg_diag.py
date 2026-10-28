"""Debug-only: localize the FLUX.3 disagg-vs-monolithic edit mismatch on ROCm."""

from __future__ import annotations

import base64
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

from sglang.multimodal_gen.test.single_test_file.test_disagg_server import (
    _LOG_DIR,
    HOST,
    DisaggCluster,
    _generate_image,
    _tail_log,
)
from sglang.multimodal_gen.test.test_utils import (
    find_free_port,
    wait_for_server_health,
)
from sglang.test.test_utils import terminate_and_kill_process_tree

MODEL = "black-forest-labs/flux-3-image-klein"
KW = dict(size="256x256", num_inference_steps=2, seed=42, quality="exact")
DIAG = Path("/tmp/flux3diag")
os.environ["SGLANG_FLUX3_DIAG_DIR"] = str(DIAG)

results: dict[str, dict[str, np.ndarray]] = {}
reference: bytes | None = None


def arr(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(BytesIO(data))).astype(np.int16)


def edit(port: int, ref: bytes) -> bytes:
    response = requests.post(
        f"http://{HOST}:{port}/v1/images/edits",
        files={"image": ("reference.png", ref, "image/png")},
        data=dict(
            model=MODEL,
            prompt="Turn the scene into a watercolor painting",
            seed=42,
            num_inference_steps=2,
            quality="exact",
            response_format="b64_json",
        ),
        timeout=600,
    )
    response.raise_for_status()
    return base64.b64decode(response.json()["data"][0]["b64_json"])


def run_requests(tag: str, port: int, concurrent_t2i: bool) -> None:
    global reference
    if concurrent_t2i:
        with ThreadPoolExecutor(max_workers=2) as ex:
            t2i, t2i_b = list(
                ex.map(lambda _: _generate_image(port, MODEL, **KW), range(2))
            )
    else:
        t2i = _generate_image(port, MODEL, **KW)
        t2i_b = None
    if reference is None:
        buf = BytesIO()
        Image.open(BytesIO(t2i)).crop((0, 0, 247, 239)).save(buf, format="PNG")
        reference = buf.getvalue()
    out = dict(t2i=arr(t2i), edit1=arr(edit(port, reference)))
    out["edit2"] = arr(edit(port, reference))
    if t2i_b is not None:
        out["t2i_b"] = arr(t2i_b)
    results[tag] = out
    print(f"[diag] {tag}: done", flush=True)


def run_disagg() -> None:
    os.environ["SGLANG_FLUX3_DIAG_TAG"] = "disagg"
    layout = {
        "prompt_encoder": [0],
        "image_encoder": [0],
        "denoiser": [1],
        "decoder": [0],
    }
    cluster = DisaggCluster(
        model=MODEL,
        name="flux3diag",
        gpu_layout=layout,
        extra_role_args={r: ["--warmup-mode", "off"] for r in (*layout, "server")},
        warmup_kwargs=KW,
    )
    cluster.__enter__()
    try:
        run_requests("disagg", cluster.api_port, concurrent_t2i=True)
    finally:
        cluster.stop()


def run_mono(tag: str, gpu: int, extra_env: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env["SGLANG_FLUX3_DIAG_TAG"] = tag
    env.update(extra_env or {})
    port = find_free_port(HOST)
    log = _LOG_DIR / f"flux3diag_{tag}.log"
    fh = open(log, "w")
    proc = subprocess.Popen(
        [
            "sglang",
            "serve",
            "--model-path",
            MODEL,
            "--port",
            str(port),
            "--num-gpus",
            "1",
            "--base-gpu-id",
            str(gpu),
            "--warmup-mode",
            "off",
        ],
        stdout=fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        env=env,
    )
    try:
        wait_for_server_health(
            f"http://{HOST}:{port}", path="/v1/models", timeout=900
        )
        run_requests(tag, port, concurrent_t2i=False)
    except Exception as e:
        print(f"[diag] {tag} failed: {e}\n{_tail_log(log, 60)}", flush=True)
    finally:
        terminate_and_kill_process_tree(proc)
        fh.close()
        time.sleep(3)


def cmp(a: np.ndarray, b: np.ndarray) -> str:
    if a.shape != b.shape:
        return f"shape {a.shape} vs {b.shape}"
    d = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return f"mismatch={int((d > 0).sum())}/{d.size} max={int(d.max())}"


def tcmp(a, b) -> str:
    if not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor):
        return "EXACT" if a == b else f"{a!r} vs {b!r}"
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"shape/dtype {tuple(a.shape)}/{a.dtype} vs {tuple(b.shape)}/{b.dtype}"
    if not a.is_floating_point():
        ne = int((a != b).sum())
        return "EXACT" if ne == 0 else f"int-mismatch={ne}/{a.numel()}"
    af, bf = a.float(), b.float()
    ne = int((af != bf).sum())
    if ne == 0:
        return "EXACT"
    return f"mismatch={ne}/{a.numel()} maxabs={float((af - bf).abs().max()):.3e}"


def _parse(stem: str) -> tuple[str, str]:
    parts = stem.split("_")
    if parts[0] == "denoise":
        return f"{parts[0]}_{parts[1]}", parts[2]
    return parts[0], parts[1]


def main() -> None:
    steps = [
        ("disagg", run_disagg),
        ("monoA_gpu1", lambda: run_mono("monoA_gpu1", 1)),
        ("monoB_gpu1", lambda: run_mono("monoB_gpu1", 1)),
        ("monoC_gpu0", lambda: run_mono("monoC_gpu0", 0)),
        (
            "monoD_gpu1_freshmiopen",
            lambda: run_mono(
                "monoD_gpu1_freshmiopen",
                1,
                {"MIOPEN_USER_DB_PATH": "/tmp/miopen-fresh-d"},
            ),
        ),
        (
            "monoE_gpu1_freshmiopen",
            lambda: run_mono(
                "monoE_gpu1_freshmiopen",
                1,
                {"MIOPEN_USER_DB_PATH": "/tmp/miopen-fresh-e"},
            ),
        ),
    ]
    for name, fn in steps:
        print(f"[diag] ===== {name} =====", flush=True)
        try:
            fn()
        except Exception as e:
            print(f"[diag] {name} raised: {e}", flush=True)

    print("\n[diag] ===== final image comparisons =====", flush=True)
    tags = list(results)
    for tag in tags:
        for base in ("disagg", "monoA_gpu1"):
            if base not in results or base == tag:
                continue
            for kind in ("t2i", "edit1", "edit2"):
                print(
                    f"[diag] {base}.{kind} vs {tag}.{kind}: "
                    f"{cmp(results[base][kind], results[tag][kind])}"
                )
        print(
            f"[diag] {tag}.edit1 vs {tag}.edit2: "
            f"{cmp(results[tag]['edit1'], results[tag]['edit2'])}"
        )
        if "t2i_b" in results[tag]:
            print(
                f"[diag] {tag}.t2i vs {tag}.t2i_b: "
                f"{cmp(results[tag]['t2i'], results[tag]['t2i_b'])}"
            )

    print("\n[diag] ===== per-stage tensor comparisons =====", flush=True)
    dumps: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for tag_dir in sorted(DIAG.iterdir()):
        files = sorted(tag_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime_ns)
        for f in files:
            stage, key = _parse(f.stem)
            dumps[(tag_dir.name, stage, key)].append(torch.load(f))
    for (tag, stage, key), got in sorted(dumps.items()):
        print(f"[diag] dumps tag={tag} stage={stage} key={key} count={len(got)}")
    keys = sorted({(stage, key) for (_, stage, key) in dumps})
    for stage, key in keys:
        for base in ("disagg", "monoA_gpu1"):
            ref = dumps.get((base, stage, key))
            if not ref:
                continue
            for tag in tags:
                got = dumps.get((tag, stage, key))
                if not got or tag == base:
                    continue
                # Compare the last occurrence: the first i2i edit is
                # second-to-last, but last-vs-last is enough to localize.
                for name in sorted(ref[-1]):
                    if name in got[-1]:
                        print(
                            f"[diag] {stage} key={key} {name}: {base}[-1] vs "
                            f"{tag}[-1]: {tcmp(ref[-1][name], got[-1][name])}"
                        )
        for tag in tags:
            got = dumps.get((tag, stage, key))
            if got and len(got) > 1:
                for name in sorted(got[0]):
                    if name in got[-1]:
                        print(
                            f"[diag] {stage} key={key} {tag} first vs last "
                            f"{name}: {tcmp(got[0][name], got[-1][name])}"
                        )


if __name__ == "__main__":
    main()
