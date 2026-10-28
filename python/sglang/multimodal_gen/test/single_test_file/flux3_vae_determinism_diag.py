"""Debug-only: find the run-to-run nondeterministic op in the FLUX.3 VAE on ROCm."""

from __future__ import annotations

import time

import torch
from safetensors.torch import load_file
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.flux3_image import Flux3ImageVAEConfig
from sglang.multimodal_gen.runtime.loader.utils import load_model_state_dict
from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
from sglang.multimodal_gen.runtime.models.vaes.flux3_image_vae import Flux3ImageVAE
from sglang.multimodal_gen.runtime.pipelines.flux3_image_package import (
    resolve_flux3_image_package,
)
from sglang.multimodal_gen.runtime.utils.precision import precision_to_dtype

MODEL = "black-forest-labs/flux-3-image-klein"
RUNS = 8


def load_vae() -> nn.Module:
    package = resolve_flux3_image_package(MODEL)
    dtype = precision_to_dtype(package.config.vae.dtype)
    print(f"[vaediag] vae dtype={dtype} path={package.vae}", flush=True)
    with torch.device("meta"):
        vae = Flux3ImageVAE(Flux3ImageVAEConfig()).to(dtype=dtype)
    load_model_state_dict(vae, load_file(str(package.vae), device="cuda"), assign=True)
    return vae.requires_grad_(False).eval()


def summarize(outs: list[torch.Tensor]) -> str:
    ref = outs[0].float()
    diffs = [(o.float() - ref).abs() for o in outs[1:]]
    distinct = len({o.float().cpu().numpy().tobytes() for o in outs})
    return (
        f"distinct={distinct}/{len(outs)} "
        f"max_abs={max(float(d.max()) for d in diffs):.3e} "
        f"max_mismatch={max(int((d > 0).sum()) for d in diffs)}/{ref.numel()}"
    )


def run(fn, x) -> tuple[list[torch.Tensor], float]:
    outs = []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(RUNS):
        with torch.inference_mode():
            outs.append(fn(x).clone())
    torch.cuda.synchronize()
    return outs, (time.perf_counter() - start) / RUNS


def localize(vae: nn.Module, fn, x, label: str) -> None:
    records: list[list[tuple[str, str, torch.Tensor | None, torch.Tensor]]] = []
    current: list = []

    def hook(name):
        def _hook(module, args, output):
            inp = args[0].detach().clone() if args and torch.is_tensor(args[0]) else None
            if torch.is_tensor(output):
                current.append((name, type(module).__name__, inp, output.detach().clone()))

        return _hook

    handles = [m.register_forward_hook(hook(n)) for n, m in vae.named_modules() if n]
    try:
        for _ in range(4):
            current = []
            with torch.inference_mode():
                fn(x)
            records.append(current)
    finally:
        for h in handles:
            h.remove()
    base = records[0]
    for k, rec in enumerate(records[1:], start=1):
        hits = []
        for (name, kind, i0, o0), (_, _, i1, o1) in zip(base, rec):
            same_in = i0 is not None and i1 is not None and torch.equal(i0, i1)
            if not torch.equal(o0, o1):
                hits.append((name, kind, same_in, float((o0.float() - o1.float()).abs().max())))
        first_same_in = [h for h in hits if h[2]]
        print(
            f"[vaediag] {label} run0 vs run{k}: {len(hits)} modules differ; "
            f"first differing={hits[:1]}",
            flush=True,
        )
        for h in first_same_in[:12]:
            print(
                f"[vaediag] {label} run0 vs run{k}: SAME INPUT, DIFFERENT OUTPUT "
                f"{h[0]} ({h[1]}) max_abs={h[3]:.3e}",
                flush=True,
            )


def main() -> None:
    torch.manual_seed(0)
    vae = load_vae()
    dtype = next(vae.parameters()).dtype
    image = (torch.rand(1, 3, 240, 256, device="cuda") * 2 - 1).to(dtype)
    print(
        f"[vaediag] torch={torch.__version__} hip={torch.version.hip} "
        f"benchmark={torch.backends.cudnn.benchmark} "
        f"deterministic={torch.backends.cudnn.deterministic}",
        flush=True,
    )

    with use_vae_fast_path(vae, False):
        latent = vae.encode(image)
        encode = vae.encode
        decode = vae.decode

        configs = [
            ("default", {}),
            ("cudnn.deterministic", {"deterministic": True}),
            ("cudnn.disabled", {"enabled": False}),
            ("cudnn.benchmark", {"benchmark": True}),
        ]
        for name, flags in configs:
            saved = {k: getattr(torch.backends.cudnn, k) for k in flags}
            for k, v in flags.items():
                setattr(torch.backends.cudnn, k, v)
            try:
                run(encode, image)  # settle algorithm selection
                outs, t = run(encode, image)
                print(f"[vaediag] encode {name}: {summarize(outs)} t={t*1e3:.1f}ms", flush=True)
                run(decode, latent)
                outs, t = run(decode, latent)
                print(f"[vaediag] decode {name}: {summarize(outs)} t={t*1e3:.1f}ms", flush=True)
            except Exception as e:
                print(f"[vaediag] {name} raised {e!r}", flush=True)
            finally:
                for k, v in saved.items():
                    setattr(torch.backends.cudnn, k, v)

        localize(vae, encode, image, "encode")
        localize(vae, decode, latent, "decode")

        # The attention block alone, per SDPA backend.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attn = vae.encoder.mid.attn_1
        h = torch.randn(1, attn.in_channels, 15, 16, device="cuda", dtype=dtype)
        for backend in (None, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION):
            try:
                ctx = sdpa_kernel(backend) if backend is not None else torch.no_grad()
                with ctx:
                    outs, t = run(attn, h)
                print(f"[vaediag] attn backend={backend}: {summarize(outs)}", flush=True)
            except Exception as e:
                print(f"[vaediag] attn backend={backend} raised {e!r}", flush=True)


if __name__ == "__main__":
    main()
