"""Mikrobenchmark pojedynczej pary conv->silu->conv1x1->silu.

Krotkie przebiegi (dziesiatki ms) + raportowanie minimum, zeby wynik nie byl
zaklocany przez throttling GPU. Wariant zfuzowany i niezfuzowany kompilowane
sa w tym samym procesie - flaga OV_GPU_FUSE_CONV_SILU_PAIR czytana jest przez
pipeline transformacji przy kazdym compile_model.
"""
import os
import sys
import time

import numpy as np
import openvino as ov
from openvino import opset14 as op

# (nazwa, ic, mid, oc, H, W, k, stride, pad) - ksztalty z yolo26m @ 480x480
CASES = [
    ("64->128->128  240x240 s2", 64, 128, 128, 240, 240, 3, 2, 1),
    ("256->256->256 120x120 s2", 256, 256, 256, 120, 120, 3, 2, 1),
    ("512->512->512  60x60  s2", 512, 512, 512, 60, 60, 3, 2, 1),
    ("512->512->512  30x30  s2", 512, 512, 512, 30, 30, 3, 2, 1),
]

ITERS = 30
WARMUP = 5


def build(ic, mid, oc, h, w, k, stride, pad):
    rng = np.random.default_rng(0)
    x = op.parameter([1, ic, h, w], ov.Type.f32, name="x")
    w1 = op.constant((rng.standard_normal((mid, ic, k, k)) * 0.05).astype(np.float32))
    b1 = op.constant((rng.standard_normal((1, mid, 1, 1)) * 0.05).astype(np.float32))
    w2 = op.constant((rng.standard_normal((oc, mid, 1, 1)) * 0.05).astype(np.float32))
    b2 = op.constant((rng.standard_normal((1, oc, 1, 1)) * 0.05).astype(np.float32))
    beta = op.constant(np.float32(1.0))

    c1 = op.convolution(x, w1, [stride, stride], [pad, pad], [pad, pad], [1, 1])
    s1 = op.swish(op.add(c1, b1), beta)
    c2 = op.convolution(s1, w2, [1, 1], [0, 0], [0, 0], [1, 1])
    s2 = op.swish(op.add(c2, b2), beta)
    return ov.Model([s2], [x], "pair")


def measure(core, model, fuse):
    os.environ["OV_GPU_FUSE_CONV_SILU_PAIR"] = "1" if fuse else "0"
    cm = core.compile_model(model, "GPU", {"INFERENCE_PRECISION_HINT": "f32", "PERF_COUNT": "YES"})

    fused = any(
        "layerType" in n.get_rt_info()
        and n.get_rt_info()["layerType"].astype(str) == "fused_conv_silu_pair"
        for n in cm.get_runtime_model().get_ops()
    )

    req = cm.create_infer_request()
    src = np.random.default_rng(1).standard_normal(model.inputs[0].shape).astype(np.float32)
    for _ in range(WARMUP):
        req.infer([src])

    wall = []
    per_node = {}
    for _ in range(ITERS):
        t0 = time.perf_counter()
        req.infer([src])
        wall.append((time.perf_counter() - t0) * 1000.0)
        for pi in req.profiling_info:
            ms = pi.real_time.total_seconds() * 1000.0
            if ms <= 0:
                continue
            key = (pi.node_name, pi.node_type)
            per_node[key] = min(per_node.get(key, 1e9), ms)

    by_type = {}
    for (_, ntype), ms in per_node.items():
        by_type[ntype] = by_type.get(ntype, 0.0) + ms
    # Reorder/Result to artefakt izolowanego modelu (w pelnym modelu wejscie
    # jest juz w fsv16), wiec nie wchodza do porownania.
    compute = sum(v for k, v in by_type.items() if k not in ("Reorder", "Result"))
    return min(wall), compute, fused, by_type


def main():
    core = ov.Core()
    print(f"{'przypadek':<26} {'GPU niefuz':>11} {'GPU fuz':>9} {'delta':>8}")
    for name, ic, mid, oc, h, w, k, s, p in CASES:
        model = build(ic, mid, oc, h, w, k, s, p)
        _, bg1, _, bd = measure(core, model, False)
        _, fg1, is_fused, fd = measure(core, model, True)
        _, fg2, _, _ = measure(core, model, True)
        _, bg2, _, _ = measure(core, model, False)
        b, f = min(bg1, bg2), min(fg1, fg2)
        assert is_fused, f"{name}: brak fuzji!"
        print(f"{name:<26} {b:11.3f} {f:9.3f} {100.0 * (f - b) / b:+7.1f}%")
        print(f"    niefuz: {', '.join(f'{k}={v:.3f}' for k, v in sorted(bd.items()))}")
        print(f"    fuz   : {', '.join(f'{k}={v:.3f}' for k, v in sorted(fd.items()))}")


if __name__ == "__main__":
    sys.exit(main())
