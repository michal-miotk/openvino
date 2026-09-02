# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Przemiatanie parametrow strojenia (NUM_SUB_GROUPS, OUTPUT_X_BLOCK_SIZE)
# fused_conv_silu_pair na ksztaltach z yolo26m. Kazdy pomiar to kilkadziesiat
# ms pracy GPU, zeby uniknac throttlingu.

import os
import sys

import numpy as np
import openvino as ov
from openvino import opset14 as op

ITERS = 20
WARMUP = 5

CASES = [
    ("64->128->128  240x240 s2", 64, 128, 128, 240, 240, 3, 2, 1),
    ("256->256->256 120x120 s2", 256, 256, 256, 120, 120, 3, 2, 1),
    ("512->512->512  60x60  s2", 512, 512, 512, 60, 60, 3, 2, 1),
    ("512->512->512  30x30  s2", 512, 512, 512, 30, 30, 3, 2, 1),
]


def build(ic, mid, oc, h, w, k, s, p):
    rng = np.random.default_rng(0)
    x = op.parameter([1, ic, h, w], ov.Type.f32, name="x")

    w1 = op.constant((rng.standard_normal((mid, ic, k, k)) * 0.05).astype(np.float32))
    c1 = op.convolution(x, w1, [s, s], [p, p], [p, p], [1, 1])
    b1 = op.constant((rng.standard_normal((1, mid, 1, 1)) * 0.05).astype(np.float32))
    a1 = op.add(c1, b1)
    s1 = op.swish(a1)

    w2 = op.constant((rng.standard_normal((oc, mid, 1, 1)) * 0.05).astype(np.float32))
    c2 = op.convolution(s1, w2, [1, 1], [0, 0], [0, 0], [1, 1])
    b2 = op.constant((rng.standard_normal((1, oc, 1, 1)) * 0.05).astype(np.float32))
    a2 = op.add(c2, b2)
    s2 = op.swish(a2)

    return ov.Model([op.result(s2)], [x])


def measure(core, model, env):
    for key, val in env.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(val)

    cm = core.compile_model(model, "GPU",
                            {"INFERENCE_PRECISION_HINT": "f32", "PERF_COUNT": "YES"})
    req = cm.create_infer_request()
    src = np.random.default_rng(1).standard_normal(model.inputs[0].shape).astype(np.float32)
    for _ in range(WARMUP):
        req.infer([src])

    per_node = {}
    for _ in range(ITERS):
        req.infer([src])
        for pi in req.profiling_info:
            ms = pi.real_time.total_seconds() * 1000.0
            if ms <= 0 or pi.node_type in ("Reorder", "Result"):
                continue
            per_node[pi.node_name] = min(per_node.get(pi.node_name, 1e9), ms)
    return sum(per_node.values())


def main():
    core = ov.Core()
    nsgs = [2, 4, 8, 16]
    bws = [2, 4, 8]

    for name, ic, mid, oc, h, w, k, s, p in CASES:
        model = build(ic, mid, oc, h, w, k, s, p)
        base = measure(core, model, {"OV_GPU_FUSE_CONV_SILU_PAIR": "0",
                                     "OV_GPU_FCSP_NSG": None, "OV_GPU_FCSP_BW": None})
        print(f"\n=== {name}   (2 konwolucje: {base:.3f} ms)")
        print("      " + "".join(f"{'bw=' + str(b):>12}" for b in bws))
        for nsg in nsgs:
            row = f"nsg={nsg:<2}"
            for bw in bws:
                try:
                    t = measure(core, model, {"OV_GPU_FUSE_CONV_SILU_PAIR": "1",
                                              "OV_GPU_FCSP_NSG": nsg, "OV_GPU_FCSP_BW": bw})
                    row += f"{t:9.3f}({100.0 * (t - base) / base:+.0f}%)"[:12].rjust(12)
                except Exception:
                    row += f"{'-':>12}"
            print(row)
            sys.stdout.flush()


if __name__ == "__main__":
    main()
