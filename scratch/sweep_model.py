# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Przemiatanie strojenia fused_conv_silu_pair na prawdziwym modelu.
# Metryka: suma minimalnych czasow GPU wezlow FusedConvSiluPair,
# porownana z suma wezlow, ktore fuzja wchlonela.

import os
import sys

import numpy as np
import openvino as ov

ITERS = 20
WARMUP = 5


def measure(core, model, env):
    for key, val in env.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(val)

    cm = core.compile_model(model, "GPU",
                            {"INFERENCE_PRECISION_HINT": "f32", "PERF_COUNT": "YES"})
    req = cm.create_infer_request()
    feed = [np.random.default_rng(1).standard_normal(i.shape).astype(np.float32)
            for i in cm.inputs]
    for _ in range(WARMUP):
        req.infer(feed)

    per_node, types = {}, {}
    for _ in range(ITERS):
        req.infer(feed)
        for pi in req.profiling_info:
            ms = pi.real_time.total_seconds() * 1000.0
            if ms > 0:
                per_node[pi.node_name] = min(per_node.get(pi.node_name, 1e9), ms)
                types[pi.node_name] = pi.node_type
    return per_node, types


def main():
    path = sys.argv[1]
    core = ov.Core()
    model = core.read_model(path)
    model.reshape({model.inputs[0]: ov.PartialShape([1, 3, 480, 480])})

    base, _ = measure(core, model, {"OV_GPU_FUSE_CONV_SILU_PAIR": "0",
                                    "OV_GPU_FCSP_DIV": None, "OV_GPU_FCSP_BW": None})

    best = None
    print(f"{'div':>4} {'bw':>3} {'fused [ms]':>11} {'wchloniete':>11} {'delta':>8}")
    for div in (1, 2, 4, 8):
        for bw in (2, 4, 8):
            try:
                nodes, types = measure(core, model, {"OV_GPU_FUSE_CONV_SILU_PAIR": "1",
                                                     "OV_GPU_FCSP_DIV": div,
                                                     "OV_GPU_FCSP_BW": bw})
            except Exception as exc:
                print(f"{div:>4} {bw:>3}  blad: {type(exc).__name__}")
                continue
            fused = sum(t for n, t in nodes.items() if types[n] == "FusedConvSiluPair")
            removed = sum(t for n, t in base.items() if n not in nodes)
            if fused <= 0:
                print(f"{div:>4} {bw:>3}  brak fuzji")
                continue
            d = 100.0 * (fused - removed) / removed
            print(f"{div:>4} {bw:>3} {fused:11.3f} {removed:11.3f} {d:+7.1f}%")
            sys.stdout.flush()
            if best is None or fused < best[0]:
                best = (fused, div, bw)

    print(f"\nnajlepsze: div={best[1]} bw={best[2]} -> {best[0]:.3f} ms")


if __name__ == "__main__":
    main()
