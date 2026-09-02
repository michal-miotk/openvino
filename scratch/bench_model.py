# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Pomiar calego modelu odporny na throttling: krotkie przebiegi, minimum
# z iteracji, czas GPU z licznikow (per wezel) obok czasu sciany.

import os
import sys
import time

import numpy as np
import openvino as ov

ITERS = 20
WARMUP = 5


def measure(core, model, fuse):
    os.environ["OV_GPU_FUSE_CONV_SILU_PAIR"] = "1" if fuse else "0"
    cm = core.compile_model(model, "GPU",
                            {"INFERENCE_PRECISION_HINT": "f32", "PERF_COUNT": "YES"})
    req = cm.create_infer_request()
    feed = [np.random.default_rng(1).standard_normal(i.shape).astype(np.float32)
            for i in cm.inputs]
    for _ in range(WARMUP):
        req.infer(feed)

    wall = []
    per_node = {}
    for _ in range(ITERS):
        t0 = time.perf_counter()
        req.infer(feed)
        wall.append((time.perf_counter() - t0) * 1000.0)
        for pi in req.profiling_info:
            ms = pi.real_time.total_seconds() * 1000.0
            if ms > 0:
                per_node[pi.node_name] = min(per_node.get(pi.node_name, 1e9), ms)

    by_type = {}
    for pi in req.profiling_info:
        if pi.node_name in per_node:
            by_type[pi.node_type] = by_type.get(pi.node_type, 0.0) + per_node[pi.node_name]

    types = {pi.node_name: pi.node_type for pi in req.profiling_info}
    return min(wall), sum(per_node.values()), by_type, per_node, types


def main():
    path = sys.argv[1]
    shape = sys.argv[2] if len(sys.argv) > 2 else None

    core = ov.Core()
    model = core.read_model(path)
    if shape:
        dims = [int(v) for v in shape.strip("[]").split(",")]
        model.reshape({model.inputs[0]: ov.PartialShape(dims)})

    res = {}
    for rnd in range(2):
        for fuse in (False, True, True, False):
            out = measure(core, model, fuse)
            if fuse not in res or out[1] < res[fuse][1]:
                res[fuse] = out

    for fuse in (False, True):
        w, g, bt = res[fuse][:3]
        top = sorted(bt.items(), key=lambda kv: -kv[1])[:6]
        print(f"FUSE={int(fuse)}  GPU={g:8.3f} ms   wall={w:7.3f} ms")
        print("    " + ", ".join(f"{k}={v:.3f}" for k, v in top))

    b, f = res[False][1], res[True][1]
    print(f"\ndelta GPU: {100.0 * (f - b) / b:+.1f}%")

    # Wezly obecne tylko przy FUSE=0 (czyli te wchloniete przez fuzje)
    # zestawione z czasem samych wezlow fused.
    base_nodes, base_types = res[False][3], res[False][4]
    fuse_nodes, fuse_types = res[True][3], res[True][4]
    removed = {n: t for n, t in base_nodes.items() if n not in fuse_nodes}
    added = {n: t for n, t in fuse_nodes.items() if fuse_types.get(n) == "FusedConvSiluPair"}
    print(f"\nwchloniete wezly ({len(removed)}, razem {sum(removed.values()):.3f} ms):")
    for n, t in sorted(removed.items(), key=lambda kv: -kv[1]):
        print(f"    {t:7.3f}  {base_types.get(n)}  {n}")
    print(f"\nwezly fused ({len(added)}, razem {sum(added.values()):.3f} ms):")
    for n, t in sorted(added.items(), key=lambda kv: -kv[1]):
        print(f"    {t:7.3f}  {n}")


if __name__ == "__main__":
    main()
