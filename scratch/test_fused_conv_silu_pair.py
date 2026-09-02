# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Walidacja fuzji conv->add->silu->conv1x1->add->silu na GPU.

Uruchamianie:
    $env:PYTHONPATH="...\\bin\\intel64\\Release\\python"
    $env:OPENVINO_LIB_PATHS="...\\bin\\intel64\\Release"
    $env:OV_GPU_FUSE_CONV_SILU_PAIR="1"   # albo brak / "0"
    py -3.14 scratch\\test_fused_conv_silu_pair.py
"""

import os
import sys

import numpy as np
import openvino as ov
from openvino import opset14 as opset


def conv2d(x, w, pad_y, pad_x, stride_y, stride_x):
    """Naiwna konwolucja NCHW przez im2col."""
    n, ic, h, wd = x.shape
    oc, ic_w, ky, kx = w.shape
    assert ic == ic_w
    xp = np.pad(x, ((0, 0), (0, 0), (pad_y, pad_y), (pad_x, pad_x)))
    oh = (h + 2 * pad_y - ky) // stride_y + 1
    ow = (wd + 2 * pad_x - kx) // stride_x + 1
    cols = np.empty((n, ic * ky * kx, oh * ow), dtype=np.float64)
    idx = 0
    for c in range(ic):
        for i in range(ky):
            for j in range(kx):
                patch = xp[:, c, i:i + stride_y * oh:stride_y, j:j + stride_x * ow:stride_x]
                cols[:, idx, :] = patch.reshape(n, -1)
                idx += 1
    wm = w.reshape(oc, -1).astype(np.float64)
    out = np.matmul(wm[None, ...], cols)
    return out.reshape(n, oc, oh, ow)


def silu(x, beta):
    return x / (1.0 + np.exp(-beta * x))


def build_case(seed, n, ic, mid, oc, h, w, ky, kx, pad_y, pad_x, stride_y, stride_x,
               beta1, beta2, dtype):
    rng = np.random.default_rng(seed)
    src = rng.standard_normal((n, ic, h, w)).astype(np.float32) * 0.5
    w1 = rng.standard_normal((mid, ic, ky, kx)).astype(np.float32) * (1.0 / np.sqrt(ic * ky * kx))
    b1 = rng.standard_normal((1, mid, 1, 1)).astype(np.float32) * 0.1
    w2 = rng.standard_normal((oc, mid, 1, 1)).astype(np.float32) * (1.0 / np.sqrt(mid))
    b2 = rng.standard_normal((1, oc, 1, 1)).astype(np.float32) * 0.1

    param = opset.parameter(src.shape, dtype, name="src")
    c1 = opset.convolution(param, opset.constant(w1.astype(dtype)),
                           strides=[stride_y, stride_x], pads_begin=[pad_y, pad_x],
                           pads_end=[pad_y, pad_x], dilations=[1, 1])
    a1 = opset.add(c1, opset.constant(b1.astype(dtype)))
    s1 = opset.swish(a1, opset.constant(np.array(beta1, dtype=dtype)))
    c2 = opset.convolution(s1, opset.constant(w2.astype(dtype)),
                           strides=[1, 1], pads_begin=[0, 0], pads_end=[0, 0], dilations=[1, 1])
    a2 = opset.add(c2, opset.constant(b2.astype(dtype)))
    s2 = opset.swish(a2, opset.constant(np.array(beta2, dtype=dtype)))
    model = ov.Model([opset.result(s2)], [param], "conv_silu_pair")

    # referencja numpy w f64
    ref = conv2d(src.astype(np.float64), w1.astype(np.float64), pad_y, pad_x, stride_y, stride_x)
    ref = silu(ref + b1.astype(np.float64), beta1)
    ref = conv2d(ref, w2.astype(np.float64), 0, 0, 1, 1)
    ref = silu(ref + b2.astype(np.float64), beta2)
    return model, src.astype(dtype), ref


CASES = [
    # nazwa,  n, ic, mid, oc,  h,  w, ky, kx, py, px, sy, sx, b1,  b2
    ("3x3 s1 p1 aligned",   1, 32,  64, 32, 16, 16, 3, 3, 1, 1, 1, 1, 1.0, 1.0),
    ("1x1 s1 p0 aligned",   1, 16,  32, 16,  8,  8, 1, 1, 0, 0, 1, 1, 1.0, 1.0),
    ("3x3 s2 p1 aligned",   1, 16,  32, 16, 16, 16, 3, 3, 1, 1, 2, 2, 1.0, 1.0),
    ("3x3 s1 p1 leftovers", 1, 20,  40, 24, 12, 12, 3, 3, 1, 1, 1, 1, 1.0, 1.0),
    ("3x3 s1 p0 x%8!=0",    1, 32,  64, 32, 13, 13, 3, 3, 0, 0, 1, 1, 1.0, 1.0),
    ("batch2 beta!=1",      2, 32,  48, 32,  8,  8, 3, 3, 1, 1, 1, 1, 1.5, 0.7),
    ("big mid",             1, 16, 128, 64, 10, 10, 3, 3, 1, 1, 1, 1, 1.0, 1.0),
]


def main():
    dtype_name = "f32"
    dtype = np.float32
    tol = 1e-4
    config = {"INFERENCE_PRECISION_HINT": "f32"}

    fuse = os.environ.get("OV_GPU_FUSE_CONV_SILU_PAIR", "0")
    core = ov.Core()
    print(f"=== dtype={dtype_name}  OV_GPU_FUSE_CONV_SILU_PAIR={fuse} ===")

    failures = 0
    for case in CASES:
        name, n, ic, mid, oc, h, w, ky, kx, py, px, sy, sx, b1, b2 = case
        model, src, ref = build_case(1234, n, ic, mid, oc, h, w, ky, kx, py, px, sy, sx,
                                     b1, b2, dtype)
        cm = core.compile_model(model, "GPU", config)

        exec_types = {node.get_rt_info()["layerType"].astype(str)
                      for node in cm.get_runtime_model().get_ordered_ops()
                      if "layerType" in node.get_rt_info()}
        fused = any("fused_conv_silu_pair" in t.lower() for t in exec_types)

        try:
            got = cm(src)[0].astype(np.float64)
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {name:22s} wyjatek: {type(exc).__name__}: {str(exc).splitlines()[-1]}  fused={fused}")
            failures += 1
            continue

        if got.shape != ref.shape:
            print(f"[FAIL] {name:22s} ksztalt {got.shape} != {ref.shape}")
            failures += 1
            continue
        denom = np.maximum(np.abs(ref), 1.0)
        rel = np.max(np.abs(got - ref) / denom)
        ok = rel < tol
        failures += 0 if ok else 1
        print(f"[{'OK ' if ok else 'FAIL'}] {name:22s} rel_err={rel:.3e}  fused={fused}")

    print(f"--- niepowodzenia: {failures}/{len(CASES)} ---")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
