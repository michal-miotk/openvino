# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Pojedynczy przypadek testowy uruchamiany w osobnym procesie (izolacja crashy GPU)."""

import sys

import numpy as np
import openvino as ov

sys.path.insert(0, "scratch")
from test_fused_conv_silu_pair import build_case  # noqa: E402


def main():
    a = [int(v) for v in sys.argv[1:14]]
    n, ic, mid, oc, h, w, ky, kx, py, px, sy, sx = a[:12]
    model, src, ref = build_case(1234, n, ic, mid, oc, h, w, ky, kx, py, px, sy, sx,
                                 1.0, 1.0, np.float32)
    core = ov.Core()
    cm = core.compile_model(model, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
    fused = any("fused_conv_silu_pair" in node.get_rt_info()["layerType"].astype(str)
                for node in cm.get_runtime_model().get_ordered_ops()
                if "layerType" in node.get_rt_info())
    got = cm(src)[0].astype(np.float64)
    rel = np.max(np.abs(got - ref) / np.maximum(np.abs(ref), 1.0))
    print(f"rel={rel:.3e} fused={fused}")


if __name__ == "__main__":
    main()
