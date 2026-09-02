"""Zrzut zfuzowanych wezlow FusedConvSiluPair z grafu wykonawczego + wyliczenie
parametrow strojenia kernela (te same wzory co FusedConvSiluPairKernel_bfyx_f16)."""
import os
import sys

import openvino as ov

XML = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\mmiotk\yolo26m_dynamic__openvino_model_end2end_true\_openvino_model\yolo26m.xml"
SHAPE = sys.argv[2] if len(sys.argv) > 2 else "1,3,480,480"

os.environ["OV_GPU_FUSE_CONV_SILU_PAIR"] = "1"

core = ov.Core()
model = core.read_model(XML)
model.reshape({model.inputs[0]: ov.PartialShape([int(v) for v in SHAPE.split(",")])})
cm = core.compile_model(model, "GPU", {"INFERENCE_PRECISION_HINT": "f32"})
rt = cm.get_runtime_model()


def ceil_div(a, b):
    return (a + b - 1) // b


SLM_BUDGET = 64 * 1024  # typowy limit SLM na work-group

print(f"{'in':>6} {'mid':>6} {'out':>6} {'HxW in':>12} {'HxW out':>12} "
      f"{'bw':>3} {'NSG':>4} {'SLM B':>7} {'x_blk':>6}")
n = 0
for node in rt.get_ops():
    info = node.get_rt_info()
    if "layerType" not in info or info["layerType"].astype(str) != "fused_conv_silu_pair":
        continue
    n += 1
    src = node.input(0).get_shape()
    dst = node.output(0).get_shape()
    ic, mid_h, mid_w = src[1], src[2], src[3]
    oc, oh, ow = dst[1], dst[2], dst[3]
    # mid_features nie widac w exec-grafie; bierzemy z ksztaltu wag conv2 (wejscie 3)
    mid = node.input(3).get_shape()[0] if node.get_input_size() > 3 else -1

    ic_blocks = ceil_div(ic, 16)
    mid_ic_blocks = ceil_div(mid, 16) if mid > 0 else -1
    oc_blocks = ceil_div(oc, 16)

    bw = 8
    while bw > 2:
        slm = mid_ic_blocks * bw * 16 * 4
        if slm <= SLM_BUDGET and bw <= ow:
            break
        bw //= 2
    slm = mid_ic_blocks * bw * 16 * 4
    nsg = min(256 // 16, max(mid_ic_blocks, oc_blocks), 8)
    print(f"{ic:6d} {mid:6d} {oc:6d} {f'{mid_h}x{mid_w}':>12} {f'{oh}x{ow}':>12} "
          f"{bw:3d} {nsg:4d} {slm:7d} {ceil_div(ow, bw):6d}")

print(f"\nlacznie zfuzowanych wezlow: {n}")
