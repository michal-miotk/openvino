"""Wyszukanie wzorca conv->add->silu->conv1x1->add->silu w modelu zrodlowym
i wyliczenie parametrow strojenia kernela FusedConvSiluPairKernel_bfyx_f16."""
import sys

import openvino as ov

XML = sys.argv[1] if len(sys.argv) > 1 else \
    r"C:\mmiotk\yolo26m_dynamic__openvino_model_end2end_true\_openvino_model\yolo26m.xml"
SHAPE = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2 else "1,3,480,480").split(",")]

core = ov.Core()
model = core.read_model(XML)
model.reshape({model.inputs[0]: ov.PartialShape(SHAPE)})


def single_consumer(node):
    outs = node.output(0).get_target_inputs()
    return next(iter(outs)).get_node() if len(outs) == 1 else None


def ceil_div(a, b):
    return (a + b - 1) // b


SLM_BUDGET = 64 * 1024


def tuning(ic, mid, oc, ow):
    ic_b, mid_b, oc_b = ceil_div(ic, 16), ceil_div(mid, 16), ceil_div(oc, 16)
    bw = 8
    while bw > 2:
        if mid_b * bw * 16 * 4 <= SLM_BUDGET and bw <= ow:
            break
        bw //= 2
    slm = mid_b * bw * 16 * 4
    nsg = min(256 // 16, max(mid_b, oc_b), 8)
    return bw, nsg, slm, ic_b, mid_b, oc_b


print(f"{'ic':>5} {'mid':>5} {'oc':>5} {'k1':>5} {'s':>2} {'out HxW':>10} "
      f"{'bw':>3} {'NSG':>4} {'SLM kB':>7} {'ICb':>4} {'MIDb':>5} {'OCb':>4} {'faza1/faza2':>12}")

n = 0
for op in model.get_ordered_ops():
    if op.get_type_name() != "Convolution":
        continue
    add1 = single_consumer(op)
    if add1 is None or add1.get_type_name() != "Add":
        continue
    silu1 = single_consumer(add1)
    if silu1 is None or silu1.get_type_name() != "Swish":
        continue
    conv2 = single_consumer(silu1)
    if conv2 is None or conv2.get_type_name() != "Convolution":
        continue
    add2 = single_consumer(conv2)
    if add2 is None or add2.get_type_name() != "Add":
        continue
    silu2 = single_consumer(add2)
    if silu2 is None or silu2.get_type_name() != "Swish":
        continue
    w2 = conv2.input(1).get_shape()
    if len(w2) != 4 or w2[2] != 1 or w2[3] != 1:
        continue

    w1 = op.input(1).get_shape()
    ic, mid, oc = w1[1], w1[0], w2[0]
    k1 = f"{w1[2]}x{w1[3]}"
    stride = op.get_strides()[0]
    out = silu2.output(0).get_shape()
    oh, ow = out[2], out[3]
    bw, nsg, slm, ic_b, mid_b, oc_b = tuning(ic, mid, oc, ow)
    # koszt fazy 1 ~ MIDb * ICb * k1 mad-ow, fazy 2 ~ OCb * MIDb
    f1 = mid_b * ic_b * w1[2] * w1[3]
    f2 = oc_b * mid_b
    n += 1
    print(f"{ic:5d} {mid:5d} {oc:5d} {k1:>5} {stride:2d} {f'{oh}x{ow}':>10} "
          f"{bw:3d} {nsg:4d} {slm/1024:7.1f} {ic_b:4d} {mid_b:5d} {oc_b:4d} {f1/f2:11.1f}x")

print(f"\nznalezionych par: {n}")
