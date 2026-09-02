import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

XML = sys.argv[1] if len(sys.argv) > 1 else r"C:\mmiotk\yolo26m_dynamic__openvino_model_end2end_true\_openvino_model\yolo26m.xml"

tree = ET.parse(XML)
root = tree.getroot()

layers = {}
for lay in root.iter("layer"):
    lid = lay.get("id")
    d = {
        "id": lid,
        "name": lay.get("name"),
        "type": lay.get("type"),
        "data": lay.find("data").attrib if lay.find("data") is not None else {},
        "out_shapes": [],
        "in_shapes": [],
    }
    out = lay.find("output")
    if out is not None:
        for p in out.findall("port"):
            d["out_shapes"].append([dim.text for dim in p.findall("dim")])
    inp = lay.find("input")
    if inp is not None:
        for p in inp.findall("port"):
            d["in_shapes"].append([dim.text for dim in p.findall("dim")])
    layers[lid] = d

consumers = defaultdict(list)   # layer id -> list of consumer ids
producers = defaultdict(list)   # layer id -> list of (producer id)
for e in root.iter("edge"):
    fl, tl = e.get("from-layer"), e.get("to-layer")
    consumers[fl].append(tl)
    producers[tl].append(fl)

ACT_TYPES = {
    "Swish", "Sigmoid", "Relu", "PRelu", "HSwish", "HSigmoid", "Mish", "Elu",
    "Gelu", "Clamp", "Tanh", "SoftPlus", "HardSigmoid", "LeakyRelu", "SoftSign",
}


def is_conv(l):
    return l["type"] in ("Convolution", "GroupConvolution")


def kernel_of(conv):
    """Return spatial kernel dims from the weights input shape."""
    prods = producers[conv["id"]]
    if len(prods) < 2:
        return None
    w = layers[prods[1]]
    if not w["out_shapes"]:
        return None
    shp = w["out_shapes"][0]
    # Convolution weights: [OC, IC, kY, kX]; GroupConvolution: [G, OC/G, IC/G, kY, kX]
    if conv["type"] == "Convolution":
        return tuple(shp[2:])
    return tuple(shp[3:])


def is_1x1(conv):
    k = kernel_of(conv)
    return k is not None and all(x == "1" for x in k)


def single_consumer(lid):
    c = consumers[lid]
    return layers[c[0]] if len(c) == 1 else None


def bias_add_after(conv):
    """conv -> Add where second input is a Constant (bias)."""
    nxt = single_consumer(conv["id"])
    if nxt is None or nxt["type"] != "Add":
        return None
    prods = producers[nxt["id"]]
    const_in = any(layers[p]["type"] == "Const" for p in prods)
    return nxt if const_in else None


def act_after(add):
    nxt = single_consumer(add["id"])
    if nxt is None:
        return None
    if nxt["type"] in ACT_TYPES:
        return nxt
    return None


# --- find all conv -> add -> act blocks ---
blocks = {}  # conv id -> (conv, add, act)
for lid, l in layers.items():
    if not is_conv(l):
        continue
    add = bias_add_after(l)
    if add is None:
        continue
    act = act_after(add)
    if act is None:
        continue
    blocks[lid] = (l, add, act)

all_convs = [l for l in layers.values() if is_conv(l)]

# --- find pairs: block1.act -> block2.conv, block2 conv is 1x1 ---
pairs = []
pairs_any_k = []
for cid, (c1, a1, act1) in blocks.items():
    nxt = single_consumer(act1["id"])
    if nxt is None or nxt["id"] not in blocks:
        continue
    c2, a2, act2 = blocks[nxt["id"]]
    pairs_any_k.append((c1, c2))
    if is_1x1(c2):
        pairs.append((c1, c2, act1["type"], act2["type"]))

n_convs = len(all_convs)
n_blocks = len(blocks)
n_pairs = len(pairs)
n_pairs_any = len(pairs_any_k)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


print(f"Model: {XML}")
print(f"Total layers:                                  {len(layers)}")
print(f"Convolution/GroupConvolution layers:           {n_convs}")
print(f"  of which 1x1:                                {sum(1 for c in all_convs if is_1x1(c))}")
print(f"conv->Add(bias)->activation blocks:            {n_blocks}  ({pct(n_blocks, n_convs):.1f}% of convs)")
print()
print(f"Pattern conv->add->act->conv->add->act (any k): {n_pairs_any}")
print(f"Pattern with 2nd conv 1x1:                     {n_pairs}")
print()
print("Percentages for the '2nd conv is 1x1' pattern:")
print(f"  vs. all convolutions            : {n_pairs} pairs = {2*n_pairs} convs / {n_convs} = {pct(2*n_pairs, n_convs):.1f}%")
print(f"  vs. all conv->add->act blocks   : {n_pairs} pairs = {2*n_pairs} blocks / {n_blocks} = {pct(2*n_pairs, n_blocks):.1f}%")
print(f"  vs. all such chained pairs      : {pct(n_pairs, n_pairs_any):.1f}%")
print(f"  vs. all layers                  : {pct(3*2*n_pairs, len(layers)):.1f}% of layers involved")
print()
print("Matched pairs:")
for c1, c2, t1, t2 in pairs:
    print(f"  {c1['name']} [k={kernel_of(c1)}] -{t1}-> {c2['name']} [k={kernel_of(c2)}] -{t2}->")
