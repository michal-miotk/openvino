# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# greater_equal paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def greater_equal(name : str, x, y, data_type, cast_to_fp32=False):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='input_x', shape=x.shape, dtype=data_type)
        node_y = paddle.static.data(name='input_y', shape=y.shape, dtype=data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.greater_equal(x=node_x, y=node_y, name='greater_equal')
        else:
            out = paddle.fluid.layers.greater_equal(x=node_x, y=node_y, name='greater_equal')
        # FuzzyTest framework doesn't support boolean so cast to fp32/int32

        if cast_to_fp32:
            data_type = "float32"

        out = paddle.cast(out, data_type)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'input_x': x, 'input_y': y},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x, node_y], fetchlist=[out],
                  inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():

    test_cases = [
        "float32",
        "int32",
        "int64"
    ]

    for test in test_cases:
        x = np.array([0, 1, 2, 3]).astype(test)
        y = np.array([1, 0, 2, 4]).astype(test)
        if test == "int64":
            greater_equal("greater_equal_" + test, x, y, test, True)
        else:
            greater_equal("greater_equal_" + test, x, y, test, False)

    x = np.array([5000000000]).astype("int64")
    y = np.array([2000000000]).astype("int64")
    greater_equal("greater_equal_big_int64", x, y, test)


if __name__ == "__main__":
    main()
