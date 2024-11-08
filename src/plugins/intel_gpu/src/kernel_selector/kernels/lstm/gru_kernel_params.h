// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"
#include "kernel_selector_params.h"
#include <vector>
#include <map>

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// gru_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct gru_params : public base_params {
    enum order_type : int32_t {
        offset_iofz,  // ONNX default
        offset_ifoz,  // caffe
        offset_izof,  // pyTorch
        offset_fizo   // OV default
    };

    gru_params() : base_params(KernelType::GRU_SEQ_CELL) {}
    order_type gate_order = offset_iofz;
    float clip = 0;
    bool input_forget = false;
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD;

    size_t GetOffsetIndex(order_type type, size_t idx) const {
        static const std::map<order_type, std::vector<size_t>> offset_map{{offset_iofz, {0, 1, 2, 3}},
                                                                          {offset_ifoz, {0, 2, 1, 3}},
                                                                          {offset_izof, {0, 3, 1, 2}},
                                                                          {offset_fizo, {1, 3, 0, 2}}};
        return offset_map.at(type)[idx];
    }

    size_t GetOffsetIndexI() const { return GetOffsetIndex(gate_order, 0); }
    size_t GetOffsetIndexO() const { return GetOffsetIndex(gate_order, 1); }
    size_t GetOffsetIndexF() const { return GetOffsetIndex(gate_order, 2); }
    size_t GetOffsetIndexZ() const { return GetOffsetIndex(gate_order, 3); }

    void SetOffsetOrder(int32_t t) { gate_order = static_cast<order_type>(t); }

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        return k;
    }
};
}  // namespace kernel_selector
