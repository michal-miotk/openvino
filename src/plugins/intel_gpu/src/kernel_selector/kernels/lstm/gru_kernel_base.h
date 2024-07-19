// Copyright (C) 2018-2025 Intel Corporation
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
        offset_zrh   // OV default is same as uro in ONEDNN
    };

    gru_params() : base_params(KernelType::GRU_SEQ_CELL) {}
    order_type gate_order = offset_zrh;
    float clip = 0;
    bool input_forget = false;
    bool sequential = false;
    ov::op::RecurrentSequenceDirection direction = ov::op::RecurrentSequenceDirection::FORWARD;

    size_t GetOffsetIndex(order_type type, size_t idx) const {
        static const std::map<order_type, std::vector<size_t>> offset_map{{offset_zrh, {0, 1, 2}}};
        return offset_map.at(type)[idx];
    }

    size_t GetOffsetIndexZ() const { return GetOffsetIndex(gate_order, 0); }
    size_t GetOffsetIndexR() const { return GetOffsetIndex(gate_order, 1); }
    size_t GetOffsetIndexH() const { return GetOffsetIndex(gate_order, 2); }

    void SetOffsetOrder(int32_t t) { gate_order = static_cast<order_type>(t); }

    ParamsKey GetParamsKey() const override {
        ParamsKey k = base_params::GetParamsKey();
        return k;
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// GRUKernelBase
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
class GRUKernelBase : public KernelBaseOpenCL {
public:
    using KernelBaseOpenCL::KernelBaseOpenCL;
    virtual ~GRUKernelBase() {}

    struct DispatchData : public CommonDispatchData {};

protected:
    virtual JitConstants GetJitConstants(const gru_params& params) const;
    KernelsData GetCommonKernelsData(const Params& params) const;

    bool Validate(const Params& p) const override {
        if (p.GetType() != KernelType::GRU_SEQ_CELL) {
            return false;
        }

        return true;
    }
};
}  // namespace kernel_selector
