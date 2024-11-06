// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "lstm_kernel_base.h"

namespace kernel_selector {
class GRUSeqKernelRef : public LSTMKernelBase {
public:
    GRUSeqKernelRef() : LSTMKernelBase("gru_cell_and_seq_ref") {}
    virtual ~GRUSeqKernelRef() {}

    KernelsData GetKernelsData(const Params& params) const override;
    KernelsPriority GetKernelsPriority(const Params& params) const override;
    ParamsKey GetSupportedKey() const override;
protected:
    bool Validate(const Params& p) const override {
        if (p.GetType() != KernelType::GRU_SEQ_CELL) {
            return false;
        }

        return true;
    }
};
}  // namespace kernel_selector
