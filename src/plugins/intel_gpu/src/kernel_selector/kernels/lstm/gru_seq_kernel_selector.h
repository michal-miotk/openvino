// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_selector.h"

namespace kernel_selector {
class gru_seq_kernel_selector : public kernel_selector_base {
public:
    static gru_seq_kernel_selector& Instance() {
        static gru_seq_kernel_selector instance_;
        return instance_;
    }

    gru_seq_kernel_selector();

    virtual ~gru_seq_kernel_selector() {}

    KernelsData GetBestKernels(const Params& params) const override;
};
}  // namespace kernel_selector
