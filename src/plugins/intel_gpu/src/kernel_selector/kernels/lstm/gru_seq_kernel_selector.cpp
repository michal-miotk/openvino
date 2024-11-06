// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gru_seq_kernel_selector.h"
#include "gru_seq_kernel_ref.h"


namespace kernel_selector {
gru_seq_kernel_selector::gru_seq_kernel_selector() {
    Attach<GRUSeqKernelRef>();
}

KernelsData gru_seq_kernel_selector::GetBestKernels(const Params& params) const {
    return GetNaiveBestKernel(params, KernelType::GRU_SEQ_CELL);
}
}  // namespace kernel_selector
