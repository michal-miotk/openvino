// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prepare_lstm_weights.hpp"
#include <memory>

#include "intel_gpu/op/fully_connected.hpp"
#include "intel_gpu/op/fully_connected_compressed.hpp"

#include "openvino/op/constant.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/lstm_sequence.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_gpu {

PrepareLSTMWeights::PrepareLSTMWeights() {
    using namespace ov::pass::pattern;
    auto m = std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<ov::op::v5::LSTMSequence>(), "PrepareLSTMWeights");
    register_matcher(m, [&](ov::pass::pattern::Matcher& m) {
        auto lstm = std::dynamic_pointer_cast<ov::op::v5::LSTMSequence>(m.get_match_root());
        if (!lstm) {
            return false;
        }
        auto rt_info = lstm->get_rt_info();
        return false;
    });
}

}  // namespace intel_gpu
}  // namespace ov
