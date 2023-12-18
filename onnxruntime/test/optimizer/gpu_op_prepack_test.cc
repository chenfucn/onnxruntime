// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"
#include "core/mlas/inc/mlas.h"
#include "core/graph/graph.h"
#include "core/optimizer/initializer.h"

#include "core/mlas/inc/mlas_q4.h"

namespace onnxruntime {
namespace test {

template <typename T>
const gsl::span<T const> NodeArgToDataSpan(const NodeArg* arg, const Graph& graph) {
  const ONNX_NAMESPACE::TensorProto* tensor_proto;
  ORT_ENFORCE(graph.GetInitializedTensor(arg->Name(), tensor_proto),
              "Missing initializer for ", arg->Name());
  Initializer initializer(*tensor_proto, graph.ModelPath());
  return initializer.DataAsSpan<T>();
}

#ifndef DISABLE_CONTRIB_OPS

TEST(GpuOpPrepackTests, MatmulNBits) {
  auto test_case = [&](int M, int N, int K, int block_size) {
    auto build_test_case = [&](ModelTestBuilder& builder) {

      int q_rows, q_cols;
      MlasBlockwiseQuantizedShape<MLFloat16, 4>((int)block_size, true, (int)K, (int)N, q_rows, q_cols);

      size_t q_data_size_in_bytes, q_scale_size, q_zp_size_in_bytes;
      MlasBlockwiseQuantizedBufferSizes(4, static_cast<int>(block_size), /* columnwise */ true,
                                        static_cast<int>(K), static_cast<int>(N),
                                        q_data_size_in_bytes, q_scale_size, &q_zp_size_in_bytes);

      auto* input_arg = builder.MakeInput<MLFloat16>({M, K}, MLFloat16(0.0f), MLFloat16(31.0f));
      auto* output_arg = builder.MakeOutput();
      auto* weight_arg = builder.MakeInitializer<uint8_t>({q_cols, q_rows}, 0, 255);
      auto* scale_arg = builder.MakeInitializer<MLFloat16>({static_cast<int>(q_scale_size)}, MLFloat16(0.0f), MLFloat16(1.5f));
      auto* zero_point_arg = builder.MakeInitializer<uint8_t>({static_cast<int>(q_zp_size_in_bytes)}, 0, 255);

      std::vector<NodeArg*> input_args{input_arg, weight_arg, scale_arg, zero_point_arg};
      Node& node = builder.AddNode("MatMulNBits", input_args, {output_arg}, kMSDomain);
      node.AddAttribute("K", static_cast<int64_t>(K));
      node.AddAttribute("N", static_cast<int64_t>(N));
      node.AddAttribute("block_size", static_cast<int64_t>(block_size));
      node.AddAttribute("bits", static_cast<int64_t>(4));
    };

    auto check_graph = [&](InferenceSessionWrapper& session) {
      const auto& graph = session.GetGraph();
      for (const auto& node : graph.Nodes()) {
        if (node.OpType() == "MatMulNBits") {
          EXPECT_EQ(node.Domain(), kMSDomain);
          const gsl::span<uint8_t const> weights_data = NodeArgToDataSpan<uint8_t>(node.InputDefs()[1], graph);
          EXPECT_EQ(weights_data[0], 0);
        }
      }
    };

    TransformerTester(build_test_case,
                      check_graph,
                      TransformerLevel::Level2,
                      TransformerLevel::Level3);
  };

  test_case(1, 12, 37, 32);
}

#endif

}  // namespace test
}  // namespace onnxruntime
