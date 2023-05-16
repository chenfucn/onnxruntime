// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/span_utils.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/session/inference_session.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_builder.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
#include "core/util/qmath.h"

#include <chrono>
#include <random>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

namespace onnxruntime {
namespace test {

TEST(MatMulFpQ4, MatMul2D) {
  OpTester test("MatMulFpQ4", 1, kMSDomain);

  // (100 x 41) X (41 x 288)
  const int64_t M = 100;
  const int64_t N = 288;
  const int64_t K = 41;

  std::vector<float> input0_vals(M * K);
  float fv = -135.f;
  for (auto& f : input0_vals) {
    f = fv / 128;
    fv++;
    if (fv > 135.f) {
      fv = -135.f;
    }
  }

  std::vector<float> input1_f_vals(N * K);
  uint8_t v = 0;
  for (size_t i = 0; i < N * K; i++) {
    input1_f_vals[i] = (float)v;
    if (++v >= 16) {
      v = 0;
    }
  }
  std::vector<uint8_t> input1_vals(MlasQ4GemmPackBSize((size_t)N, (size_t)K));
  MlasQ4GemmPackB(input1_vals.data(), input1_f_vals.data(), (size_t)N, (size_t)K, (size_t)N);

  std::vector<float> expected_vals(M * N);
  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; k++) {
        sum += input0_vals[m * K + k] * input1_f_vals[k * N + n];
      }
      expected_vals[m * N + n] = sum;
    }
  }

  test.AddInput<float>("A", {M, K}, input0_vals, false);
  test.AddInput<uint8_t>("B", {(int64_t)input1_vals.size()}, input1_vals, true);
  test.AddInput<int64_t>("B_shape", {(int64_t)2}, {(int64_t)K, (int64_t)N}, true);

  test.AddOutput<float>("Y", {M, N}, expected_vals);

  test.Run();
}

}  // namespace test
}  // namespace onnxruntime
