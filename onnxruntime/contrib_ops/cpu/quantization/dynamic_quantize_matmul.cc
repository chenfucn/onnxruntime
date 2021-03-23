// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/cpu/math/matmul_integer_base.h"
#include "core/util/math_cpuonly.h"
#include "core/util/qmath.h"

#include <algorithm>

namespace onnxruntime {
namespace contrib {

/**
 * @brief Data structure for computing each Gemm in a thread
*/
struct GemmWorkContext {
 public:
  GemmWorkContext(float* Output,
                  size_t LeadingDimensionOutput,
                  const float* Scale,
                  const float* Bias,
                  MLAS_QGEMM_OUTPUT_MODE Mode = MLAS_QGEMM_OUTPUT_MODE::ZeroMode,
                  MLAS_QUANTIZATION_GRANULARITY QuantGran = MLAS_QUANTIZATION_GRANULARITY::PerMatrix) : scale_bias_processor_(Output, LeadingDimensionOutput, Scale, Bias, Mode, QuantGran)
  {
    work_block_.Parameters = &params_;
    params_.OutputProcessor = &scale_bias_processor_;
  }

  MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR scale_bias_processor_;
  MLAS_GEMM_U8X8_PARAMETERS params_;
  MLAS_GEMM_U8X8_WORK_BLOCK work_block_;
};

class MatMulIntegerToFloatBase : public MatMulIntegerBase {
 public:
  MatMulIntegerToFloatBase(const OpKernelInfo& info) : MatMulIntegerBase(info) {
  }

  enum OutputTensors : int { OUT_Y = 0 };

protected:
  Status ComputeCommon(OpKernelContext* ctx,
                       const uint8_t* a_data,
                       const TensorShape& a_shape,
                       uint8_t a_zero_point,
                       const Tensor* b,
                       uint8_t b_zero_point,
                       float multiplier,
                       const Tensor* bias_tensor) const;
};

Status MatMulIntegerToFloatBase::ComputeCommon(OpKernelContext* ctx,
                                               const uint8_t* a_data,
                                               const TensorShape& a_shape,
                                               uint8_t a_zero_point,
                                               const Tensor* b,
                                               uint8_t b_zero_point,
                                               float multiplier,
                                               const Tensor* bias_tensor) const {
  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a_shape, packed_b_ ? b_shape_ : b->Shape()));
  Tensor* y = ctx->Output(OUT_Y, helper.OutputShape());

  // Bail out early if the output is going to be empty
  if (y->Shape().Size() == 0)
    return Status::OK();

  auto* y_data = y->template MutableData<float>();
  const auto* bias_data = bias_tensor != nullptr ? bias_tensor->Data<float>() : nullptr;

  size_t M = static_cast<size_t>(helper.M());
  size_t N = static_cast<size_t>(helper.N());
  size_t K = static_cast<size_t>(helper.K());
  const int num_gemms = static_cast<int>(helper.OutputOffsets().size());
  if (num_gemms == 1) {
    MLAS_GEMM_U8X8_PARAMETERS gemm_params;
    gemm_params.M = M;
    gemm_params.N = N;
    gemm_params.K = K;
    gemm_params.lda = gemm_params.K;
    gemm_params.ZeroPointA = a_zero_point;
    gemm_params.ldb = gemm_params.N;
    gemm_params.ZeroPointB = &b_zero_point;
    gemm_params.ldc = gemm_params.N;
    MLAS_QGEMM_SCALE_BIAS_OUTPUT_PROCESSOR scale_bias_processor(
        y_data,
        N,
        &multiplier,
        bias_data);

    gemm_params.A = a_data;
    if (packed_b_) {
      gemm_params.B = packed_b_.get();
      gemm_params.BIsPacked = true;
      gemm_params.BIsSigned = b_is_signed_;
    } else {
      gemm_params.B = static_cast<const uint8_t*>(b->DataRaw());
      gemm_params.BIsSigned = b->IsDataType<int8_t>();
    }
    gemm_params.C = reinterpret_cast<int32_t*>(y_data);
    gemm_params.OutputProcessor = &scale_bias_processor;
    MlasGemm(&gemm_params, ctx->GetOperatorThreadPool());
    return Status::OK();
  }

  // Segment the work for parallelization. This is a two dimentional work partition.
  // The idea is to generate a group of not-too-small work chunks and let the thread
  // pool load balance them
  // TODO!! maybe just delegate this entirely to the threadpool?
  const std::ptrdiff_t num_work_blks_hint = concurrency::ThreadPool::DegreeOfParallelism(ctx->GetOperatorThreadPool()) * std::ptrdiff_t(4);
  std::ptrdiff_t num_blks_per_gemm = (num_work_blks_hint + num_gemms - 1) / num_gemms;
  std::ptrdiff_t thread_m;
  std::ptrdiff_t thread_n;
  double cost;
  MlasGemmSegWork(M, N, K, num_blks_per_gemm, num_blks_per_gemm, thread_m, thread_n, cost);

  // setup parameters for each gemm so that we can run them in parallel
  std::vector<GemmWorkContext> gemm_context;
  gemm_context.reserve(num_gemms);
  for (size_t gemm_idx = 0; gemm_idx < num_gemms; gemm_idx++) {
    gemm_context.emplace_back(y_data + helper.OutputOffsets()[gemm_idx],
                              N,
                              &multiplier,
                              bias_data);

    auto& gemm_params = gemm_context[gemm_idx].params_;
    gemm_params.M = M;
    gemm_params.N = N;
    gemm_params.K = K;
    gemm_params.lda = gemm_params.K;
    gemm_params.ZeroPointA = a_zero_point;
    gemm_params.ldb = gemm_params.N;
    gemm_params.ZeroPointB = &b_zero_point;
    gemm_params.ldc = gemm_params.N;
    gemm_params.A = a_data + helper.LeftOffsets()[gemm_idx];
    if (packed_b_) {
      gemm_params.B = packed_b_.get();
      gemm_params.BIsPacked = true;
      gemm_params.BIsSigned = b_is_signed_;
    } else {
      gemm_params.B = static_cast<const uint8_t*>(b->DataRaw()) + +helper.RightOffsets()[gemm_idx];
      gemm_params.BIsSigned = b->IsDataType<int8_t>();
    }
    gemm_params.C = reinterpret_cast<int32_t*>(y_data) + helper.OutputOffsets()[gemm_idx];

    gemm_context[gemm_idx].work_block_.ThreadCountM = thread_m;
    gemm_context[gemm_idx].work_block_.ThreadCountN = thread_n;
  }

  concurrency::ThreadPool::TryParallelFor(
      ctx->GetOperatorThreadPool(), num_blks_per_gemm * num_gemms, cost,
      [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
        for (auto idx = begin; idx < end; idx++) {
          const auto gemm_i = idx / num_blks_per_gemm;
          const auto blk_i = idx % num_blks_per_gemm;
          // TODO!! coalescing consequtive iterations in the same call
          MlasGemmU8X8Threaded(&(gemm_context[gemm_i].work_block_), blk_i);
        }
      });

  return Status::OK();
}

class DynamicQuantizeMatMul final : public MatMulIntegerToFloatBase {
 public:
  DynamicQuantizeMatMul(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_B_SCALE = 2,
    IN_B_ZERO_POINT = 3,
    IN_BIAS = 4
  };

 protected:
  int GetBIdx() override { return IN_B; }
};

class MatMulIntegerToFloat final : public MatMulIntegerToFloatBase {
 public:
  MatMulIntegerToFloat(const OpKernelInfo& info) : MatMulIntegerToFloatBase(info) {}

  Status Compute(OpKernelContext* context) const override;

  enum InputTensors : int {
    IN_A = 0,
    IN_B = 1,
    IN_A_SCALE = 2,
    IN_B_SCALE = 3,
    IN_A_ZERO_POINT = 4,
    IN_B_ZERO_POINT = 5,
    IN_BIAS = 6
  };

 protected:
  int GetBIdx() override { return IN_B; }
};

Status DynamicQuantizeMatMul::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale_tensor),
              "DynamicQuantizeMatMul : input B scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
  float b_scale = *b_scale_tensor->template Data<float>();

  const Tensor* b_zero_point_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  uint8_t b_zero_point = 0;
  if (b_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point_tensor),
                "DynamicQuantizeMatMul : input B zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    b_zero_point = *static_cast<const uint8_t*>(b_zero_point_tensor->DataRaw());
  }

  auto start_time = std::chrono::high_resolution_clock::now();
  // calculate quantization parameter of a
  const float* a_data = a->template Data<float>();
  int64_t num_of_elements = a->Shape().Size();

  float a_scale;
  uint8_t a_zero_point;
  GetQuantizationParameter(a_data, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  auto end_minmax = std::chrono::high_resolution_clock::now();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(ctx->GetTempSpaceAllocator(&allocator));
  uint8_t* a_data_quant = static_cast<uint8_t*>(allocator->Alloc(SafeInt<size_t>(num_of_elements) * sizeof(uint8_t)));
  BufferUniquePtr a_buffer_quant_holder(a_data_quant, BufferDeleter(allocator));

  ParQuantizeLinear(a_data, a_data_quant, num_of_elements, a_scale, a_zero_point, ctx->GetOperatorThreadPool());

  auto end_quant = std::chrono::high_resolution_clock::now();

  auto status = ComputeCommon(ctx,
                       a_data_quant,
                       a->Shape(),
                       a_zero_point,
                       b,
                       b_zero_point,
                       a_scale * b_scale,
                       ctx->Input<Tensor>(IN_BIAS));

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> minmax_dur = end_minmax - start_time;
  std::chrono::duration<double> quant_dur = end_quant - end_minmax;
  std::chrono::duration<double> matmul_dur = end - end_quant;
  std::chrono::duration<double> dur = end - start_time;

  const auto& logger = ctx->Logger();
  LOGS(logger, VERBOSE) << "{ node : \"" << Node().Name() << "\", A : \""
                        << a->Shape().ToString().c_str() << "\", B : \""
                        << (b ? b->Shape() : b_shape_).ToString().c_str() << "\", b_signed: \""
                        << (b ? b->IsDataType<int8_t>() : b_is_signed_)
                        << "\", minmax: " << minmax_dur.count() << ", quant: " << quant_dur.count() << ", matmul: " << matmul_dur.count()
                        << ", dur: " << dur.count()
                        << "}";
  return status;
}

Status MatMulIntegerToFloat::Compute(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(IN_A);
  const Tensor* b = packed_b_ ? nullptr : ctx->Input<Tensor>(IN_B);

  const Tensor* a_scale_tensor = ctx->Input<Tensor>(IN_A_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(a_scale_tensor),
              "MatMulIntegerToFloat : input A scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
  float a_scale = *a_scale_tensor->template Data<float>();

  const Tensor* b_scale_tensor = ctx->Input<Tensor>(IN_B_SCALE);
  ORT_ENFORCE(IsScalarOr1ElementVector(b_scale_tensor),
              "MatMulIntegerToFloat : input B scale must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
  float b_scale = *b_scale_tensor->template Data<float>();

  // validate zero points
  uint8_t a_zero_point = 0;
  const Tensor* a_zero_point_tensor = ctx->Input<Tensor>(IN_A_ZERO_POINT);
  if (a_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point_tensor),
                "MatMulIntegerToFloat : input A zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    a_zero_point = *a_zero_point_tensor->Data<uint8_t>();
  }

  uint8_t b_zero_point = 0;
  const Tensor* b_zero_point_tensor = ctx->Input<Tensor>(IN_B_ZERO_POINT);
  if (b_zero_point_tensor != nullptr) {
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point_tensor),
                "MatMulIntegerToFloat : input B zero point must be a scalar or 1D tensor of size 1. Per-Channel is not supported yet.");
    b_zero_point = *static_cast<const uint8_t*>(b_zero_point_tensor->DataRaw());
  }

  return ComputeCommon(ctx,
                       a->Data<uint8_t>(),
                       a->Shape(),
                       a_zero_point,
                       b,
                       b_zero_point,
                       a_scale * b_scale,
                       ctx->Input<Tensor>(IN_BIAS));
}

ONNX_OPERATOR_TYPED_KERNEL_EX(
    DynamicQuantizeMatMul,
    kMSDomain,
    1,
    float,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
    DynamicQuantizeMatMul);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulIntegerToFloat,
    kMSDomain,
    1,
    uint8_t,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", {DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<float>()),
    MatMulIntegerToFloat);

}  // namespace contrib
}  // namespace onnxruntime
