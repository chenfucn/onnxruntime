// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This module define MatMulFp32Q4 operator, it is basically
// matmul float32 with right hand side being a 2-D matrix
// pre-packed and block-compacted into int4
//

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "matmul_nbits.cuh"
#include "dequantize_blockwise.cuh"

#include "blk_q4/f16_prepack_sm80.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
using namespace onnxruntime::cuda;

template <typename T>
class MatMulNBits final : public CudaKernel {
 public:
  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for MatMulNBits op,"
                " additional bits support is planned.");
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool column_wise_quant_blk_{true};
};

template <typename T>
Status MatMulNBits<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor* a = ctx->Input<Tensor>(0);
  const Tensor* b = ctx->Input<Tensor>(1);
  const Tensor* scales = ctx->Input<Tensor>(2);
  const Tensor* zero_points = ctx->Input<Tensor>(3);

  const auto* a_data = a->Data<T>();
  const uint8_t* blob_data = b->Data<uint8_t>();
  const auto* scales_data = scales->Data<T>();
  const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

  typedef typename ToCudaType<T>::MappedType CudaT;

  constexpr bool transa = false;
  constexpr bool transb = true;
  MatMulComputeHelper helper;
  TensorShape b_shape({N_, K_});
  ORT_RETURN_IF_ERROR(
      helper.Compute(a->Shape(), b_shape, transa, transb));

  Tensor* Y = ctx->Output(0, helper.OutputShape());
  // Bail out early if the output is going to be empty
  if (Y->Shape().Size() == 0) return Status::OK();

  bool is_4bit_done = TryMatMul4Bits(
      reinterpret_cast<CudaT*>(Y->MutableData<T>()),
      reinterpret_cast<const CudaT*>(a_data),
      blob_data,
      reinterpret_cast<const CudaT*>(scales_data),
      zero_points_data,
      SafeInt<int>(helper.M()),
      SafeInt<int>(helper.N()),
      SafeInt<int>(helper.K()),
      SafeInt<int>(block_size_),
      SafeInt<int>(GetDeviceProp().sharedMemPerBlock),
      static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  if (!is_4bit_done) {
    int64_t K_padded = (K_ + block_size_ - 1) / block_size_ * block_size_;
    IAllocatorUniquePtr<T> b_data_ptr = GetScratchBuffer<T>(N_ * K_padded, ctx->GetComputeStream());
    auto* b_data = b_data_ptr.get();
    if (column_wise_quant_blk_) {
      // column-wise block
      ORT_RETURN_IF_ERROR(Dequantize4Bits(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          zero_points_data,
          SafeInt<int>(K_padded),
          SafeInt<int>(N_),
          SafeInt<int>(block_size_),
          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
    } else {
      // row-wise block
      K_padded = K_;

      ORT_RETURN_IF_ERROR(DequantizeBlockwise4b(
          reinterpret_cast<CudaT*>(b_data),
          blob_data,
          reinterpret_cast<const CudaT*>(scales_data),
          zero_points_data,
          SafeInt<int>(block_size_),
          column_wise_quant_blk_,
          SafeInt<int>(K_),
          SafeInt<int>(N_),
          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
    }
#if 0
  cudaStreamSynchronize(static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
  T* b_data_cpu = new T[K_ * N_];
  cudaMemcpy(b_data_cpu, b_data, K_ * N_ * sizeof(T), cudaMemcpyDeviceToHost);
  delete[] b_data_cpu;
#endif

    const CudaT alpha = ToCudaType<T>::FromFloat(1.f);
    const CudaT zero = ToCudaType<T>::FromFloat(0.f);

    if (helper.OutputOffsets().size() == 1) {
      CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
          GetCublasHandle(ctx),
          CUBLAS_OP_T,
          CUBLAS_OP_N,
          SafeInt<int>(helper.N()),
          SafeInt<int>(helper.M()),
          SafeInt<int>(helper.K()),
          &alpha,
          reinterpret_cast<const CudaT*>(b_data),
          SafeInt<int>(K_padded),
          reinterpret_cast<const CudaT*>(a_data),
          helper.Lda(transa),
          &zero,
          reinterpret_cast<CudaT*>(Y->MutableData<T>()),
          helper.Ldc(),
          GetDeviceProp()));
    }
  }

  return Status::OK();
}


template <>
class MatMulNBits<MLFloat16> final : public CudaKernel {
 public:
  using CudaT = typename ToCudaType<MLFloat16>::MappedType;
  static_assert(sizeof(CudaT) == sizeof(MLFloat16), "Type T and CudaT should have same size.");

  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info), packed_buf_(this) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for MatMulNBits op,"
                " additional bits support is planned.");

    const Tensor* zp;
    has_offsets_ = Info().TryGetConstantInput(3, &zp);

    auto device_prop = GetDeviceProp();
    switch (block_size_)
    {
    case 16:
      if (column_wise_quant_blk_) {
        SetPrepackFields<16, true>(device_prop);
      } else {
        SetPrepackFields<16, false>(device_prop);
      }
      break;
    case 32:
      if (column_wise_quant_blk_) {
        SetPrepackFields<32, true>(device_prop);
      } else {
        SetPrepackFields<32, false>(device_prop);
      }
      break;
    case 64:
      if (column_wise_quant_blk_) {
        SetPrepackFields<64, true>(device_prop);
      } else {
        SetPrepackFields<64, false>(device_prop);
      }
      break;
    }
  }

  template<int BlkSize, bool ColumnWiseQuantBlk>
  void SetPrepackFields(const cudaDeviceProp& device_prop){
    bool should_prepack = onnxruntime::cuda::BlkQuantGemmSm80Supported<MLFloat16, BlkSize, 4>(K_, N_, device_prop);
    packed_b_shape_ = BlockwiseQuantization<MLFloat16, BlkSize, 4, ColumnWiseQuantBlk>::get_quant_weights_shape(K_, N_);
    packed_meta_shape_ = BlockwiseQuantization<MLFloat16, BlkSize, 4, ColumnWiseQuantBlk>::get_quant_meta_shape(K_, N_);
    if (should_prepack) {
      size_t packed_buf_size = packed_b_shape_.product() + packed_meta_shape_.product() * sizeof(MLFloat16);
      tensors_to_prepack_ = 2;
      if (has_offsets_) {
        packed_buf_size += packed_meta_shape_.product();
        tensors_to_prepack_ = 3;
      }
      packed_buf_.AllocCpuPtr(packed_buf_size);
    }
  }

  gsl::span<uint8_t> GetPackedWeights() const {
    return gsl::make_span<uint8_t>(reinterpret_cast<uint8_t*>(packed_buf_.CpuPtr()),
                                   SafeInt<size_t>(packed_b_shape_.product()));
  }

  gsl::span<MLFloat16> GetPackedScales() const {
    auto* start = reinterpret_cast<MLFloat16*>(packed_buf_.CpuPtr() + packed_b_shape_.product());
    return gsl::make_span<MLFloat16>(start, SafeInt<size_t>(packed_meta_shape_.product()));
  }

  gsl::span<uint8_t> GetPackedOffsets() const {
    auto* start = reinterpret_cast<uint8_t*>(packed_buf_.CpuPtr() + packed_b_shape_.product()
        + packed_meta_shape_.product() * sizeof(CudaT));
    return gsl::make_span<uint8_t>(start, SafeInt<size_t>(packed_meta_shape_.product()));
  }

  gsl::span<uint8_t const> GetPackedWeightsGpu() const {
    return gsl::make_span<uint8_t const>(reinterpret_cast<uint8_t const*>(packed_buf_.GpuPtr()),
                                         SafeInt<size_t>(packed_b_shape_.product()));
  }

  gsl::span<CudaT const> GetPackedScalesGpu() const {
    CudaT const* start = reinterpret_cast<CudaT const*>(packed_buf_.GpuPtr() + packed_b_shape_.product());
    return gsl::make_span<CudaT const>(start, SafeInt<size_t>(packed_meta_shape_.product()));
  }

  gsl::span<uint8_t const> GetPackedOffsetsGpu() const {
    uint8_t const* start = reinterpret_cast<uint8_t const*>(packed_buf_.GpuPtr() + packed_b_shape_.product()
        + packed_meta_shape_.product() * sizeof(CudaT));
    return gsl::make_span<uint8_t const>(start, SafeInt<size_t>(packed_meta_shape_.product()));
  }

  Status
  PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
          /*out*/ bool& is_packed, /*out*/ PrePackedWeights* /*prepacked_weights*/) override {

    auto alloc_info = alloc->Info().ToString();
    std::cout << "alloc_info: " << alloc_info << std::endl;

    is_packed = false;
    if (tensors_to_prepack_ == 0) {
      return Status::OK();
    }

    if (input_idx == 1) {
      // packing weight
      uint8_t const* w_data = tensor.Data<uint8_t>();
      ORT_ENFORCE(tensor.Location().device.Type() == OrtDevice::CPU, "Unexpected non-CPU weight tensor: ", tensor.Location().device.ToString());
      gsl::span<uint8_t const> weights = gsl::make_span<uint8_t const>(w_data, SafeInt<size_t>(tensor.Shape().Size()));
      switch (block_size_)
      {
      case 16:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 16, 4, true>::prepack_weights(
              K_, N_, weights, GetPackedWeights());
        } else {
          BlockwiseQuantization<MLFloat16, 16, 4, false>::prepack_weights(
              K_, N_, weights, GetPackedWeights());
        }
        break;
      case 32:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 32, 4, true>::prepack_weights(
              K_, N_, weights, GetPackedWeights());
        } else {
          BlockwiseQuantization<MLFloat16, 32, 4, false>::prepack_weights(
              K_, N_, weights, GetPackedWeights());
        }
        break;
      case 64:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 64, 4, true>::prepack_weights(
              K_, N_, weights, GetPackedWeights());
        } else {
          BlockwiseQuantization<MLFloat16, 64, 4, false>::prepack_weights(
              K_, N_, weights, GetPackedWeights());
        }
        break;
      default:
        ORT_THROW("Unsupported block size: ", block_size_);
      }
      tensors_to_prepack_--;
    }

    if (input_idx == 2){
      // packing scales
      MLFloat16 const* s_data = tensor.Data<MLFloat16>();
      gsl::span<MLFloat16 const> scales = gsl::make_span<MLFloat16 const>(s_data, SafeInt<size_t>(tensor.Shape().Size()));
      gsl::span<MLFloat16> packed_scales = GetPackedScales();
      switch (block_size_)
      {
      case 16:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 16, 4, true>::prepack_quant_scales(
              K_, N_, scales, packed_scales);
        } else {
          BlockwiseQuantization<MLFloat16, 16, 4, false>::prepack_quant_scales(
              K_, N_, scales, packed_scales);
        }
        break;
      case 32:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 32, 4, true>::prepack_quant_scales(
              K_, N_, scales, packed_scales);
        } else {
          BlockwiseQuantization<MLFloat16, 32, 4, false>::prepack_quant_scales(
              K_, N_, scales, packed_scales);
        }
        break;
      case 64:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 64, 4, true>::prepack_quant_scales(
              K_, N_, scales, packed_scales);
        } else {
          BlockwiseQuantization<MLFloat16, 64, 4, false>::prepack_quant_scales(
              K_, N_, scales, packed_scales);
        }
        break;
      default:
        ORT_THROW("Unsupported block size: ", block_size_);
      }
      tensors_to_prepack_--;
    }

    if (input_idx == 3){
      // packing zero points
      uint8_t const* z_data = tensor.Data<uint8_t>();
      gsl::span<uint8_t const> zero_points = gsl::make_span<uint8_t const>(z_data, SafeInt<size_t>(tensor.Shape().Size()));
      gsl::span<uint8_t> packed_offsets = GetPackedOffsets();
      switch (block_size_)
      {
      case 16:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 16, 4, true>::prepack_quant_offsets(
              K_, N_, zero_points, packed_offsets);
        } else {
          BlockwiseQuantization<MLFloat16, 16, 4, false>::prepack_quant_offsets(
              K_, N_, zero_points, packed_offsets);
        }
        break;
      case 32:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 32, 4, true>::prepack_quant_offsets(
              K_, N_, zero_points, packed_offsets);
        } else {
          BlockwiseQuantization<MLFloat16, 32, 4, false>::prepack_quant_offsets(
              K_, N_, zero_points, packed_offsets);
        }
        break;
      case 64:
        if (column_wise_quant_blk_) {
          BlockwiseQuantization<MLFloat16, 64, 4, true>::prepack_quant_offsets(
              K_, N_, zero_points, packed_offsets);
        } else {
          BlockwiseQuantization<MLFloat16, 64, 4, false>::prepack_quant_offsets(
              K_, N_, zero_points, packed_offsets);
        }
        break;
      default:
        ORT_THROW("Unsupported block size: ", block_size_);
      }
      tensors_to_prepack_--;
    }

    if (tensors_to_prepack_ == 0) {
      // packed_buf_.CopyToGpu(ctx->GetComputeStream());
    }
    return Status::OK();
  }

  Status ComputeInternal(OpKernelContext* ctx) const override  {
    const Tensor* a = ctx->Input<Tensor>(0);

    constexpr bool transa = false;
    constexpr bool transb = true;
    MatMulComputeHelper helper;
    TensorShape b_shape({N_, K_});
    ORT_RETURN_IF_ERROR(
        helper.Compute(a->Shape(), b_shape, transa, transb));

    Tensor* Y = ctx->Output(0, helper.OutputShape());
    // Bail out early if the output is going to be empty
    if (Y->Shape().Size() == 0) return Status::OK();

    if (packed_buf_.count() > 0){
      // packed_buf_ is not empty, it's prepacked for sm80 specialized kernel
      CudaT* Y_data = reinterpret_cast<CudaT*>(Y->MutableData<MLFloat16>());
      auto y_span = gsl::make_span(Y_data, static_cast<size_t>(Y->Shape().Size()));

      // can't load DataAsSpan from libonnxruntime_providers_cuda.so why???
      CudaT const* a_data = reinterpret_cast<CudaT const*>(a->Data<MLFloat16>());
      auto a_span = gsl::make_span(a_data, static_cast<size_t>(a->Shape().Size()));

      return blkq4_fp16_gemm_sm80_dispatch(
          block_size_,
          column_wise_quant_blk_,
          SafeInt<int>(helper.M()),
          SafeInt<int>(N_),
          SafeInt<int>(K_),
          static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()),
          a_span,
          GetPackedWeightsGpu(),
          GetPackedScalesGpu(),
          has_offsets_ ? GetPackedOffsetsGpu() : gsl::span<uint8_t const,0>{},
          y_span);
    }

    const Tensor* b = ctx->Input<Tensor>(1);
    const Tensor* scales = ctx->Input<Tensor>(2);
    const Tensor* zero_points = ctx->Input<Tensor>(3);

    const auto* a_data = a->Data<MLFloat16>();
    const uint8_t* blob_data = b->Data<uint8_t>();
    const auto* scales_data = scales->Data<MLFloat16>();
    const auto* zero_points_data = zero_points == nullptr ? nullptr : zero_points->Data<uint8_t>();

    bool is_4bit_done = TryMatMul4Bits(
        reinterpret_cast<CudaT*>(Y->MutableData<MLFloat16>()),
        reinterpret_cast<const CudaT*>(a_data),
        blob_data,
        reinterpret_cast<const CudaT*>(scales_data),
        zero_points_data,
        SafeInt<int>(helper.M()),
        SafeInt<int>(helper.N()),
        SafeInt<int>(helper.K()),
        SafeInt<int>(block_size_),
        SafeInt<int>(GetDeviceProp().sharedMemPerBlock),
        static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle()));
    if (!is_4bit_done) {
      int64_t K_padded = (K_ + block_size_ - 1) / block_size_ * block_size_;
      IAllocatorUniquePtr<MLFloat16> b_data_ptr = GetScratchBuffer<MLFloat16>(N_ * K_padded, ctx->GetComputeStream());
      auto* b_data = b_data_ptr.get();
      if (column_wise_quant_blk_) {
        // column-wise block
        ORT_RETURN_IF_ERROR(Dequantize4Bits(
            reinterpret_cast<CudaT*>(b_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            zero_points_data,
            SafeInt<int>(K_padded),
            SafeInt<int>(N_),
            SafeInt<int>(block_size_),
            static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
      } else {
        // row-wise block
        K_padded = K_;

        ORT_RETURN_IF_ERROR(DequantizeBlockwise4b(
            reinterpret_cast<CudaT*>(b_data),
            blob_data,
            reinterpret_cast<const CudaT*>(scales_data),
            zero_points_data,
            SafeInt<int>(block_size_),
            column_wise_quant_blk_,
            SafeInt<int>(K_),
            SafeInt<int>(N_),
            static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())));
      }

      const CudaT alpha = ToCudaType<MLFloat16>::FromFloat(1.f);
      const CudaT zero = ToCudaType<MLFloat16>::FromFloat(0.f);

      if (helper.OutputOffsets().size() == 1) {
        CUBLAS_RETURN_IF_ERROR(cublasGemmHelper(
            GetCublasHandle(ctx),
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            SafeInt<int>(helper.N()),
            SafeInt<int>(helper.M()),
            SafeInt<int>(helper.K()),
            &alpha,
            reinterpret_cast<const CudaT*>(b_data),
            SafeInt<int>(K_padded),
            reinterpret_cast<const CudaT*>(a_data),
            helper.Lda(transa),
            &zero,
            reinterpret_cast<CudaT*>(Y->MutableData<MLFloat16>()),
            helper.Ldc(),
            GetDeviceProp()));
      }
    }

    return Status::OK();
  }

 private:
  int64_t K_;
  int64_t N_;
  int64_t block_size_;
  int64_t nbits_;
  bool column_wise_quant_blk_{true};
  bool has_offsets_{false};

  // pre-packed weight
  int tensors_to_prepack_{0};
  Position<2> packed_b_shape_;
  Position<2> packed_meta_shape_;
  CudaAsyncBuffer<uint8_t> packed_buf_;
};


ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    float,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits<float>);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MatMulNBits,
    kMSDomain,
    1,
    MLFloat16,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<MLFloat16>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()),
    MatMulNBits<MLFloat16>);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
