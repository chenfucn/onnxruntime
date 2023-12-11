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
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/matrix_layout.h"
#include "matmul_nbits.cuh"
#include "dequantize_blockwise.cuh"

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

template <
    typename ElementT,
    int block_size,
    int qbits,
    bool Columnwise,
    bool ExtraBoundsCheck = false>
struct BlockwiseQuantization {
  static_assert(qbits == 4, "Only 4b block quantization is supported!");
  static_assert(sizeof(ElementT) == 2, "Only 16b floating point types are supported!");

  using QuantBlocking =
      std::conditional_t<Columnwise,
                         MatrixShape<block_size, 1>,
                         MatrixShape<1, block_size>>;

  using ElementW = uint8_t;  // <- Weight is int4, uint8 for two of them
  // We pack 4 weights into one 16b element, so we can leverage cutlass tile iterators
  // for async share memory loading, and minimizing bank conflict during matrix loading
  using ElementWPack = ElementT;
  using LayoutWPack = ColumnMajorLayout;  // <- layout of packed weight, must be column major

  // Current Ampere kernel use 8b zero point, need to shrink it to 4b in the future
  using ElementQOffset = uint8_t;

  // Layout of the quantization parameters (scales and zero points)
  // Major on the dimension that has the most parameters per squarish weight block.
  // E.g. for column-wise quantization, a [64, 64] block has [2, 64] parameters,
  // where each row has more data, so we use row major layout so that warp threads
  // can use less load instructions to load more parameters.
  using LayoutQmeta =
      typename std::conditional<Columnwise,
                                RowMajorLayout, ColumnMajorLayout>::type;

  /**
   * @brief  Get quantized weight tensor dimensions.
   * Actual weight type is int4, we use ElementW = uint8 to avoid possible compilation
   * troubles. Since the layout is column major, we are packing 2 weights in a column
   * into one int8
   */
  static inline auto get_quant_weights_shape(int rows, int columns) {
    return make_Position(rows / 2, columns);
  }

  static inline auto get_quant_meta_shape(int rows, int columns) {
    return make_Position(rows / QuantBlocking::kRow, columns / QuantBlocking::kColumn);
  }
};

static inline size_t align_to_16(size_t size) {
  return ((size + 15) / 16) * 16;
}

template <>
class MatMulNBits<MLFloat16> final : public CudaKernel {
 public:
  using CudaT = typename ToCudaType<MLFloat16>::MappedType;
  static_assert(sizeof(CudaT) == sizeof(MLFloat16), "Type T and CudaT should have same size.");

  MatMulNBits(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("K", &K_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("N", &N_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("block_size", &block_size_));
    ORT_ENFORCE(Status::OK() == info.GetAttr<int64_t>("bits", &nbits_));
    ORT_ENFORCE(nbits_ == 4,
                "Only 4b quantization is supported for MatMulNBits op,"
                " additional bits support is planned.");

    const Tensor* zp;
    has_offsets_ = Info().TryGetConstantInput(3, &zp);

    // prepacking related
    if (K_ % block_size_ != 0) {
      return;
    }

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
    ORT_UNUSED_PARAMETER(device_prop);
    packed_b_shape_ = BlockwiseQuantization<MLFloat16, BlkSize, 4, ColumnWiseQuantBlk>::get_quant_weights_shape(K_, N_);
    packed_meta_shape_ = BlockwiseQuantization<MLFloat16, BlkSize, 4, ColumnWiseQuantBlk>::get_quant_meta_shape(K_, N_);
      size_t packed_buf_size = align_to_16(packed_b_shape_.product())
                               + align_to_16(packed_meta_shape_.product() * sizeof(MLFloat16));
      tensors_to_prepack_ = 2;
      if (has_offsets_) {
        packed_buf_size += align_to_16(packed_meta_shape_.product());
        tensors_to_prepack_ = 3;
      }
      pack_buf_gpu_ = IAllocator::MakeUniquePtr<uint8_t>(
          Info().GetAllocator(OrtMemType::OrtMemTypeDefault), packed_buf_size);
  }

  gsl::span<uint8_t> GetPackedWeights() const {
    return gsl::make_span<uint8_t>(pack_buf_gpu_.get(), SafeInt<size_t>(packed_b_shape_.product()));
  }

  gsl::span<MLFloat16> GetPackedScales() const {
    MLFloat16* start = reinterpret_cast<MLFloat16*>(pack_buf_gpu_.get() + align_to_16(packed_b_shape_.product()));
    return gsl::make_span<MLFloat16>(start, SafeInt<size_t>(packed_meta_shape_.product()));
  }

  gsl::span<uint8_t> GetPackedOffsets() const {
    uint8_t* start = pack_buf_gpu_.get() + align_to_16(packed_b_shape_.product())
                     + align_to_16(packed_meta_shape_.product() * sizeof(MLFloat16));
    return gsl::make_span<uint8_t>(start, SafeInt<size_t>(packed_meta_shape_.product()));
  }

  Status
  PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
          /*out*/ bool& is_packed, /*out*/ PrePackedWeights* prepacked_weights) override {
    ORT_UNUSED_PARAMETER(alloc);
    ORT_UNUSED_PARAMETER(prepacked_weights);

    is_packed = false;
    if (tensors_to_prepack_ == 0) {
      return Status::OK();
    }

    if (input_idx == 1) {
      // packing weight
      uint8_t const* w_data = tensor.Data<uint8_t>();
      ORT_ENFORCE(tensor.Location().device.Type() == OrtDevice::GPU, "Unexpected non-GPU weight tensor: ", tensor.Location().device.ToString());
      gsl::span<uint8_t const> weights = gsl::make_span<uint8_t const>(w_data, SafeInt<size_t>(tensor.Shape().Size()));
      ORT_ENFORCE(weights.size_bytes() <= GetPackedWeights().size_bytes(), "Unexpected weight size: ", weights.size_bytes(), " vs ", GetPackedWeights().size_bytes(), " N: ", N_, " K: ", K_);
      auto cuda_err = cudaMemcpy(GetPackedWeights().data(), weights.data(), weights.size_bytes(), cudaMemcpyDeviceToDevice);
      if (cuda_err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cudaMemCpy failed: ", cudaGetErrorString(cuda_err));
      }
      is_packed = true;
      tensors_to_prepack_--;
    }

    if (input_idx == 2){
      // packing scales
      MLFloat16 const* s_data = tensor.Data<MLFloat16>();
      gsl::span<MLFloat16 const> scales = gsl::make_span<MLFloat16 const>(s_data, SafeInt<size_t>(tensor.Shape().Size()));
      ORT_ENFORCE(scales.size_bytes() <= GetPackedScales().size_bytes(), "Unexpected scale size: ", scales.size_bytes(), " vs ", GetPackedScales().size_bytes());
      auto cuda_err = cudaMemcpy(GetPackedScales().data(), scales.data(), scales.size_bytes(), cudaMemcpyDeviceToDevice);
      if (cuda_err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cudaMemCpy failed: ", cudaGetErrorString(cuda_err));
      }
      is_packed = true;
      tensors_to_prepack_--;
    }

    if (input_idx == 3 && has_offsets_){
      // packing zero points
      uint8_t const* z_data = tensor.Data<uint8_t>();
      gsl::span<uint8_t const> zero_points = gsl::make_span<uint8_t const>(z_data, SafeInt<size_t>(tensor.Shape().Size()));
      ORT_ENFORCE(zero_points.size_bytes() <= GetPackedOffsets().size_bytes(), "Unexpected zero point size: ", zero_points.size_bytes(), " vs ", GetPackedOffsets().size_bytes());
      auto cuda_err = cudaMemcpy(GetPackedOffsets().data(), zero_points.data(), zero_points.size_bytes(), cudaMemcpyDeviceToDevice);
      if (cuda_err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "cudaMemCpy failed: ", cudaGetErrorString(cuda_err));
      }
      is_packed = true;
      tensors_to_prepack_--;
    }

    return Status::OK();
  }

  Status ComputeInternal(OpKernelContext* ctx) const override {
    const Tensor* a = ctx->Input<Tensor>(0);
    const Tensor* b = pack_buf_gpu_ ? nullptr : ctx->Input<Tensor>(1);
    const Tensor* scales = pack_buf_gpu_ ? nullptr : ctx->Input<Tensor>(2);
    const Tensor* zero_points = pack_buf_gpu_ ? nullptr : ctx->Input<Tensor>(3);

    const auto* a_data = a->Data<MLFloat16>();
    const uint8_t* blob_data = pack_buf_gpu_ ? GetPackedWeights().data() : b->Data<uint8_t>();
    const auto* scales_data = pack_buf_gpu_ ? GetPackedScales().data() : scales->Data<MLFloat16>();
    const auto* zero_points_data = has_offsets_ ?
        (pack_buf_gpu_ ? GetPackedOffsets().data() : zero_points->Data<uint8_t>())
        : nullptr;

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
      auto b_data_ptr = GetScratchBuffer<MLFloat16>(N_ * K_padded, ctx->GetComputeStream());
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

  // prepack related
  int tensors_to_prepack_{0};
  Position<2> packed_b_shape_;
  Position<2> packed_meta_shape_;
  IAllocatorUniquePtr<uint8_t> pack_buf_gpu_;
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
