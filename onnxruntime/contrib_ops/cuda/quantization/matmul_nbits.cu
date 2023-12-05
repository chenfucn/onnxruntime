// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "matmul_nbits.cuh"

#include "blk_q4/f16_gemm_sm80.h"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, half scale, uint8_t zp, const half* a) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};
  uint4 vec_a = *(reinterpret_cast<const uint4*>(a));

  half2 element01 = __halves2half2(__uint2half_rn(values_quant & 0xF), __uint2half_rn((values_quant >> 4) & 0xF));
  half2 v0 = element01 * scale_half2 + zp_adjust2;

  half2 element23 = __halves2half2(__uint2half_rn((values_quant >> 8) & 0xF), __uint2half_rn((values_quant >> 12) & 0xF));
  half2 v1 = element23 * scale_half2 + zp_adjust2;

  half2 element45 = __halves2half2(__uint2half_rn((values_quant >> 16) & 0xF), __uint2half_rn((values_quant >> 20) & 0xF));
  half2 v2 = element45 * scale_half2 + zp_adjust2;

  half2 element67 = __halves2half2(__uint2half_rn((values_quant >> 24) & 0xF), __uint2half_rn((values_quant >> 28) & 0xF));
  half2 v3 = element67 * scale_half2 + zp_adjust2;

  v0 = v0 * (*(reinterpret_cast<half2*>(&(vec_a.x))));
  v1 = v1 * (*(reinterpret_cast<half2*>(&(vec_a.y))));
  v2 = v2 * (*(reinterpret_cast<half2*>(&(vec_a.z)))) + v0;
  v3 = v3 * (*(reinterpret_cast<half2*>(&(vec_a.w)))) + v1;
  v3 = v2 + v3;
  return float(v3.x) + float(v3.y);
}

__device__ __forceinline__ float AccumulateEightElements(uint32_t values_quant, float scale, uint8_t zp, const float* a) {
  float4 a_vec_0 = *(reinterpret_cast<const float4*>(a));
  float4 a_vec_1 = *(reinterpret_cast<const float4*>(a + 4));

  float zp_adjust = -scale * zp;
  float v0 = float(values_quant & 0xF) * scale + zp_adjust;
  float v1 = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  float v2 = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  float v3 = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  float v4 = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  float v5 = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  float v6 = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  float v7 = float((values_quant >> 28) & 0xF) * scale + zp_adjust;

  v0 = v0 * a_vec_0.x;
  v1 = v1 * a_vec_0.y;
  v2 = v2 * a_vec_0.z;
  v3 = v3 * a_vec_0.w;
  v4 = v4 * a_vec_1.x + v0;
  v5 = v5 * a_vec_1.y + v1;
  v6 = v6 * a_vec_1.z + v2;
  v7 = v7 * a_vec_1.w + v3;
  return v4 + v5 + v6 + v7;
}

constexpr int kColsPerThreadBlock = 8;
constexpr int kWarpSize = 32;

// kernel for 4bits quantized gemv, i.e., computing A(1,K) x B(K, N)
// B(K, N) is quantized blockwise with 4bits and stored as [N, (K + block_size - 1)/block_size, blob]
// The thread block size is (kWarpSize, kColsPerThreadBlock) and grid size is (N/kColsPerThreadBlock, 1)
// Each thread block computes [1, K] x [kColsPerThreadBlock, (K + block_size - 1)/block_size, blob],
//     i.e., computing kColsPerThreadBlock per block and a warp reduce (1, K) x (K)
template <class T, int block_size>
__global__ void MatMulFloatInt4Kernel(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int blocks_per_K) {
  int n_block_id = blockIdx.x;
  int m_id = blockIdx.y;
  int lane_id = threadIdx.x;
  int warp_id = threadIdx.y;
  int n_id = n_block_id * kColsPerThreadBlock + warp_id;
  int thread_id = warp_id * kWarpSize + lane_id;
  constexpr int k_per_iter = 256;
  int k_iter = k / k_per_iter;

  // blocks_per_k is the number of scales and zero points on the k dim
  const int b_zp_k = (blocks_per_K + 1)/ 2;

  extern __shared__ char shared_buffer[];

  // load scale to shared buffer
  T* b_scale_vec = (T*)shared_buffer;
  uint8_t* b_zp_vec = reinterpret_cast<uint8_t*>(b_scale_vec + kColsPerThreadBlock * blocks_per_K);
  int offset = n_block_id * kColsPerThreadBlock * blocks_per_K;
  for (int i = thread_id; i < kColsPerThreadBlock * blocks_per_K; i += kColsPerThreadBlock * kWarpSize) {
    b_scale_vec[i] = scales_data[offset + i];
  }

  int zp_offset = n_block_id * kColsPerThreadBlock * b_zp_k;
  for (int i = thread_id; i < kColsPerThreadBlock * b_zp_k; i += kColsPerThreadBlock * kWarpSize) {
    b_zp_vec[i] = zero_points != nullptr ? zero_points[zp_offset + i] : uint8_t(0x88);
  }
  __syncthreads();

  a_data += m_id * k;
  b_data_quant += n_id * blocks_per_K * (block_size / 2);

  const int scale_col_offset = warp_id * blocks_per_K;
  const int zp_col_offset = warp_id * b_zp_k;

  float sum = 0.f;
  int k_id = 0;
  for (; k_id < (k & 0xffffff00); k_id += k_per_iter) {
    const int t_k = k_id + (lane_id << 3);  // k index for this thread
    const int t_meta_k = t_k / block_size;  // k index for this thread, points to the scale and zero point
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + (t_k >> 1)));
    T scale = b_scale_vec[scale_col_offset + t_meta_k];
    uint8_t zp = b_zp_vec[zp_col_offset + t_meta_k/2];
    zp = (t_meta_k & 0x01) ? (zp >> 4) : (zp & 0x0f);
    sum += AccumulateEightElements(value, scale, zp, a_data + k_id + (lane_id << 3));
  }

  // handle reminder
  if (k_id + lane_id * 8 < k) {
    const int t_k = k_id + (lane_id << 3);  // k index for this thread
    const int t_meta_k = t_k / block_size;  // k index for this thread, points to the scale and zero point
    uint32_t value = *(reinterpret_cast<const uint32_t*>(b_data_quant + k_iter * 128 + lane_id * 4));
    T scale = b_scale_vec[scale_col_offset + t_meta_k];
    uint8_t zp = b_zp_vec[zp_col_offset + t_meta_k/2];
    zp = (t_meta_k & 0x01) ? (zp >> 4) : (zp & 0x0f);
    sum += AccumulateEightElements(value, scale, zp, a_data + k_id + (lane_id << 3));
  }

  // warp reduction
  for (int i = 16; i > 0; i = i / 2) {
    sum += __shfl_down_sync(0xffffffff, sum, i);
  }

  if (lane_id == 0) {
    output[m_id * n + n_id] = sum;
  }
}

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream) {
  if (n % kColsPerThreadBlock != 0 || k % 8 != 0 || m > 1) {
    return false;
  }
  dim3 blocks((n + kColsPerThreadBlock - 1) / kColsPerThreadBlock, m);
  dim3 threads(kWarpSize, kColsPerThreadBlock);
  int blocks_per_K = (k + block_size - 1) / block_size;
  int blocks_per_thread_block = blocks_per_K * kColsPerThreadBlock;
  int shared_mem_size = sizeof(T) * blocks_per_thread_block + blocks_per_thread_block / 2;
  if (shared_mem_size > shared_mem_per_block) {
    return false;
  }

  if (16 == block_size) {
    MatMulFloatInt4Kernel<T, 16><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (32 == block_size) {
    MatMulFloatInt4Kernel<T, 32><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (64 == block_size) {
    MatMulFloatInt4Kernel<T, 64><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else if (128 == block_size) {
    MatMulFloatInt4Kernel<T, 128><<<blocks, threads, shared_mem_size, stream>>>(
        output, a_data, b_data_quant, scales_data, zero_points, m, n, k, blocks_per_K);
  } else {
    ORT_THROW("block size ", block_size, " is not supported");
  }

  return true;
}

template bool TryMatMul4Bits<float>(
    float* output,
    const float* a_data,
    const uint8_t* b_data_quant,
    const float* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

template bool TryMatMul4Bits<half>(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    int shared_mem_per_block,
    cudaStream_t stream);

/**
 * @brief Helper function to run the GEMM kernel for 4bits quantized gemm on SM80.
 * Only support fp16 for now.
*/
template<
    int block_size,
    bool column_wise_blocking,
    bool small_m,
    bool has_offsets>
Status blkq4_gemm_sm80(int m, int n, int k, cudaStream_t stream,
                     gsl::span<half const> a,
                     gsl::span<uint8_t const> weights,
                     gsl::span<half const> scales,
                     gsl::span<uint8_t const> offsets,
                     gsl::span<half> output) {

  using ElementDequant = cutlass::half_t;
  using QuantBlocking =
    typename std::conditional<column_wise_blocking,
                     cutlass::MatrixShape<block_size, 1>,
                     cutlass::MatrixShape<1, block_size>>::type;

  using GemmRunner = BlkQ4F16GemmImpl<ElementDequant, QuantBlocking, small_m, has_offsets>;

  using ElementAccumulator = typename GemmRunner::ElementAccumulator;
  using ElementComputeEpilogue = typename GemmRunner::ElementComputeEpilogue;
  using ElementOutput = typename GemmRunner::ElementOutput;
  using ElementW = typename GemmRunner::ElementW;
  using ElementWPack = typename GemmRunner::ElementWPack;
  using ElementQScale = typename GemmRunner::ElementQScale;
  using ElementQOffset = typename GemmRunner::ElementQOffset;

  using LayoutInputA = typename GemmRunner::LayoutInputA;
  using LayoutOutput = typename GemmRunner::LayoutOutput;
  using LayoutInputWPack = typename GemmRunner::LayoutInputWPack;
  using LayoutInputQScale = typename GemmRunner::LayoutInputQScale;

  const cutlass::gemm::GemmCoord problem_size = {m, n, k};

  ORT_RETURN_IF_NOT(a.size_bytes() == m * k * sizeof(ElementDequant), "Activation tensor size is not correct");
  cutlass::TensorRef<ElementDequant const, LayoutInputA> ref_a(
    reinterpret_cast<ElementDequant const *>(a.data()),
    LayoutInputA::packed({m, k}));

  ORT_RETURN_IF_NOT(weights.size_bytes() == k/2 * n/2 * sizeof(ElementWPack), "weights size is not correct");
  cutlass::TensorRef<ElementWPack const, LayoutInputWPack> ref_W(
    reinterpret_cast<ElementWPack const *>(weights.data()),
    LayoutInputWPack::packed({k/2, n/2}));

  ORT_RETURN_IF_NOT(scales.size_bytes() == (k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQScale),
              "scales size is not correct");
  cutlass::TensorRef<ElementQScale const, LayoutInputQScale> ref_scales(
    reinterpret_cast<ElementQScale const *>(scales.data()),
    LayoutInputQScale::packed({k/QuantBlocking::kRow, n/QuantBlocking::kColumn}));

  ORT_RETURN_IF_NOT(output.size_bytes() == m * n * sizeof(ElementOutput), "output size is not correct");
  cutlass::TensorRef<ElementOutput, LayoutOutput> ref_output(
    reinterpret_cast<ElementOutput *>(output.data()),
    LayoutOutput::packed({m, n}));

  // run GEMM
  cutlass::Status status;
  if constexpr (has_offsets) {
    ORT_RETURN_IF_NOT(offsets.size_bytes() == (k/QuantBlocking::kRow) * (n/QuantBlocking::kColumn) * sizeof(ElementQOffset),
                "offsets size is not correct");
    cutlass::TensorRef<ElementQOffset const, LayoutInputQScale> ref_offsets(
      reinterpret_cast<ElementQOffset const *>(offsets.data()),
      LayoutInputQScale::packed({k/QuantBlocking::kRow, n/QuantBlocking::kColumn}));
    status = GemmRunner::run(
      stream, problem_size, ref_a, ref_W, ref_scales, ref_offsets,
      ref_output, ref_output);
  } else {
    status = GemmRunner::run(
      stream, problem_size, ref_a, ref_W, ref_scales,
      ref_output, ref_output);
  }
  ORT_RETURN_IF_NOT(status == cutlass::Status::kSuccess, "Kernel execution failed: ", cutlassGetStatusString(status));
  return Status::OK();
}

Status blkq4_fp16_gemm_sm80_dispatch(
  int block_size,
  bool column_wise_blocking,
  int m, int n, int k, cudaStream_t stream,
  gsl::span<half const> a,
  gsl::span<uint8_t const> weights,
  gsl::span<half const> scales,
  gsl::span<uint8_t const> offsets,
  gsl::span<half> output) {

  switch (block_size)
  {
  case 16:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<16, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<16, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 32:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<32, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<32, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;

  case 64:
    if (column_wise_blocking) {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, true, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, true, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, true, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, true, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    } else {
      if (m > 16) {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, false, false, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, false, false, true>(m, n, k, stream, a, weights, scales, offsets, output);
      } else {
        if (offsets.empty())
          return blkq4_gemm_sm80<64, false, true, false>(m, n, k, stream, a, weights, scales, offsets, output);
        else
          return blkq4_gemm_sm80<64, false, true, true>(m, n, k, stream, a, weights, scales, offsets, output);
      }
    }
    break;
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported block size: ", block_size);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
