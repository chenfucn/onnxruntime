/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    q4gemm.cpp

Abstract:

    This module implements the fp32 matrix multiplication with compressed
    weight tensor (right hand side). The assumption is the right hand side
    tensor can be pre-packed and compressed using int-4 quantization to save
    memory. Quantized weights are expanded to fp32 before matrix
    multiplication.

--*/

#include "q4common.h"

#include <type_traits>
#include <immintrin.h>

struct MLAS_FP_Q4_GEMM_KERNEL_DEFAULT {
    static constexpr size_t StrideM = 256;
};

/**
 * @brief Horizontally sum 4 vectors and store
 *        the results in the returned vector
 */
static
MLAS_FORCEINLINE
__m128
FoldAccumulators(
    const __m512& acc0,
    const __m512& acc1,
    const __m512& acc2,
    const __m512& acc3
    )
{
    __m512 acc_lo01 = _mm512_unpacklo_ps(acc0, acc1);
    __m512 acc_hi01 = _mm512_unpackhi_ps(acc0, acc1);
    __m512 acc_lo23 = _mm512_unpacklo_ps(acc2, acc3);
    __m512 acc_hi23 = _mm512_unpackhi_ps(acc2, acc3);

    __m512 acc_lo0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
    __m512 acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_lo01), _mm512_castps_pd(acc_lo23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpacklo_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm512_castpd_ps(
        _mm512_unpackhi_pd(_mm512_castps_pd(acc_hi01), _mm512_castps_pd(acc_hi23)));
    acc_lo0123 = _mm512_add_ps(acc_lo0123, acc_hi0123);

    __m256 acc_y =
        _mm256_add_ps(_mm512_extractf32x8_ps(acc_lo0123, 0), _mm512_extractf32x8_ps(acc_lo0123, 1));
    return _mm_add_ps(_mm256_extractf32x4_ps(acc_y, 0), _mm256_extractf32x4_ps(acc_y, 1));
}

template<typename Q4Type>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernelAvx512f(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    const __m256i lowMask = _mm256_set1_epi8(0xF);

    for (size_t m = 0; m < CountM; m++) {
        const auto* b_col = PackedB;
        auto* sum_ptr = C;
        const auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN) - 4;
        while (nblk >= 0) {
            __m512 acc_lo0 = _mm512_setzero();
            __m512 acc_lo1 = _mm512_setzero();
            __m512 acc_lo2 = _mm512_setzero();
            __m512 acc_lo3 = _mm512_setzero();
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                size_t ck = std::min(CountK - k, (typename Q4Type::BlkLen));

                // Load A row vectors
                uint32_t mask = 0xffffffff >> (typename Q4Type::BlkLen - ck);
                __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k);

                mask = mask >> 16;
                __m512 av_hi = mask == 0 ? _mm512_setzero_ps()
                                         : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + 16);

                // Load 4 B column vectors (quantized to int4 blobs)
                const float scale_v0 = MlasQ4BlkScale<Q4Type>(b);
                const __m128i bvi4_0 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));

                const float scale_v1 = MlasQ4BlkScale<Q4Type>(b + ldb);
                const __m128i bvi4_1 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb));

                const float scale_v2 = MlasQ4BlkScale<Q4Type>(b + ldb * 2);
                const __m128i bvi4_2 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 2));

                const float scale_v3 = MlasQ4BlkScale<Q4Type>(b + ldb * 3);
                const __m128i bvi4_3 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 3));

                // expand 4b into byte array
                __m256i bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
                __m256i bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
                __m256i bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
                __m256i bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
                bytes0 = _mm256_and_si256(lowMask, bytes0);
                bytes1 = _mm256_and_si256(lowMask, bytes1);
                bytes2 = _mm256_and_si256(lowMask, bytes2);
                bytes3 = _mm256_and_si256(lowMask, bytes3);

                // Subtract zero-point from the integers
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                    // Subtract zero-point from the integers
                    bytes0 = _mm256_sub_epi8(
                        bytes0, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b)));
                    bytes1 = _mm256_sub_epi8(
                        bytes1, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb)));
                    bytes2 = _mm256_sub_epi8(
                        bytes2,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 2)));
                    bytes3 = _mm256_sub_epi8(
                        bytes3,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 3)));
                } else {
                    // Subtract 8 from the integers
                    bytes0 = _mm256_sub_epi8(bytes0, _mm256_set1_epi8(8));
                    bytes1 = _mm256_sub_epi8(bytes1, _mm256_set1_epi8(8));
                    bytes2 = _mm256_sub_epi8(bytes2, _mm256_set1_epi8(8));
                    bytes3 = _mm256_sub_epi8(bytes3, _mm256_set1_epi8(8));
                }

                // Convert to 16-bit int
                const __m256i vx16_lo0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 0));
                const __m256i vx16_hi0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes0, 1));
                const __m256i vx16_lo1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 0));
                const __m256i vx16_hi1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes1, 1));
                const __m256i vx16_lo2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 0));
                const __m256i vx16_hi2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes2, 1));
                const __m256i vx16_lo3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 0));
                const __m256i vx16_hi3 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes3, 1));

                // Convert to 32-bit int -> float 32
                __m512 bvf_lo0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo0));
                __m512 bvf_hi0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi0));
                __m512 bvf_lo1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo1));
                __m512 bvf_hi1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi1));
                __m512 bvf_lo2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo2));
                __m512 bvf_hi2 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi2));
                __m512 bvf_lo3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo3));
                __m512 bvf_hi3 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi3));
                __m512 s = _mm512_set1_ps(scale_v0);
                bvf_lo0 = _mm512_mul_ps(bvf_lo0, s);
                bvf_hi0 = _mm512_mul_ps(bvf_hi0, s);
                s = _mm512_set1_ps(scale_v1);
                bvf_lo1 = _mm512_mul_ps(bvf_lo1, s);
                bvf_hi1 = _mm512_mul_ps(bvf_hi1, s);
                s = _mm512_set1_ps(scale_v2);
                bvf_lo2 = _mm512_mul_ps(bvf_lo2, s);
                bvf_hi2 = _mm512_mul_ps(bvf_hi2, s);
                s = _mm512_set1_ps(scale_v3);
                bvf_lo3 = _mm512_mul_ps(bvf_lo3, s);
                bvf_hi3 = _mm512_mul_ps(bvf_hi3, s);

                acc_lo0 = _mm512_fmadd_ps(bvf_lo0, av_lo, acc_lo0);
                acc_lo0 = _mm512_fmadd_ps(bvf_hi0, av_hi, acc_lo0);
                acc_lo1 = _mm512_fmadd_ps(bvf_lo1, av_lo, acc_lo1);
                acc_lo1 = _mm512_fmadd_ps(bvf_hi1, av_hi, acc_lo1);
                acc_lo2 = _mm512_fmadd_ps(bvf_lo2, av_lo, acc_lo2);
                acc_lo2 = _mm512_fmadd_ps(bvf_hi2, av_hi, acc_lo2);
                acc_lo3 = _mm512_fmadd_ps(bvf_lo3, av_lo, acc_lo3);
                acc_lo3 = _mm512_fmadd_ps(bvf_hi3, av_hi, acc_lo3);

                b += Q4Type::BlobSize;
            }

            __m128 acc_x = FoldAccumulators(acc_lo0, acc_lo1, acc_lo2, acc_lo3);
            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }
            _mm_store_ps(sum_ptr, acc_x);

            // move to next 4 columns
            b_col += 4 * ldb;
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m512 acc_lo[4]{};
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                size_t ck = std::min(CountK - k, (typename Q4Type::BlkLen));

                uint32_t mask = 0xffffffff >> ((typename Q4Type::BlkLen) - ck);
                __m512 av_lo = _mm512_maskz_loadu_ps(__mmask16(mask), A + k);

                mask = mask >> 16;
                __m512 av_hi = mask == 0 ? _mm512_setzero_ps()
                    : _mm512_maskz_loadu_ps(__mmask16(mask), A + k + 16);

                for (int64_t nn = 0; nn < nblk; nn++) {
                    const auto* bb = b + ldb * nn;
                    const float scale_v = MlasQ4BlkScale<Q4Type>(bb);

                    const __m128i bvi4 =
                        _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(bb));
                    __m256i bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
                    bytes = _mm256_and_si256(lowMask, bytes);

                    if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>)
                    {
                        // Subtract zero-point from the integers
                        const uint8_t zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(bb);
                        bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(zp));
                    }
                    else {
                        // Subtract 8 from the integers
                        bytes = _mm256_sub_epi8(bytes, _mm256_set1_epi8(8));
                    }

                    // Convert to 16-bit int
                    const __m256i vx16_lo =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 0));
                    const __m256i vx16_hi =
                        _mm256_cvtepi8_epi16(_mm256_extracti128_si256(bytes, 1));

                    // Convert to 32-bit int -> float 32
                    __m512 bvf_lo = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_lo));
                    __m512 bvf_hi = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(vx16_hi));
                    __m512 s = _mm512_set1_ps(scale_v);
                    bvf_lo = _mm512_mul_ps(bvf_lo, s);
                    bvf_hi = _mm512_mul_ps(bvf_hi, s);

                    acc_lo[nn] = _mm512_fmadd_ps(bvf_lo, av_lo, acc_lo[nn]);
                    acc_lo[nn] = _mm512_fmadd_ps(bvf_hi, av_hi, acc_lo[nn]);
                }
                b += (typename Q4Type::BlobSize);
            }

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = _mm512_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        A += lda;
    }
    return CountM;
}

template<typename Q4TYPE, typename KERNEL>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
);

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK1,MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK1>(A, PackedB, C, CountM, CountN, CountK, lda,
                                                     ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ4GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    const float* A,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ4GemmKernelAvx512f<MLAS_Q4TYPE_BLK0>(A, PackedB, C, CountM, CountN, CountK, lda,
                                                     ldb, ldc, Bias);
}


template <typename Q4TYPE, typename KERNEL>
void MLASCALL
MlasQ4GemmOperation(
    const size_t K,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t lda = DataParams->lda;
    const size_t ldc = DataParams->ldc;

    const size_t k_blks = MlasDivRoundup(K, (typename Q4TYPE::BlkLen));
    const size_t ldb = k_blks * (typename Q4TYPE::BlobSize);
    const float* A = DataParams->A + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)DataParams->B;
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;
    const float* Bias = DataParams->Bias;

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        CountN = std::min(RangeCountN - n, (size_t)128);

        //
        // Step through each slice of matrix A along the M dimension.
        //
        const float* bias = (Bias == nullptr) ? nullptr : Bias + RangeStartN + n;
        const uint8_t* b_col = PackedB + (RangeStartN + n) * ldb;
        float* c_blk = C + n;
        const float* a_row = A;

        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
            auto RowsHandled = MlasQ4GemmKernel<Q4TYPE, KERNEL>(a_row, b_col, c_blk, RowsRemaining,
                                                                CountN, K, lda, ldb, ldc, bias);

            if (DataParams->OutputProcessor != nullptr) {
                DataParams->OutputProcessor->Process(
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                    RowsHandled, CountN, DataParams->ldc);
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}


////////////////////////////////////////////////////////////
//  Block int8 quantization, currently we only
//  implement symmetric quant, with no zero-point

template<typename QType>
MLAS_FORCEINLINE
void
MlasQ80BlkQuantRow(const float* A, void* Qblob, size_t size)
{
    static_assert((typename QType::BlkLen) % 16 == 0);
    const __m512 signBit = _mm512_set1_ps(-0.0f);
    int8_t* blob = reinterpret_cast<int8_t*>(Qblob);
    for (size_t k = 0; k < size; k += (typename QType::BlkLen)) {
        const size_t step = std::min((typename QType::BlkLen), size - k);

        __m512 maxAbs = _mm512_setzero();
        for (size_t kk = 0; kk < step; kk += 16) {
            const size_t klen = std::min(size_t(16), step - kk);
 
            uint32_t mask = 0xffff >> (16 - klen);
            __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);

            // Compute max(abs(e)) for the block
            maxAbs = _mm512_max_ps(maxAbs, _mm512_andnot_ps(signBit, v0));
        }

        __m256 max8 =
            _mm256_max_ps(_mm512_extractf32x8_ps(maxAbs, 1), _mm512_extractf32x8_ps(maxAbs, 0));
        __m128 max4 = _mm_max_ps(_mm256_extractf128_ps(max8, 1), _mm256_castps256_ps128(max8));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        const float maxScalar = _mm_cvtss_f32(max4);

        // Quantize these floats
        const float scale = maxScalar / 127.f;
        *reinterpret_cast<float*>(blob) = scale;
        blob += sizeof(float);

        const float inverse_scale = (maxScalar != 0.0f) ? 127.f / maxScalar : 0.0f;
        const __m512 mul = _mm512_set1_ps(inverse_scale);
        __m128i* dst = reinterpret_cast<__m128i*>(blob);

        for (size_t kk = 0; kk < step; kk += 16) {
            const size_t klen = std::min(size_t(16), step - kk);

            uint32_t mask = 0xffff >> (16 - klen);
            __m512 v0 = _mm512_maskz_loadu_ps(__mmask16(mask), A + k + kk);
            v0 = _mm512_mul_ps(v0, mul);

            // Round to nearest integer
            v0 = _mm512_roundscale_ps(v0, _MM_ROUND_NEAREST);

            // Convert floats to integers
            __m512i i0 = _mm512_cvtps_epi32(v0);

            // Convert int32 to int8
            _mm_storeu_si128(dst++, _mm512_cvtepi32_epi8(i0));
        }
        if (step < (typename QType::BlkLen)) {
            memset(blob + step, 0, (typename QType::BlkLen) - step);
        }
        blob += (typename QType::BlkLen);
    }
}

/**
 * @brief Compute the size of a quantized block, one byte per value + fp32 scale
 * @tparam QType 
 * @return 
*/
template<typename QType>
constexpr size_t
Q8BlobUnitSize()
{
    return (QType::BlkLen + sizeof(float));
}

template<typename QType>
MLAS_FORCEINLINE
size_t
MlasQ80BlkQuantSizeImpl(size_t M, size_t K)
{
    const size_t KBlocks = MlasDivRoundup(K, QType::BlkLen);

    const size_t NumBlocks = M * KBlocks;

    return NumBlocks * Q8BlobUnitSize<QType>();
}

size_t
MLASCALL
MlasQ80BlkQuantSize(MLAS_BLK_QUANT_TYPE QType, size_t M, size_t K)
{
    switch (QType) {
        case BlkQ4Zp8:
            return MlasQ80BlkQuantSizeImpl<MLAS_Q4TYPE_BLK1>(M, K);
        case BlkQ4Sym64:
            return MlasQ80BlkQuantSizeImpl<MLAS_Q4TYPE_BLK2>(M, K);
        default:
            return MlasQ80BlkQuantSizeImpl<MLAS_Q4TYPE_BLK0>(M, K);
    }
}

template<typename QType>
MLAS_FORCEINLINE
void
Q80BlkQuant(void* Qblob, const float* A, size_t M, size_t K, size_t lda, MLAS_THREADPOOL* ThreadPool)
{
    const size_t parts = (size_t)ceil(double(M) * K / (16.0 * 1024));
    const size_t TargetThreadCnt =
        std::max(std::min(parts, (size_t)MlasGetMaximumThreadCount(ThreadPool)), (size_t)1);
    const size_t linesize = MlasQ80BlkQuantSizeImpl<QType>(1, K);

    size_t M_stride = MlasDivRoundup(M, TargetThreadCnt);
    size_t threads = MlasDivRoundup(M, M_stride);
    MlasTrySimpleParallel(ThreadPool, threads, [&](ptrdiff_t tid) {
        const size_t m = tid * M_stride;
        const float* src = A + lda * m;
        uint8_t* dst = reinterpret_cast<uint8_t*>(Qblob) + m * linesize;
        for (size_t i = 0; i < std::min(M_stride, M-m); i++) {
            MlasQ80BlkQuantRow<QType>(src, dst, K);
            src += lda;
            dst += linesize;
        }
    });
}

void
MLASCALL
MlasQ80BlkQuant(
    MLAS_BLK_QUANT_TYPE QType,
    void* Qblob,
    const float* A,
    size_t M,
    size_t K,
    size_t lda,
    MLAS_THREADPOOL* ThreadPool
    )
{
    switch (QType) {
        case BlkQ4Zp8:
            return Q80BlkQuant<MLAS_Q4TYPE_BLK1>(Qblob, A, M, K, lda, ThreadPool);
        case BlkQ4Sym64:
            return Q80BlkQuant<MLAS_Q4TYPE_BLK2>(Qblob, A, M, K, lda, ThreadPool);
        default:
            return Q80BlkQuant<MLAS_Q4TYPE_BLK0>(Qblob, A, M, K, lda, ThreadPool);
    }
}

static
MLAS_FORCEINLINE
__m128
FoldAccumulators(
    const __m256& acc0,
    const __m256& acc1,
    const __m256& acc2,
    const __m256& acc3
    )
{
    __m256 acc_lo01 = _mm256_unpacklo_ps(acc0, acc1);
    __m256 acc_hi01 = _mm256_unpackhi_ps(acc0, acc1);
    __m256 acc_lo23 = _mm256_unpacklo_ps(acc2, acc3);
    __m256 acc_hi23 = _mm256_unpackhi_ps(acc2, acc3);

    __m256 acc_lo0123 = _mm256_castpd_ps(
        _mm256_unpacklo_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23)));
    __m256 acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpackhi_pd(_mm256_castps_pd(acc_lo01), _mm256_castps_pd(acc_lo23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpacklo_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);
    acc_hi0123 = _mm256_castpd_ps(
        _mm256_unpackhi_pd(_mm256_castps_pd(acc_hi01), _mm256_castps_pd(acc_hi23)));
    acc_lo0123 = _mm256_add_ps(acc_lo0123, acc_hi0123);

    return _mm_add_ps(_mm256_extractf32x4_ps(acc_lo0123, 0), _mm256_extractf32x4_ps(acc_lo0123, 1));
}

static inline float
mm256_reduce_add_ps(__m256 x)
{
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}


template<typename Q4Type>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernelAvx512f(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i lowMask = _mm256_set1_epi8(0xF);

    for (size_t m = 0; m < CountM; m++) {
        const uint8_t* b_col = PackedB;
        auto* sum_ptr = C;
        auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN) - 4;
        while (nblk >= 0) {
            __m256 acc_r0c0 = _mm256_setzero_ps();
            __m256 acc_r0c1 = _mm256_setzero_ps();
            __m256 acc_r0c2 = _mm256_setzero_ps();
            __m256 acc_r0c3 = _mm256_setzero_ps();
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                const float a_scale = *reinterpret_cast<const float*>(ablob);
                ablob += sizeof(float);
                const __m256i a_bytes0 = _mm256_loadu_si256((const __m256i*)ablob);
                ablob += 32;

                // Load 4 B column vectors (quantized to int4 blobs)
                const float scale_v0 = MlasQ4BlkScale<Q4Type>(b) * a_scale;
                const __m128i bvi4_0 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));

                const float scale_v1 = MlasQ4BlkScale<Q4Type>(b + ldb) * a_scale;
                const __m128i bvi4_1 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb));

                const float scale_v2 = MlasQ4BlkScale<Q4Type>(b + ldb * 2) * a_scale;
                const __m128i bvi4_2 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 2));

                const float scale_v3 = MlasQ4BlkScale<Q4Type>(b + ldb * 3) * a_scale;
                const __m128i bvi4_3 =
                    _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b + ldb * 3));

                // expand 4b into byte array
                __m256i bytes0 = _mm256_set_m128i(_mm_srli_epi16(bvi4_0, 4), bvi4_0);
                __m256i bytes1 = _mm256_set_m128i(_mm_srli_epi16(bvi4_1, 4), bvi4_1);
                __m256i bytes2 = _mm256_set_m128i(_mm_srli_epi16(bvi4_2, 4), bvi4_2);
                __m256i bytes3 = _mm256_set_m128i(_mm_srli_epi16(bvi4_3, 4), bvi4_3);
                bytes0 = _mm256_and_si256(lowMask, bytes0);
                bytes1 = _mm256_and_si256(lowMask, bytes1);
                bytes2 = _mm256_and_si256(lowMask, bytes2);
                bytes3 = _mm256_and_si256(lowMask, bytes3);

                // Subtract zero-point from the integers
                if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                    bytes0 = _mm256_sub_epi8(
                        bytes0, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b)));
                    bytes1 = _mm256_sub_epi8(
                        bytes1, _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb)));
                    bytes2 = _mm256_sub_epi8(
                        bytes2,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 2)));
                    bytes3 = _mm256_sub_epi8(
                        bytes3,
                        _mm256_set1_epi8(MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b + ldb * 3)));
                } else {
                    const __m256i eight = _mm256_set1_epi8(8);
                    bytes0 = _mm256_sub_epi8(bytes0, eight);
                    bytes1 = _mm256_sub_epi8(bytes1, eight);
                    bytes2 = _mm256_sub_epi8(bytes2, eight);
                    bytes3 = _mm256_sub_epi8(bytes3, eight);
                }

                // to use vnni unsigned x signed int, negate all negative
                // b vals to make it all positive, and then also negate the
                // corresponding a vals to compensate
                const __m256i summed_pairs0 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes0, bytes0), _mm256_sign_epi8(a_bytes0, bytes0));
                const __m256i summed_pairs1 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes1, bytes1), _mm256_sign_epi8(a_bytes0, bytes1));
                const __m256i summed_pairs2 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes2, bytes2), _mm256_sign_epi8(a_bytes0, bytes2));
                const __m256i summed_pairs3 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes3, bytes3), _mm256_sign_epi8(a_bytes0, bytes3));

                const __m256 sums0 = _mm256_cvtepi32_ps(summed_pairs0);
                const __m256 sums1 = _mm256_cvtepi32_ps(summed_pairs1);
                const __m256 sums2 = _mm256_cvtepi32_ps(summed_pairs2);
                const __m256 sums3 = _mm256_cvtepi32_ps(summed_pairs3);
                acc_r0c0 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v0), sums0, acc_r0c0);
                acc_r0c1 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v1), sums1, acc_r0c1);
                acc_r0c2 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v2), sums2, acc_r0c2);
                acc_r0c3 = _mm256_fmadd_ps(_mm256_set1_ps(scale_v3), sums3, acc_r0c3);
                b += Q4Type::BlobSize;
            }

            __m128 acc_x = FoldAccumulators(acc_r0c0, acc_r0c1, acc_r0c2, acc_r0c3);
            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }
            _mm_store_ps(sum_ptr, acc_x);

            // move to next 4 columns
            b_col += 4 * ldb;
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m256 acc_lo[4]{};
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                const float a_scale = *reinterpret_cast<const float*>(ablob);
                ablob += sizeof(float);
                const __m256i a_bytes = _mm256_loadu_si256((const __m256i*)ablob);
                ablob += 32;

                for (int64_t nn = 0; nn < nblk; nn++) {
                    const auto* bb = b + ldb * nn;
                    const float scale_v = MlasQ4BlkScale<Q4Type>(bb) * a_scale;
                    const __m128i bvi4 =
                        _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(bb));
                    __m256i b_bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
                    b_bytes = _mm256_and_si256(lowMask, b_bytes);

                    if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>)
                    {
                        // Subtract zero-point from the integers
                        const uint8_t zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(bb);
                        b_bytes = _mm256_sub_epi8(b_bytes, _mm256_set1_epi8(zp));
                    }
                    else {
                        // Subtract 8 from the integers
                        b_bytes = _mm256_sub_epi8(b_bytes, _mm256_set1_epi8(8));
                    }

                    // to use vnni unsigned x signed int, negate all negative
                    // b vals to make it all positive, 
                    const __m256i ax = _mm256_sign_epi8(b_bytes, b_bytes);
                    // and then also negate the corresponding a vals to compensate
                    const __m256i sy = _mm256_sign_epi8(a_bytes, b_bytes);
                    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                    const __m256 sum = _mm256_cvtepi32_ps(summed_pairs);
                    acc_lo[nn] = _mm256_fmadd_ps(_mm256_set1_ps(scale_v), sum, acc_lo[nn]);
                }
                b += typename Q4Type::BlobSize;
            }

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = mm256_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        QuantA += lda;
    }
    return CountM;
}

template<typename Q4TYPE, typename KERNEL>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    );


template<>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel<MLAS_Q4TYPE_BLK1,MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ8Q4GemmKernelAvx512f<MLAS_Q4TYPE_BLK1>(QuantA, PackedB, C, CountM, CountN, CountK,
                                                       lda, ldb, ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
MlasQ8Q4GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    const int8_t* QuantA,
    const uint8_t* PackedB,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldb,
    size_t ldc,
    const float* Bias
    )
{
    return MlasQ8Q4GemmKernelAvx512f<MLAS_Q4TYPE_BLK0>(QuantA, PackedB, C, CountM, CountN, CountK,
                                                       lda, ldb, ldc, Bias);
}

template<typename Q4Type>
MLAS_FORCEINLINE
void
MlasQ8Q4DequantBAvx512f(
    int8_t* DequantB,
    const uint8_t* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb
    )
{
    const __m256i zero = _mm256_setzero_si256();
    const __m256i lowMask = _mm256_set1_epi8(0xF);

    const uint8_t* b_col = PackedB;

    for (size_t n = 0; n < CountN; n++) {
        const auto* b = b_col;

        for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
            const float scale = MlasQ4BlkScale<Q4Type>(b);
            *reinterpret_cast<float*>(DequantB) = scale;
            DequantB += sizeof(float);

            const __m128i bvi4 = _mm_loadu_si128((const __m128i*)MlasQ4BlkData<Q4Type>(b));
            __m256i b_bytes = _mm256_set_m128i(_mm_srli_epi16(bvi4, 4), bvi4);
            b_bytes = _mm256_and_si256(lowMask, b_bytes);

            // Subtract zero-point
            if constexpr (std::is_same_v<Q4Type, MLAS_Q4TYPE_BLK1>) {
                const uint8_t zp = MlasQ4BlkZeroPoint<MLAS_Q4TYPE_BLK1>(b);
                b_bytes = _mm256_sub_epi8(b_bytes, _mm256_set1_epi8(zp));
            } else {
                b_bytes = _mm256_sub_epi8(b_bytes, _mm256_set1_epi8(8));
            }

            _mm256_storeu_epi8(DequantB, b_bytes);
            DequantB += 32;
            b += typename Q4Type::BlobSize;
        }
        b_col += ldb;
    }
}

template<typename Q4Type, typename KERNEL>
MLAS_FORCEINLINE
void
MlasQ8Q4DequantB(
    int8_t* DequantB,
    const uint8_t* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb
    );

template<>
MLAS_FORCEINLINE
void
MlasQ8Q4DequantB<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    int8_t* DequantB,
    const uint8_t* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb
)
{
    MlasQ8Q4DequantBAvx512f<MLAS_Q4TYPE_BLK0>(DequantB, PackedB, CountN, CountK, ldb);
}

template<>
MLAS_FORCEINLINE
void
MlasQ8Q4DequantB<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    int8_t* DequantB,
    const uint8_t* PackedB,
    size_t CountN,
    size_t CountK,
    size_t ldb
)
{
    MlasQ8Q4DequantBAvx512f<MLAS_Q4TYPE_BLK1>(DequantB, PackedB, CountN, CountK, ldb);
}

template<typename Q4Type>
MLAS_FORCEINLINE
size_t
BlkQ8GemmKernelAvx512Vnni(
    const int8_t* QuantA,
    const int8_t* Q8B,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    const float* Bias
    )
{
    const __m256i zero = _mm256_setzero_si256();
    const size_t k_blks = MlasDivRoundup(CountK, typename Q4Type::BlkLen);
    const size_t ldb = k_blks * (sizeof(float) + typename Q4Type::BlkLen);

    int64_t rows = (int64_t)CountM;
    while (rows >= 4) {
        const int8_t* b_col = Q8B;
        auto* sum_ptr = C;
        auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN) - 4;
        while (nblk >= 0) {
            __m256 acc_r0c0 = _mm256_setzero_ps();
            __m256 acc_r0c1 = _mm256_setzero_ps();
            __m256 acc_r0c2 = _mm256_setzero_ps();
            __m256 acc_r0c3 = _mm256_setzero_ps();
            __m256 acc_r1c0 = _mm256_setzero_ps();
            __m256 acc_r1c1 = _mm256_setzero_ps();
            __m256 acc_r1c2 = _mm256_setzero_ps();
            __m256 acc_r1c3 = _mm256_setzero_ps();
            __m256 acc_r2c0 = _mm256_setzero_ps();
            __m256 acc_r2c1 = _mm256_setzero_ps();
            __m256 acc_r2c2 = _mm256_setzero_ps();
            __m256 acc_r2c3 = _mm256_setzero_ps();
            __m256 acc_r3c0 = _mm256_setzero_ps();
            __m256 acc_r3c1 = _mm256_setzero_ps();
            __m256 acc_r3c2 = _mm256_setzero_ps();
            __m256 acc_r3c3 = _mm256_setzero_ps();
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                const float a_scale0 = *reinterpret_cast<const float*>(ablob);
                const __m256i a_bytes0 = _mm256_loadu_si256((const __m256i*)(ablob + sizeof(float)));

                // Load 4 B column vectors (quantized to int4 blobs)
                const float b_scale0 = (*reinterpret_cast<const float*>(b));
                const float b_scale1 = (*reinterpret_cast<const float*>(b + ldb));
                const float b_scale2 = (*reinterpret_cast<const float*>(b + ldb * 2));
                const float b_scale3 = (*reinterpret_cast<const float*>(b + ldb * 3));
                b += sizeof(float);

                __m256i bytes0 = _mm256_loadu_epi8(b);
                __m256i bytes1 = _mm256_loadu_epi8(b + ldb);
                __m256i bytes2 = _mm256_loadu_epi8(b + ldb * 2);
                __m256i bytes3 = _mm256_loadu_epi8(b + ldb * 3);
                b += 32;

                const float a_scale1 = *reinterpret_cast<const float*>(ablob + lda);
                const __m256i a_bytes1 = _mm256_loadu_si256((const __m256i*)(ablob + lda + sizeof(float)));

                // to use vnni unsigned x signed int, negate all negative
                // b vals to make it all positive, and then also negate the
                // corresponding a vals to compensate
                const __m256i abs0 = _mm256_sign_epi8(bytes0, bytes0);
                const __m256i abs1 = _mm256_sign_epi8(bytes1, bytes1);
                const __m256i abs2 = _mm256_sign_epi8(bytes2, bytes2);
                const __m256i abs3 = _mm256_sign_epi8(bytes3, bytes3);
                __m256i summed_pairs0 = _mm256_dpbusd_epi32(
                    zero, abs0, _mm256_sign_epi8(a_bytes0, bytes0));
                __m256i summed_pairs1 = _mm256_dpbusd_epi32(
                    zero, abs1, _mm256_sign_epi8(a_bytes0, bytes1));
                __m256i summed_pairs2 = _mm256_dpbusd_epi32(
                    zero, abs2, _mm256_sign_epi8(a_bytes0, bytes2));
                __m256i summed_pairs3 = _mm256_dpbusd_epi32(
                    zero, abs3, _mm256_sign_epi8(a_bytes0, bytes3));

                __m256 sums0 = _mm256_cvtepi32_ps(summed_pairs0);
                __m256 sums1 = _mm256_cvtepi32_ps(summed_pairs1);
                __m256 sums2 = _mm256_cvtepi32_ps(summed_pairs2);
                __m256 sums3 = _mm256_cvtepi32_ps(summed_pairs3);
                acc_r0c0 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale0 * a_scale0), sums0, acc_r0c0);
                acc_r0c1 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale1 * a_scale0), sums1, acc_r0c1);
                acc_r0c2 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale2 * a_scale0), sums2, acc_r0c2);
                acc_r0c3 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale3 * a_scale0), sums3, acc_r0c3);

                const float a_scale2 = *reinterpret_cast<const float*>(ablob + lda * 2);
                const __m256i a_bytes2 = _mm256_loadu_si256((const __m256i*)(ablob + lda * 2 + sizeof(float)));

                // to use vnni unsigned x signed int, negate all negative
                // b vals to make it all positive, and then also negate the
                // corresponding a vals to compensate
                summed_pairs0 = _mm256_dpbusd_epi32(
                    zero, abs0, _mm256_sign_epi8(a_bytes1, bytes0));
                summed_pairs1 = _mm256_dpbusd_epi32(
                    zero, abs1, _mm256_sign_epi8(a_bytes1, bytes1));
                summed_pairs2 = _mm256_dpbusd_epi32(
                    zero, abs2, _mm256_sign_epi8(a_bytes1, bytes2));
                summed_pairs3 = _mm256_dpbusd_epi32(
                    zero, abs3, _mm256_sign_epi8(a_bytes1, bytes3));

                sums0 = _mm256_cvtepi32_ps(summed_pairs0);
                sums1 = _mm256_cvtepi32_ps(summed_pairs1);
                sums2 = _mm256_cvtepi32_ps(summed_pairs2);
                sums3 = _mm256_cvtepi32_ps(summed_pairs3);
                acc_r1c0 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale0 * a_scale1), sums0, acc_r1c0);
                acc_r1c1 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale1 * a_scale1), sums1, acc_r1c1);
                acc_r1c2 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale2 * a_scale1), sums2, acc_r1c2);
                acc_r1c3 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale3 * a_scale1), sums3, acc_r1c3);

                const float a_scale3 = *reinterpret_cast<const float*>(ablob + lda * 3);
                const __m256i a_bytes3 = _mm256_loadu_si256((const __m256i*)(ablob + lda * 3 + sizeof(float)));

                // to use vnni unsigned x signed int, negate all negative
                // b vals to make it all positive, and then also negate the
                // corresponding a vals to compensate
                summed_pairs0 = _mm256_dpbusd_epi32(
                    zero, abs0, _mm256_sign_epi8(a_bytes2, bytes0));
                summed_pairs1 = _mm256_dpbusd_epi32(
                    zero, abs1, _mm256_sign_epi8(a_bytes2, bytes1));
                summed_pairs2 = _mm256_dpbusd_epi32(
                    zero, abs2, _mm256_sign_epi8(a_bytes2, bytes2));
                summed_pairs3 = _mm256_dpbusd_epi32(
                    zero, abs3, _mm256_sign_epi8(a_bytes2, bytes3));

                sums0 = _mm256_cvtepi32_ps(summed_pairs0);
                sums1 = _mm256_cvtepi32_ps(summed_pairs1);
                sums2 = _mm256_cvtepi32_ps(summed_pairs2);
                sums3 = _mm256_cvtepi32_ps(summed_pairs3);
                acc_r2c0 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale0 * a_scale2), sums0, acc_r2c0);
                acc_r2c1 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale1 * a_scale2), sums1, acc_r2c1);
                acc_r2c2 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale2 * a_scale2), sums2, acc_r2c2);
                acc_r2c3 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale3 * a_scale2), sums3, acc_r2c3);

                // to use vnni unsigned x signed int, negate all negative
                // b vals to make it all positive, and then also negate the
                // corresponding a vals to compensate
                summed_pairs0 = _mm256_dpbusd_epi32(
                    zero, abs0, _mm256_sign_epi8(a_bytes3, bytes0));
                summed_pairs1 = _mm256_dpbusd_epi32(
                    zero, abs1, _mm256_sign_epi8(a_bytes3, bytes1));
                summed_pairs2 = _mm256_dpbusd_epi32(
                    zero, abs2, _mm256_sign_epi8(a_bytes3, bytes2));
                summed_pairs3 = _mm256_dpbusd_epi32(
                    zero, abs3, _mm256_sign_epi8(a_bytes3, bytes3));

                sums0 = _mm256_cvtepi32_ps(summed_pairs0);
                sums1 = _mm256_cvtepi32_ps(summed_pairs1);
                sums2 = _mm256_cvtepi32_ps(summed_pairs2);
                sums3 = _mm256_cvtepi32_ps(summed_pairs3);
                acc_r3c0 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale0 * a_scale3), sums0, acc_r3c0);
                acc_r3c1 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale1 * a_scale3), sums1, acc_r3c1);
                acc_r3c2 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale2 * a_scale3), sums2, acc_r3c2);
                acc_r3c3 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale3 * a_scale3), sums3, acc_r3c3);

                ablob += 32 + sizeof(float);
            }

            __m128 acc_x0 = FoldAccumulators(acc_r0c0, acc_r0c1, acc_r0c2, acc_r0c3);
            __m128 acc_x1 = FoldAccumulators(acc_r1c0, acc_r1c1, acc_r1c2, acc_r1c3);
            __m128 acc_x2 = FoldAccumulators(acc_r2c0, acc_r2c1, acc_r2c2, acc_r2c3);
            __m128 acc_x3 = FoldAccumulators(acc_r3c0, acc_r3c1, acc_r3c2, acc_r3c3);
            if (Bias != nullptr) {
                acc_x0 = _mm_add_ps(acc_x0, _mm_loadu_ps(bias_ptr));
                acc_x1 = _mm_add_ps(acc_x1, _mm_loadu_ps(bias_ptr));
                acc_x2 = _mm_add_ps(acc_x2, _mm_loadu_ps(bias_ptr));
                acc_x3 = _mm_add_ps(acc_x3, _mm_loadu_ps(bias_ptr));
            }
            _mm_store_ps(sum_ptr, acc_x0);
            _mm_store_ps(sum_ptr + ldc, acc_x1);
            _mm_store_ps(sum_ptr + ldc * 2, acc_x2);
            _mm_store_ps(sum_ptr + ldc * 3, acc_x3);

            // move to next 4 columns
            b_col += 4 * ldb;
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m256 acc_r0[4]{};
            __m256 acc_r1[4]{};
            __m256 acc_r2[4]{};
            __m256 acc_r3[4]{};
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                const float a_scale0 = *reinterpret_cast<const float*>(ablob);
                const __m256i a_bytes0 = _mm256_loadu_si256((const __m256i*)(ablob + sizeof(float)));
                const float a_scale1 = *reinterpret_cast<const float*>(ablob + lda);
                const __m256i a_bytes1 = _mm256_loadu_si256((const __m256i*)(ablob + lda + sizeof(float)));
                const float a_scale2 = *reinterpret_cast<const float*>(ablob + lda * 2);
                const __m256i a_bytes2 =
                    _mm256_loadu_si256((const __m256i*)(ablob + lda * 2 + sizeof(float)));
                const float a_scale3 = *reinterpret_cast<const float*>(ablob + lda * 3);
                const __m256i a_bytes3 =
                    _mm256_loadu_si256((const __m256i*)(ablob + lda * 3 + sizeof(float)));
                ablob += 32 + sizeof(float);

                for (int64_t nn = 0; nn < nblk; nn++) {
                    const auto* bb = b + ldb * nn;
                    const float b_scale = (*reinterpret_cast<const float*>(bb));
                    __m256i b_bytes = _mm256_loadu_epi8(bb + sizeof(float));

                    // to use vnni unsigned x signed int, negate all negative
                    // b vals to make it all positive, 
                    const __m256i ax = _mm256_sign_epi8(b_bytes, b_bytes);
                    // and then also negate the corresponding a vals to compensate
                    __m256i sy = _mm256_sign_epi8(a_bytes0, b_bytes);
                    __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                    __m256 sum = _mm256_cvtepi32_ps(summed_pairs);
                    acc_r0[nn] = _mm256_fmadd_ps(_mm256_set1_ps(a_scale0 * b_scale), sum, acc_r0[nn]);

                    sy = _mm256_sign_epi8(a_bytes1, b_bytes);
                    summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                    sum = _mm256_cvtepi32_ps(summed_pairs);
                    acc_r1[nn] =
                        _mm256_fmadd_ps(_mm256_set1_ps(a_scale1 * b_scale), sum, acc_r1[nn]);

                    sy = _mm256_sign_epi8(a_bytes2, b_bytes);
                    summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                    sum = _mm256_cvtepi32_ps(summed_pairs);
                    acc_r2[nn] =
                        _mm256_fmadd_ps(_mm256_set1_ps(a_scale2 * b_scale), sum, acc_r2[nn]);

                    sy = _mm256_sign_epi8(a_bytes3, b_bytes);
                    summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                    sum = _mm256_cvtepi32_ps(summed_pairs);
                    acc_r3[nn] =
                        _mm256_fmadd_ps(_mm256_set1_ps(a_scale3 * b_scale), sum, acc_r3[nn]);
                }
                b += sizeof(float) + typename Q4Type::BlkLen;
            }

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = mm256_reduce_add_ps(acc_r0[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
            sum_ptr += ldc;
            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = mm256_reduce_add_ps(acc_r1[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
            sum_ptr += ldc;
            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = mm256_reduce_add_ps(acc_r2[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
            sum_ptr += ldc;
            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = mm256_reduce_add_ps(acc_r3[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next 4 rows
        C += ldc * 4;
        QuantA += lda * 4;
        rows -= 4;
    }


    while (rows > 0) {
        const int8_t* b_col = Q8B;
        auto* sum_ptr = C;
        auto* bias_ptr = Bias;

        int64_t nblk = (int64_t)(CountN)-4;
        while (nblk >= 0) {
            __m256 acc_r0c0 = _mm256_setzero_ps();
            __m256 acc_r0c1 = _mm256_setzero_ps();
            __m256 acc_r0c2 = _mm256_setzero_ps();
            __m256 acc_r0c3 = _mm256_setzero_ps();
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                const float a_scale0 = *reinterpret_cast<const float*>(ablob);
                ablob += sizeof(float);
                const __m256i a_bytes0 = _mm256_loadu_si256((const __m256i*)ablob);
                ablob += 32;

                // Load 4 B column vectors (quantized to int4 blobs)
                const float b_scale0 = (*reinterpret_cast<const float*>(b));
                const float b_scale1 = (*reinterpret_cast<const float*>(b + ldb));
                const float b_scale2 = (*reinterpret_cast<const float*>(b + ldb * 2));
                const float b_scale3 = (*reinterpret_cast<const float*>(b + ldb * 3));
                b += sizeof(float);

                __m256i bytes0 = _mm256_loadu_epi8(b);
                __m256i bytes1 = _mm256_loadu_epi8(b + ldb);
                __m256i bytes2 = _mm256_loadu_epi8(b + ldb * 2);
                __m256i bytes3 = _mm256_loadu_epi8(b + ldb * 3);
                b += 32;

                // to use vnni unsigned x signed int, negate all negative
                // b vals to make it all positive, and then also negate the
                // corresponding a vals to compensate
                const __m256i summed_pairs0 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes0, bytes0), _mm256_sign_epi8(a_bytes0, bytes0));
                const __m256i summed_pairs1 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes1, bytes1), _mm256_sign_epi8(a_bytes0, bytes1));
                const __m256i summed_pairs2 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes2, bytes2), _mm256_sign_epi8(a_bytes0, bytes2));
                const __m256i summed_pairs3 = _mm256_dpbusd_epi32(
                    zero, _mm256_sign_epi8(bytes3, bytes3), _mm256_sign_epi8(a_bytes0, bytes3));

                const __m256 sums0 = _mm256_cvtepi32_ps(summed_pairs0);
                const __m256 sums1 = _mm256_cvtepi32_ps(summed_pairs1);
                const __m256 sums2 = _mm256_cvtepi32_ps(summed_pairs2);
                const __m256 sums3 = _mm256_cvtepi32_ps(summed_pairs3);
                acc_r0c0 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale0 * a_scale0), sums0, acc_r0c0);
                acc_r0c1 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale1 * a_scale0), sums1, acc_r0c1);
                acc_r0c2 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale2 * a_scale0), sums2, acc_r0c2);
                acc_r0c3 = _mm256_fmadd_ps(_mm256_set1_ps(b_scale3 * a_scale0), sums3, acc_r0c3);
            }

            __m128 acc_x = FoldAccumulators(acc_r0c0, acc_r0c1, acc_r0c2, acc_r0c3);
            if (Bias != nullptr) {
                acc_x = _mm_add_ps(acc_x, _mm_loadu_ps(bias_ptr));
            }
            _mm_store_ps(sum_ptr, acc_x);

            // move to next 4 columns
            b_col += 4 * ldb;
            sum_ptr += 4;
            bias_ptr += 4;
            nblk -= 4;
        }

        // left over columns less than 4 ?
        nblk += 4;
        if (nblk > 0) {
            __m256 acc_lo[4]{};
            const int8_t* ablob = QuantA;
            const auto* b = b_col;

            for (size_t k = 0; k < CountK; k += (typename Q4Type::BlkLen)) {
                const float a_scale = *reinterpret_cast<const float*>(ablob);
                ablob += sizeof(float);
                const __m256i a_bytes = _mm256_loadu_si256((const __m256i*)ablob);
                ablob += 32;

                for (int64_t nn = 0; nn < nblk; nn++) {
                    const auto* bb = b + ldb * nn;
                    const float scale_v = (*reinterpret_cast<const float*>(bb)) * a_scale;
                    __m256i b_bytes = _mm256_loadu_epi8(bb + sizeof(float));

                    // to use vnni unsigned x signed int, negate all negative
                    // b vals to make it all positive,
                    const __m256i ax = _mm256_sign_epi8(b_bytes, b_bytes);
                    // and then also negate the corresponding a vals to compensate
                    const __m256i sy = _mm256_sign_epi8(a_bytes, b_bytes);
                    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
                    const __m256 sum = _mm256_cvtepi32_ps(summed_pairs);
                    acc_lo[nn] = _mm256_fmadd_ps(_mm256_set1_ps(scale_v), sum, acc_lo[nn]);
                }
                b += sizeof(float) + typename Q4Type::BlkLen;
            }

            for (int64_t nn = 0; nn < nblk; nn++) {
                sum_ptr[nn] = mm256_reduce_add_ps(acc_lo[nn]);
                sum_ptr[nn] += Bias == nullptr ? 0.0f : bias_ptr[nn];
            }
        }

        // Prepare pointers for the next row
        C += ldc;
        QuantA += lda;
        rows--;
    }
    return CountM;
}

template<typename Q4Type, typename KERNEL>
MLAS_FORCEINLINE
size_t
BlkQ8GemmKernel(
    const int8_t* QuantA,
    const int8_t* Q8B,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    const float* Bias
    );

template<>
MLAS_FORCEINLINE
size_t
BlkQ8GemmKernel<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    const int8_t* QuantA,
    const int8_t* Q8B,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    const float* Bias
)
{
    return BlkQ8GemmKernelAvx512Vnni<MLAS_Q4TYPE_BLK0>(QuantA, Q8B, C, CountM, CountN, CountK, lda,
                                                       ldc, Bias);
}

template<>
MLAS_FORCEINLINE
size_t
BlkQ8GemmKernel<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>(
    const int8_t* QuantA,
    const int8_t* Q8B,
    float* C,
    size_t CountM,
    size_t CountN,
    size_t CountK,
    size_t lda,
    size_t ldc,
    const float* Bias
)
{
    return BlkQ8GemmKernelAvx512Vnni<MLAS_Q4TYPE_BLK1>(QuantA, Q8B, C, CountM, CountN, CountK, lda,
                                                       ldc, Bias);
}


template <typename Q4TYPE, typename KERNEL>
void MLASCALL
MlasQ8Q4GemmOperation(
    const size_t K,
    const MLAS_Q8Q4_GEMM_DATA_PARAMS* DataParams,
    const size_t RangeStartM,
    const size_t RangeCountM,
    const size_t RangeStartN,
    const size_t RangeCountN
)
{
    const size_t k_blks = MlasDivRoundup(K, (typename Q4TYPE::BlkLen));
    const size_t ldb = k_blks * (typename Q4TYPE::BlobSize);
    const size_t lda = k_blks * Q8BlobUnitSize<Q4TYPE>();
    const size_t ldc = DataParams->ldc;

    const int8_t* A = reinterpret_cast<const int8_t*>(DataParams->A) + RangeStartM * lda;
    const uint8_t* PackedB = (const uint8_t*)DataParams->B;
    float* C = DataParams->C + RangeStartM * ldc + RangeStartN;
    const float* Bias = DataParams->Bias;

    size_t bufsize = k_blks * (typename Q4TYPE::BlkLen) * 64 * sizeof(float);
    MlasThreadedBufAlloc(bufsize);
    auto* dequant_b = reinterpret_cast<int8_t*>(ThreadedBufHolder.get());

    //
    // Step through each slice of matrix B along the N dimension.
    //

    size_t CountN;
    for (size_t n = 0; n < RangeCountN; n += CountN) {
        CountN = std::min(RangeCountN - n, (size_t)64);

        //
        // Step through each slice of matrix A along the M dimension.
        //
        const float* bias = (Bias == nullptr) ? nullptr : Bias + RangeStartN + n;
        const uint8_t* b_col = PackedB + (RangeStartN + n) * ldb;
        float* c_blk = C + n;
        const int8_t* a_row = A;

        MlasQ8Q4DequantB<Q4TYPE, KERNEL>(dequant_b, b_col, CountN, K, ldb);
        size_t RowsRemaining = RangeCountM;
        while (RowsRemaining > 0) {
            auto RowsHandled = BlkQ8GemmKernel<Q4TYPE, KERNEL>(a_row, dequant_b, c_blk, RowsRemaining,
                                                               CountN, K, lda, ldc, bias);
            //MlasQ8Q4GemmKernel<Q4TYPE, KERNEL>(
            //    a_row, b_col, c_blk, RowsRemaining, CountN, K, lda, ldb, ldc, bias);

            if (DataParams->OutputProcessor != nullptr) {
                DataParams->OutputProcessor->Process(
                    DataParams->C, RangeStartM + RangeCountM - RowsRemaining, RangeStartN,
                    RowsHandled, CountN, DataParams->ldc);
            }

            c_blk += ldc * RowsHandled;
            a_row += lda * RowsHandled;
            RowsRemaining -= RowsHandled;
        }
    }
}


template<typename ParamBlockType>
MLAS_FORCEINLINE
void
MlasQ4GemmBatchDriver(
    MLAS_BLK_QUANT_TYPE QType,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const ParamBlockType* DataParams,
    MLAS_THREADPOOL* ThreadPool
    )
{
    //const MLAS_Q4GEMM_DISPATCH* dispatch = MlasQ4GemmGetDispatch();
    //MLAS_Q4GEMM_OPERATION* operation = dispatch->Operation;
    void (*operation)(const size_t, const ParamBlockType*, const size_t, const size_t, const size_t,
                      const size_t) = nullptr;

    if constexpr (std::is_same_v<ParamBlockType, MLAS_Q4_GEMM_DATA_PARAMS>)
    {
        operation = QType == BlkQ4Sym
                        ? MlasQ4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>
                        : MlasQ4GemmOperation<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>;
    }
    else {
        operation = QType == BlkQ4Sym
                        ? MlasQ8Q4GemmOperation<MLAS_Q4TYPE_BLK0, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>
                        : MlasQ8Q4GemmOperation<MLAS_Q4TYPE_BLK1, MLAS_FP_Q4_GEMM_KERNEL_DEFAULT>;
    }

    if (ThreadPool == nullptr) {
        for (size_t gemm_i = 0; gemm_i < BatchN; gemm_i++) {
            auto Data = &DataParams[gemm_i];
            operation(K, Data, 0, M, 0, N);
        }
        return;
    }

    //
    // Compute the number of target threads given the complexity of the SGEMM
    // operation. Small requests should run using the single threaded path.
    //

    const double Complexity = double(M) * double(N) * double(K) * double(BatchN);

    ptrdiff_t TargetThreadCount = ptrdiff_t(Complexity / double(MLAS_QGEMM_THREAD_COMPLEXITY)) + 1;

    ptrdiff_t MaximumThreadCount = MlasGetMaximumThreadCount(ThreadPool);

    if (TargetThreadCount >= MaximumThreadCount) {
        TargetThreadCount = MaximumThreadCount;
    }

    ptrdiff_t ThreadsPerGemm = TargetThreadCount / BatchN;
    if (ThreadsPerGemm < 1) {
        ThreadsPerGemm = 1;
    }

    const size_t StrideM = 1024;  // dispatch->StrideM;

    size_t nc = N;
    if (ThreadsPerGemm > 1) {
        // more than one thread per GEMM

        const size_t BlockedM = MlasDivRoundup(M, StrideM);
        const size_t max_nc = MlasDivRoundup(N * BlockedM, ThreadsPerGemm);
        if (max_nc < nc) {
            nc = std::min(nc, MlasDivRoundup(max_nc, MLAS_QGEMM_STRIDEN_THREAD_ALIGN) *
                                  MLAS_QGEMM_STRIDEN_THREAD_ALIGN);
        }
    }
    const size_t StrideN = nc;

    const size_t ThreadCountM = MlasDivRoundup(M, StrideM);
    const size_t ThreadCountN = MlasDivRoundup(N, StrideN);
    ThreadsPerGemm = ThreadCountM * ThreadCountN;

    MlasTrySimpleParallel(ThreadPool, ThreadsPerGemm * BatchN, [&](ptrdiff_t tid) {
        const auto gemm_i = tid / ThreadsPerGemm;
        const auto blk_i = tid % ThreadsPerGemm;
        auto Data = &DataParams[gemm_i];

        const ptrdiff_t ThreadIdN = blk_i / ThreadCountM;
        const ptrdiff_t ThreadIdM = blk_i % ThreadCountM;

        const size_t RangeStartM = ThreadIdM * StrideM;
        const size_t RangeCountM = std::min(M - RangeStartM, (size_t)StrideM);

        const size_t RangeStartN = ThreadIdN * StrideN;
        const size_t RangeCountN = std::min(N - RangeStartN, (size_t)StrideN);

        operation(K, Data, RangeStartM, RangeCountM, RangeStartN, RangeCountN);
    });
}


void
MLASCALL
MlasQ4GemmBatch(
    MLAS_BLK_QUANT_TYPE QType,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_Q4_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MlasQ4GemmBatchDriver(QType, M, N, K, BatchN, DataParams, ThreadPool);
}

void
MLASCALL
MlasQ8Q4GemmBatch(
    MLAS_BLK_QUANT_TYPE QType,
    const size_t M,
    const size_t N,
    const size_t K,
    const size_t BatchN,
    const MLAS_Q8Q4_GEMM_DATA_PARAMS* DataParams,
    MLAS_THREADPOOL* ThreadPool
    )
{
    MlasQ4GemmBatchDriver(QType, M, N, K, BatchN, DataParams, ThreadPool);
}
