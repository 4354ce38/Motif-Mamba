/******************************************************************************
 * Copyright (c) 2023, MotifMamba Team.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK
#include <ATen/cuda/Atomic.cuh>  // For atomicAdd on complex

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
    #include <cub/block/block_reduce.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "reverse_scan.cuh"
#include "static_switch.h"

template<typename scalar_t> __device__ __forceinline__ scalar_t conj(scalar_t x);
template<> __device__ __forceinline__ float conj<float>(float x) { return x; }
template<> __device__ __forceinline__ complex_t conj<complex_t>(complex_t x) { return std::conj(x); }

__device__ __forceinline__ void compute_m_n_scalar_bwd(
    const SSMParamsBase &params,
    int dim_id,
    int state_idx,
    float dt,
    float gamma_fallback,
    float &m_corr,
    float &n_corr
) {
    if (!params.has_pq || params.P_ptr == nullptr || params.Q_ptr == nullptr) {
        m_corr = 1.f;
        n_corr = 1.f;
        return;
    }
    const float *P = reinterpret_cast<const float *>(params.P_ptr)
        + dim_id * params.P_d_stride + state_idx * params.P_dstate_stride;
    const float *Q = reinterpret_cast<const float *>(params.Q_ptr)
        + dim_id * params.Q_d_stride + state_idx * params.Q_dstate_stride;
    float pq_diag = 0.f;
    #pragma unroll 1
    for (int k = 0; k < params.pq_rank; ++k) {
        pq_diag += P[k * params.P_r_stride] * Q[k * params.Q_r_stride];
    }
    float gamma = gamma_fallback;
    if (params.gamma_ptr != nullptr) {
        const float *gamma_ptr = reinterpret_cast<const float *>(params.gamma_ptr);
        gamma = gamma_ptr[dim_id * params.gamma_d_stride + state_idx * params.gamma_dstate_stride];
    }
    float m_exp = dt * pq_diag - 0.5f * dt * gamma * dt * pq_diag + 0.5f * pq_diag * dt * gamma;
    m_exp = fmaxf(-20.f, fminf(20.f, m_exp));
    m_corr = expf(m_exp);
    const float eps = 1e-6f;
    const float gamma_safe = fabsf(gamma) > eps ? gamma : (gamma >= 0.f ? eps : -eps);
    const float alpha = dt * pq_diag / gamma_safe;
    const float denom = 1.f - alpha;
    n_corr = 1.f - alpha / (fabsf(denom) > eps ? denom : (denom >= 0.f ? eps : -eps));
    if (!isfinite(m_corr)) { m_corr = 1.f; }
    if (!isfinite(n_corr)) { n_corr = 1.f; }
    n_corr = fmaxf(-8.f, fminf(8.f, n_corr));
}

constexpr int kMaxDensePQRankBwd = 8;
constexpr int kMaxDenseKBwd = 3 * kMaxDensePQRankBwd;

__device__ __forceinline__ void matmul_small_bwd(const float *A, const float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int t = 0; t < k; ++t) { acc += A[i * k + t] * B[t * n + j]; }
            C[i * n + j] = acc;
        }
    }
}

__device__ __forceinline__ void matvec_small_bwd(const float *A, const float *x, float *y, int m, int n) {
    for (int i = 0; i < m; ++i) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) { acc += A[i * n + j] * x[j]; }
        y[i] = acc;
    }
}

__device__ __forceinline__ bool invert_small_bwd(float *A, float *Ainv, int n) {
    for (int i = 0; i < n * n; ++i) { Ainv[i] = 0.f; }
    for (int i = 0; i < n; ++i) { Ainv[i * n + i] = 1.f; }
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        float max_abs = fabsf(A[i * n + i]);
        for (int r = i + 1; r < n; ++r) {
            float v = fabsf(A[r * n + i]);
            if (v > max_abs) { max_abs = v; pivot = r; }
        }
        if (max_abs < 1e-12f) { return false; }
        if (pivot != i) {
            for (int c = 0; c < n; ++c) {
                float tmp = A[i * n + c]; A[i * n + c] = A[pivot * n + c]; A[pivot * n + c] = tmp;
                float tmpi = Ainv[i * n + c]; Ainv[i * n + c] = Ainv[pivot * n + c]; Ainv[pivot * n + c] = tmpi;
            }
        }
        const float diag = A[i * n + i];
        const float inv_diag = 1.f / diag;
        for (int c = 0; c < n; ++c) {
            A[i * n + c] *= inv_diag;
            Ainv[i * n + c] *= inv_diag;
        }
        for (int r = 0; r < n; ++r) {
            if (r == i) { continue; }
            const float f = A[r * n + i];
            if (f == 0.f) { continue; }
            for (int c = 0; c < n; ++c) {
                A[r * n + c] -= f * A[i * n + c];
                Ainv[r * n + c] -= f * Ainv[i * n + c];
            }
        }
    }
    return true;
}

__device__ __forceinline__ void expm_small_bwd(float *A, float *Aexp, int n) {
    float norm1 = 0.f;
    for (int c = 0; c < n; ++c) {
        float col = 0.f;
        for (int r = 0; r < n; ++r) { col += fabsf(A[r * n + c]); }
        norm1 = fmaxf(norm1, col);
    }
    int s = 0;
    while (norm1 > 0.5f && s < 6) { norm1 *= 0.5f; ++s; }
    const float scale = ldexpf(1.f, -s);

    float As[kMaxDenseKBwd * kMaxDenseKBwd];
    float term[kMaxDenseKBwd * kMaxDenseKBwd];
    float tmp[kMaxDenseKBwd * kMaxDenseKBwd];
    for (int i = 0; i < n * n; ++i) {
        As[i] = A[i] * scale;
        Aexp[i] = 0.f;
        term[i] = 0.f;
    }
    for (int i = 0; i < n; ++i) {
        Aexp[i * n + i] = 1.f;
        term[i * n + i] = 1.f;
    }
    for (int k = 1; k <= 12; ++k) {
        matmul_small_bwd(term, As, tmp, n, n, n);
        const float invk = 1.f / float(k);
        for (int i = 0; i < n * n; ++i) {
            term[i] = tmp[i] * invk;
            Aexp[i] += term[i];
        }
    }
    for (int i = 0; i < s; ++i) {
        matmul_small_bwd(Aexp, Aexp, tmp, n, n, n);
        for (int j = 0; j < n * n; ++j) { Aexp[j] = tmp[j]; }
    }
}

template<typename input_t>
__global__ void selective_scan_bwd_dense_pq_kernel(SSMParamsBwd params) {
    if (threadIdx.x != 0) { return; }
    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / params.dim_ngroups_ratio;
    if (batch_id >= params.batch || dim_id >= params.dim) { return; }
    if (!params.has_pq || params.pq_rank <= 0 || params.pq_rank > kMaxDensePQRankBwd || params.dstate > MAX_DSTATE) { return; }
    if (params.z_ptr != nullptr) { return; }  // v1 limitation

    const input_t *u = reinterpret_cast<const input_t *>(params.u_ptr) + batch_id * params.u_batch_stride + dim_id * params.u_d_stride;
    const input_t *delta = reinterpret_cast<const input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride + dim_id * params.delta_d_stride;
    const input_t *dout = reinterpret_cast<const input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride + dim_id * params.dout_d_stride;
    input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride + dim_id * params.du_d_stride;
    input_t *ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride + dim_id * params.ddelta_d_stride;

    const float *A = reinterpret_cast<const float *>(params.A_ptr) + dim_id * params.A_d_stride;
    const float *B_const = params.is_variable_B ? nullptr : (reinterpret_cast<const float *>(params.B_ptr) + dim_id * params.B_d_stride);
    const float *C_const = params.is_variable_C ? nullptr : (reinterpret_cast<const float *>(params.C_ptr) + dim_id * params.C_d_stride);
    const input_t *B_var = params.is_variable_B ? (reinterpret_cast<const input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride) : nullptr;
    const input_t *C_var = params.is_variable_C ? (reinterpret_cast<const input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride) : nullptr;
    float *dA = reinterpret_cast<float *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    float *dB_const = params.is_variable_B ? nullptr : (reinterpret_cast<float *>(params.dB_ptr) + dim_id * params.dB_d_stride);
    float *dC_const = params.is_variable_C ? nullptr : (reinterpret_cast<float *>(params.dC_ptr) + dim_id * params.dC_d_stride);
    float *dB_var = params.is_variable_B ? (reinterpret_cast<float *>(params.dB_ptr) + batch_id * params.dB_batch_stride + group_id * params.dB_group_stride) : nullptr;
    float *dC_var = params.is_variable_C ? (reinterpret_cast<float *>(params.dC_ptr) + batch_id * params.dC_batch_stride + group_id * params.dC_group_stride) : nullptr;
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;

    const float *P = reinterpret_cast<const float *>(params.P_ptr) + dim_id * params.P_d_stride;
    const float *Q = reinterpret_cast<const float *>(params.Q_ptr) + dim_id * params.Q_d_stride;
    float *dP = params.dP_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dP_ptr) + dim_id * params.dP_d_stride;
    float *dQ = params.dQ_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dQ_ptr) + dim_id * params.dQ_d_stride;

    const int n = params.dstate;
    const int r = params.pq_rank;
    const float delta_bias = params.delta_bias_ptr == nullptr ? 0.f : reinterpret_cast<const float *>(params.delta_bias_ptr)[dim_id];
    const float D_val = params.D_ptr == nullptr ? 0.f : reinterpret_cast<const float *>(params.D_ptr)[dim_id];

    float dx_next[MAX_DSTATE];
    float dx_prev[MAX_DSTATE];
    float g_state[MAX_DSTATE];
    float x_prev[MAX_DSTATE];
    float x_cur[MAX_DSTATE];
    float q_vec[kMaxDensePQRankBwd];
    float gq[kMaxDensePQRankBwd];
    float aexp[MAX_DSTATE];
    float b_raw[MAX_DSTATE];
    float c_raw[MAX_DSTATE];

    for (int i = 0; i < n; ++i) { dx_next[i] = 0.f; }

    float dD_val = 0.f;
    float ddelta_bias_val = 0.f;

    for (int t = params.seqlen - 1; t >= 0; --t) {
        // Replay forward to get state_{t-1} and state_t.
        for (int i = 0; i < n; ++i) {
            x_prev[i] = 0.f;
            x_cur[i] = 0.f;
        }
        for (int s = 0; s <= t; ++s) {
            float dt_s = float(delta[s]) + delta_bias;
            if (params.delta_softplus) {
                dt_s = dt_s <= 20.f ? log1pf(expf(dt_s)) : dt_s;
            }
            const float u_s = float(u[s]);

            for (int i = 0; i < n; ++i) {
                aexp[i] = expf(dt_s * A[i * params.A_dstate_stride]);
                b_raw[i] = !params.is_variable_B
                    ? B_const[i * params.B_dstate_stride]
                    : float(B_var[i * params.B_dstate_stride + s]);
                c_raw[i] = !params.is_variable_C
                    ? C_const[i * params.C_dstate_stride]
                    : float(C_var[i * params.C_dstate_stride + s]);
            }

            for (int j = 0; j < r; ++j) {
                float acc = 0.f;
                for (int i = 0; i < n; ++i) {
                    const float q = Q[j * params.Q_r_stride + i * params.Q_dstate_stride];
                    acc += q * x_prev[i];
                }
                q_vec[j] = acc;
            }

            for (int i = 0; i < n; ++i) {
                float pq_term = 0.f;
                for (int j = 0; j < r; ++j) {
                    const float p = P[i * params.P_dstate_stride + j * params.P_r_stride];
                    pq_term += p * q_vec[j];
                }
                const float b_term = dt_s * u_s * b_raw[i];
                x_cur[i] = aexp[i] * x_prev[i] + dt_s * pq_term + b_term;
            }
            if (s < t) {
                for (int i = 0; i < n; ++i) { x_prev[i] = x_cur[i]; }
            }
        }

        const float dt_raw = float(delta[t]) + delta_bias;
        float dt = dt_raw;
        if (params.delta_softplus) {
            dt = dt <= 20.f ? log1pf(expf(dt)) : dt;
        }
        const float u_val = float(u[t]);
        const float dout_t = float(dout[t]);

        if (dD != nullptr) { dD_val += dout_t * u_val; }
        float du_t = D_val * dout_t;
        float ddt = 0.f;

        for (int i = 0; i < n; ++i) {
            if (!params.is_variable_C) {
                gpuAtomicAdd(dC_const + i * params.dC_dstate_stride, dout_t * x_cur[i]);
            } else {
                gpuAtomicAdd(dC_var + i * params.dC_dstate_stride + t, dout_t * x_cur[i]);
            }
            g_state[i] = dx_next[i] + dout_t * c_raw[i];
            dx_prev[i] = 0.f;
        }
        for (int j = 0; j < r; ++j) { gq[j] = 0.f; }

        for (int i = 0; i < n; ++i) {
            const float g = g_state[i];
            const float A_i = A[i * params.A_dstate_stride];

            gpuAtomicAdd(dA + i * params.dA_dstate_stride, g * x_prev[i] * dt * aexp[i]);
            if (!params.is_variable_B) {
                gpuAtomicAdd(dB_const + i * params.dB_dstate_stride, g * dt * u_val);
            } else {
                gpuAtomicAdd(dB_var + i * params.dB_dstate_stride + t, g * dt * u_val);
            }

            du_t += g * dt * b_raw[i];
            ddt += g * (A_i * aexp[i] * x_prev[i] + u_val * b_raw[i]);
            dx_prev[i] += g * aexp[i];

            for (int j = 0; j < r; ++j) {
                const float p = P[i * params.P_dstate_stride + j * params.P_r_stride];
                if (dP != nullptr) {
                    gpuAtomicAdd(dP + i * params.dP_dstate_stride + j * params.dP_r_stride, g * dt * q_vec[j]);
                }
                gq[j] += g * dt * p;
            }
            {
                float pq_term_i = 0.f;
                for (int j = 0; j < r; ++j) {
                    const float p = P[i * params.P_dstate_stride + j * params.P_r_stride];
                    pq_term_i += p * q_vec[j];
                }
                ddt += g * pq_term_i;
            }
        }

        for (int i = 0; i < n; ++i) {
            float acc = 0.f;
            for (int j = 0; j < r; ++j) {
                const float q = Q[j * params.Q_r_stride + i * params.Q_dstate_stride];
                acc += gq[j] * q;
            }
            dx_prev[i] += acc;
        }

        if (dQ != nullptr) {
            for (int j = 0; j < r; ++j) {
                for (int i = 0; i < n; ++i) {
                    gpuAtomicAdd(dQ + j * params.dQ_r_stride + i * params.dQ_dstate_stride, gq[j] * x_prev[i]);
                }
            }
        }

        float ddelta_t = ddt;
        if (params.delta_softplus) {
            const float neg = expf(-dt_raw);
            ddelta_t = dt_raw <= 20.f ? ddelta_t / (1.f + neg) : ddelta_t;
        }
        ddelta[t] = input_t(ddelta_t);
        du[t] = input_t(du_t);
        ddelta_bias_val += ddelta_t;

        for (int i = 0; i < n; ++i) { dx_next[i] = dx_prev[i]; }
    }

    if (dD != nullptr) { gpuAtomicAdd(dD, dD_val); }
    if (ddelta_bias != nullptr) { gpuAtomicAdd(ddelta_bias, ddelta_bias_val); }
}

template<int kNThreads_, int kNItems_, bool kIsEvenLen_, bool kIsVariableB_, bool kIsVariableC_,
         bool kDeltaSoftplus_, bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_bwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kDeltaSoftplus = kDeltaSoftplus_;
    static constexpr bool kHasZ = kHasZ_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads with float improves occupancy.
    // For complex this would lead to massive register spilling, so we keep it at 2.
    static constexpr int kMinBlocks = kNThreads == 128 && !kIsComplex ? 3 : 2;
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockReverseScanT = BlockReverseScan<scan_t, kNThreads>;
    using BlockReduceT = cub::BlockReduce<scan_t, kNThreads>;
    using BlockReduceFloatT = cub::BlockReduce<float, kNThreads>;
    using BlockReduceComplexT = cub::BlockReduce<complex_t, kNThreads>;
    using BlockExchangeT = cub::BlockExchange<float, kNThreads, !kIsComplex ? kNItems : kNItems * 2>;

    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                    sizeof(typename BlockLoadVecT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                    (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                    sizeof(typename BlockStoreT::TempStorage),
                                                    sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemExchangeSize = (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockExchangeT::TempStorage);
    static constexpr int kSmemReduceSize = sizeof(typename BlockReduceT::TempStorage);
    static constexpr int kSmemSize = kSmemIOSize + kSmemExchangeSize + kSmemReduceSize + sizeof(typename BlockScanT::TempStorage) + sizeof(typename BlockReverseScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_bwd_kernel(SSMParamsBwd params) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kDeltaSoftplus = Ktraits::kDeltaSoftplus;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    using input_t = typename Ktraits::input_t;
    using weight_t = typename Ktraits::weight_t;
    using scan_t = typename Ktraits::scan_t;

    // Shared memory.
    extern __shared__ char smem_[];
    // cast to lvalue reference of expected type
    // char *smem_loadstorescan = smem_ + 2 * MAX_DSTATE * sizeof(weight_t);
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_ + 2 * MAX_DSTATE * sizeof(weight_t));
    // auto& smem_load = reinterpret_cast<typename BlockLoadT::TempStorage&>(smem_loadstorescan);
    auto& smem_load = reinterpret_cast<typename Ktraits::BlockLoadT::TempStorage&>(smem_);
    auto& smem_load_weight = reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage&>(smem_);
    auto& smem_load_weight1 = *reinterpret_cast<typename Ktraits::BlockLoadWeightT::TempStorage*>(smem_ + sizeof(typename Ktraits::BlockLoadWeightT::TempStorage));
    auto& smem_store = reinterpret_cast<typename Ktraits::BlockStoreT::TempStorage&>(smem_);
    auto& smem_exchange = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    auto& smem_exchange1 = *reinterpret_cast<typename Ktraits::BlockExchangeT::TempStorage*>(smem_ + Ktraits::kSmemIOSize + sizeof(typename Ktraits::BlockExchangeT::TempStorage));
    auto& smem_reduce = *reinterpret_cast<typename Ktraits::BlockReduceT::TempStorage*>(reinterpret_cast<char *>(&smem_exchange) + Ktraits::kSmemExchangeSize);
    auto& smem_reduce_float = *reinterpret_cast<typename Ktraits::BlockReduceFloatT::TempStorage*>(&smem_reduce);
    auto& smem_reduce_complex = *reinterpret_cast<typename Ktraits::BlockReduceComplexT::TempStorage*>(&smem_reduce);
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(reinterpret_cast<char *>(&smem_reduce) + Ktraits::kSmemReduceSize);
    auto& smem_reverse_scan = *reinterpret_cast<typename Ktraits::BlockReverseScanT::TempStorage*>(reinterpret_cast<char *>(&smem_scan) + sizeof(typename Ktraits::BlockScanT::TempStorage));
    weight_t *smem_delta_a = reinterpret_cast<weight_t *>(smem_ + Ktraits::kSmemSize);
    scan_t *smem_running_postfix = reinterpret_cast<scan_t *>(smem_delta_a + 2 * MAX_DSTATE + kNThreads);
    weight_t *smem_da = reinterpret_cast<weight_t *>(smem_running_postfix + MAX_DSTATE);
    weight_t *smem_dbc = reinterpret_cast<weight_t *>(smem_da + MAX_DSTATE);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * params.delta_d_stride;
    input_t *dout = reinterpret_cast<input_t *>(params.dout_ptr) + batch_id * params.dout_batch_stride
        + dim_id * params.dout_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    weight_t *dA = reinterpret_cast<weight_t *>(params.dA_ptr) + dim_id * params.dA_d_stride;
    weight_t *dB = reinterpret_cast<weight_t *>(params.dB_ptr)
        + (!kIsVariableB ? dim_id * params.dB_d_stride : batch_id * (!kIsComplex ? params.dB_batch_stride : params.dB_batch_stride / 2) + group_id * params.dB_group_stride);
    weight_t *dC = reinterpret_cast<weight_t *>(params.dC_ptr)
        + (!kIsVariableC ? dim_id * params.dC_d_stride : batch_id * (!kIsComplex ? params.dC_batch_stride : params.dC_batch_stride / 2) + group_id * params.dC_group_stride);
    const float *P_base = params.P_ptr == nullptr ? nullptr : reinterpret_cast<const float *>(params.P_ptr);
    const float *Q_base = params.Q_ptr == nullptr ? nullptr : reinterpret_cast<const float *>(params.Q_ptr);
    float *dP_base = params.dP_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dP_ptr);
    float *dQ_base = params.dQ_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dQ_ptr);
    float *dD = params.dD_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.dD_ptr) + dim_id;
    float D_val = params.D_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.D_ptr)[dim_id];
    float *ddelta_bias = params.ddelta_bias_ptr == nullptr ? nullptr : reinterpret_cast<float *>(params.ddelta_bias_ptr) + dim_id;
    float delta_bias = params.delta_bias_ptr == nullptr ? 0 : reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id];
    scan_t *x = params.x_ptr == nullptr
        ? nullptr
        : reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id) * (params.n_chunks) * params.dstate;
    float dD_val = 0;
    float ddelta_bias_val = 0;

    constexpr int kChunkSize = kNThreads * kNItems;
    u += (params.n_chunks - 1) * kChunkSize;
    delta += (params.n_chunks - 1) * kChunkSize;
    dout += (params.n_chunks - 1) * kChunkSize;
    Bvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    Cvar += (params.n_chunks - 1) * kChunkSize * (!kIsComplex ? 1 : 2);
    for (int chunk = params.n_chunks - 1; chunk >= 0; --chunk) {
        input_t u_vals[kNItems];
        input_t delta_vals_load[kNItems];
        input_t dout_vals_load[kNItems];
        __syncthreads();
        load_input<Ktraits>(u, u_vals, smem_load, params.seqlen - chunk * kChunkSize);
        u -= kChunkSize;
        __syncthreads();
        load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        // Will reload delta at the same location if kDeltaSoftplus
        if constexpr (!kDeltaSoftplus) { delta -= kChunkSize; }
        __syncthreads();
        load_input<Ktraits>(dout, dout_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
        dout -= kChunkSize;

        float dout_vals[kNItems], delta_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) {
            dout_vals[i] = float(dout_vals_load[i]);
            delta_vals[i] = float(delta_vals_load[i]) + delta_bias;
            if constexpr (kDeltaSoftplus) {
                delta_vals[i] = delta_vals[i] <= 20.f ? log1pf(expf(delta_vals[i])) : delta_vals[i];
            }
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * params.z_d_stride + chunk * kChunkSize;
            input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
                + dim_id * params.out_d_stride + chunk * kChunkSize;
            input_t *dz = reinterpret_cast<input_t *>(params.dz_ptr) + batch_id * params.dz_batch_stride
                + dim_id * params.dz_d_stride + chunk * kChunkSize;
            input_t z_vals[kNItems], out_vals[kNItems];
            __syncthreads();
            load_input<Ktraits>(z, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
            __syncthreads();
            load_input<Ktraits>(out, out_vals, smem_load, params.seqlen - chunk * kChunkSize);
            float dz_vals[kNItems], z_silu_vals[kNItems];
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float z_val = z_vals[i];
                float z_sigmoid_val = 1.0f / (1.0f + expf(-z_val));
                z_silu_vals[i] = z_val * z_sigmoid_val;
                dz_vals[i] = dout_vals[i] * float(out_vals[i]) * z_sigmoid_val
                             * (1.0f + z_val * (1.0f - z_sigmoid_val));
                dout_vals[i] *= z_silu_vals[i];
            }
            __syncthreads();
            store_output<Ktraits>(dz, dz_vals, smem_store, params.seqlen - chunk * kChunkSize);
            if (params.out_z_ptr != nullptr) {  // Recompute and store out_z
                float out_z_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) { out_z_vals[i] = float(out_vals[i]) * z_silu_vals[i]; }
                // if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
                    // printf("out_val=%f, z_silu_val = %f, out_z_val = %f\n", float(out_vals[0]), z_silu_vals[0], out_z_vals[0]);
                // }
                input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                    + dim_id * params.out_z_d_stride + chunk * kChunkSize;
                __syncthreads();
                store_output<Ktraits>(out_z, out_z_vals, smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        float du_vals[kNItems];
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { du_vals[i] = D_val * dout_vals[i]; }
        #pragma unroll
        for (int i = 0; i < kNItems; ++i) { dD_val += dout_vals[i] * float(u_vals[i]); }

        float ddelta_vals[kNItems] = {0};
        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            const weight_t A_val = A[state_idx * params.A_dstate_stride];
            // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
            weight_t A_scaled;
            constexpr float kLog2e = M_LOG2E;
            if constexpr (!kIsComplex) {
                A_scaled = A_val * kLog2e;
            } else {
                A_scaled = complex_t(A_val.real_ * kLog2e, A_val.imag_);
            }
            weight_t B_val, C_val;
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (!kIsVariableB) {
                B_val = B[state_idx * params.B_dstate_stride];
            } else {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
            }
            if constexpr (!kIsVariableC) {
                C_val = C[state_idx * params.C_dstate_stride];
            } else {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
            }
            // const weight_t A_val = smem_a[state_idx];
            scan_t thread_data[kNItems], thread_reverse_data[kNItems];
            if constexpr (!kIsComplex) {
                const bool need_dpq = params.has_pq && P_base != nullptr && Q_base != nullptr && dP_base != nullptr && dQ_base != nullptr;
                const float *P_state = need_dpq
                    ? (P_base + dim_id * params.P_d_stride + state_idx * params.P_dstate_stride)
                    : nullptr;
                const float *Q_state = need_dpq
                    ? (Q_base + dim_id * params.Q_d_stride + state_idx * params.Q_dstate_stride)
                    : nullptr;
                float pq_diag_state = 0.f;
                if (need_dpq) {
                    #pragma unroll 1
                    for (int k = 0; k < params.pq_rank; ++k) {
                        pq_diag_state += P_state[k * params.P_r_stride] * Q_state[k * params.Q_r_stride];
                    }
                }
                float gamma_state = float(A_val);
                if (params.gamma_ptr != nullptr) {
                    const float *gamma_ptr = reinterpret_cast<const float *>(params.gamma_ptr);
                    gamma_state = gamma_ptr[dim_id * params.gamma_d_stride + state_idx * params.gamma_dstate_stride];
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float m_corr = 1.f, n_corr = 1.f;
                    compute_m_n_scalar_bwd(params, dim_id, state_idx, delta_vals[i], float(A_val), m_corr, n_corr);
                    float log2_a = delta_vals[i] * A_scaled;
                    log2_a = fmaxf(-80.f, fminf(80.f, log2_a));
                    const float delta_a_exp = exp2f(log2_a) * m_corr;
                    thread_data[i] = make_float2(delta_a_exp, n_corr * (!kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]));
                    if (i == 0) {
                        smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = delta_a_exp;
                    } else {
                        thread_reverse_data[i - 1].x = delta_a_exp;
                    }
                    thread_reverse_data[i].y = dout_vals[i] *
                        (!kIsVariableC
                         ? (!kIsVariableB ? B_val * C_val : C_val)
                         : (!kIsVariableB ? B_val * C_vals[i] : C_vals[i]));
                }
                __syncthreads();
                thread_reverse_data[kNItems - 1].x = threadIdx.x == kNThreads - 1
                    ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                    : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];
                // Initialize running total
                scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float2(1.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                    thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                );
                if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }
                weight_t dA_val = 0, dBC_val = 0;
                weight_t dB_vals[kNItems], dC_vals[kNItems];
                float dPQ_val = 0.f;
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float m_corr = 1.f, n_corr = 1.f;
                    compute_m_n_scalar_bwd(params, dim_id, state_idx, delta_vals[i], float(A_val), m_corr, n_corr);
                    const float dx = thread_reverse_data[i].y;
                    const float b0 = !kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i];
                    const float ddelta_u = !kIsVariableB ? dx * n_corr : dx * B_vals[i] * n_corr;
                    du_vals[i] += ddelta_u * delta_vals[i];
                    const float a = thread_data[i].y - n_corr * b0;
                    ddelta_vals[i] += ddelta_u * float(u_vals[i]) + dx * (A_val * m_corr) * a;
                    dA_val += dx * delta_vals[i] * a * m_corr;
                    if (need_dpq) {
                        const float dt = delta_vals[i];
                        const float dsdpq = dt - 0.5f * dt * gamma_state * dt + 0.5f * dt * gamma_state;
                        const float m_exp_raw = dt * pq_diag_state - 0.5f * dt * gamma_state * dt * pq_diag_state + 0.5f * pq_diag_state * dt * gamma_state;
                        const float dmexp = (m_exp_raw > -20.f && m_exp_raw < 20.f) ? dx * a : 0.f;
                        const float eps = 1e-6f;
                        const float gamma_safe = fabsf(gamma_state) > eps ? gamma_state : (gamma_state >= 0.f ? eps : -eps);
                        const float alpha = dt * pq_diag_state / gamma_safe;
                        const float denom = 1.f - alpha;
                        const float denom_safe = fabsf(denom) > eps ? denom : (denom >= 0.f ? eps : -eps);
                        const float n_raw = 1.f - alpha / denom_safe;
                        const float dndpq = -dt / (gamma_safe * denom_safe * denom_safe);
                        const float dn = (n_raw > -8.f && n_raw < 8.f) ? dx * b0 : 0.f;
                        dPQ_val += dmexp * dsdpq + dn * dndpq;
                    }
                    if constexpr (!kIsVariableB || !kIsVariableC) {
                        if constexpr (!kIsVariableB) {  // dBC_val is dB_val
                            dBC_val += dout_vals[i] * (!kIsVariableC ? thread_data[i].y : thread_data[i].y * C_vals[i]);
                        } else {  // dBC_val is dC_val
                            dBC_val += dout_vals[i] * thread_data[i].y;
                        }
                    }
                    if constexpr (kIsVariableB) { dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]) * n_corr; }
                    if constexpr (kIsVariableC) {
                        dC_vals[i] = dout_vals[i] * (!kIsVariableB ? thread_data[i].y * B_val : thread_data[i].y);
                    }
                }
                // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                if constexpr (kIsVariableB || kIsVariableC) {
                    if constexpr (kIsVariableB) {
                        typename Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals, dB_vals);
                    }
                    if constexpr (kIsVariableC) {
                        auto &smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
                        typename Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals, dC_vals);
                    }
                    const int seqlen_remaining = params.seqlen - chunk * kChunkSize - threadIdx.x;
                    weight_t *dB_cur = dB + state_idx * params.dB_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    weight_t *dC_cur = dC + state_idx * params.dC_dstate_stride + chunk * kChunkSize + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItems; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            if constexpr (kIsVariableB) { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals[i]); }
                            if constexpr (kIsVariableC) { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals[i]); }
                        }
                    }
                }
                if constexpr (!kIsVariableB || !kIsVariableC) {
                    float2 dA_dBC_val = make_float2(dA_val, dBC_val);
                    dA_dBC_val = typename Ktraits::BlockReduceT(smem_reduce).Sum(dA_dBC_val);
                    dA_val = dA_dBC_val.x;
                    if (threadIdx.x == 0) {
                        smem_dbc[state_idx] = chunk == params.n_chunks - 1 ? dA_dBC_val.y : dA_dBC_val.y + smem_dbc[state_idx];
                    }
                } else {
                    dA_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dA_val);
                }
                if (threadIdx.x == 0) {
                    smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
                }
                if (need_dpq) {
                    __syncthreads();
                    float dPQ_sum = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dPQ_val);
                    if (threadIdx.x == 0) {
                        float *dP_state = dP_base + dim_id * params.dP_d_stride + state_idx * params.dP_dstate_stride;
                        float *dQ_state = dQ_base + dim_id * params.dQ_d_stride + state_idx * params.dQ_dstate_stride;
                        #pragma unroll 1
                        for (int k = 0; k < params.pq_rank; ++k) {
                            const float p_val = P_state[k * params.P_r_stride];
                            const float q_val = Q_state[k * params.Q_r_stride];
                            gpuAtomicAdd(dP_state + k * params.dP_r_stride, dPQ_sum * q_val);
                            gpuAtomicAdd(dQ_state + k * params.dQ_r_stride, dPQ_sum * p_val);
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    // Pytorch's implementation of complex exp (which calls thrust) is very slow
                    complex_t delta_a_exp = cexp2f(delta_vals[i] * A_scaled);
                    weight_t B_delta_u_val = !kIsVariableB ? delta_vals[i] * float(u_vals[i]) : B_vals[i] * delta_vals[i] * float(u_vals[i]);
                    thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                    if (i == 0) {
                        smem_delta_a[threadIdx.x == 0 ? state_idx + (chunk % 2) * MAX_DSTATE : threadIdx.x + 2 * MAX_DSTATE] = delta_a_exp;
                    } else {
                        thread_reverse_data[i - 1].x = delta_a_exp.real_;
                        thread_reverse_data[i - 1].y = -delta_a_exp.imag_;
                    }
                    complex_t dout_BC = 2 * dout_vals[i]
                        * conj(!kIsVariableC
                                ? (!kIsVariableB ? B_val * C_val : C_val)
                                : (!kIsVariableB ? B_val * C_vals[i] : C_vals[i]));
                    thread_reverse_data[i].z = dout_BC.real_;
                    thread_reverse_data[i].w = dout_BC.imag_;
                }
                __syncthreads();
                complex_t delta_a_exp = threadIdx.x == kNThreads - 1
                    ? (chunk == params.n_chunks - 1 ? 1.f : smem_delta_a[state_idx + ((chunk + 1) % 2) * MAX_DSTATE])
                    : smem_delta_a[threadIdx.x + 1 + 2 * MAX_DSTATE];
                thread_reverse_data[kNItems - 1].x = delta_a_exp.real_;
                thread_reverse_data[kNItems - 1].y = -delta_a_exp.imag_;
                // Initialize running total
                scan_t running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? x[(chunk - 1) * params.dstate + state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                scan_t running_postfix = chunk < params.n_chunks - 1 && threadIdx.x % 32 == 0 ? smem_running_postfix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                SSMScanPrefixCallbackOp<weight_t> postfix_op(running_postfix);
                typename Ktraits::BlockReverseScanT(smem_reverse_scan).InclusiveReverseScan(
                    thread_reverse_data, thread_reverse_data, SSMScanOp<weight_t>(), postfix_op
                );
                if (threadIdx.x == 0) { smem_running_postfix[state_idx] = postfix_op.running_prefix; }
                weight_t dA_val = 0, dBC_val = 0;
                weight_t dB_vals[kNItems], dC_vals[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    complex_t x = complex_t(thread_data[i].z, thread_data[i].w);
                    complex_t dx = complex_t(thread_reverse_data[i].z, thread_reverse_data[i].w);
                    float ddelta_u = !kIsVariableB ? dx.real_ : (dx * conj(B_vals[i])).real_;
                    if constexpr (!kIsVariableB || !kIsVariableC) {
                        if constexpr (!kIsVariableB) {  // dBC_val is dB_val
                            dBC_val += (2 * dout_vals[i]) * conj(!kIsVariableC ? x : x * C_vals[i]);
                        } else {  // dBC_val is dC_val
                            dBC_val += (2 * dout_vals[i]) * conj(x);
                        }
                    }
                    const complex_t a_conj = conj(x - (!kIsVariableB ? delta_vals[i] * float(u_vals[i]) : delta_vals[i] * float(u_vals[i]) * B_vals[i]));
                    du_vals[i] += ddelta_u * delta_vals[i];
                    ddelta_vals[i] += ddelta_u * float(u_vals[i]) + (dx * conj(A_val) * a_conj).real_;
                    dA_val += delta_vals[i] * dx * a_conj;
                    if constexpr (kIsVariableB) { dB_vals[i] = dx * delta_vals[i] * float(u_vals[i]); }
                    if constexpr (kIsVariableC) {
                        dC_vals[i] = (2 * dout_vals[i]) * conj(!kIsVariableB ? x * B_val : x);
                    }
                }
                // Block-exchange to make the atomicAdd's coalesced, otherwise they're much slower
                if constexpr (kIsVariableB || kIsVariableC) {
                    float dB_vals_f[kNItems * 2], dC_vals_f[kNItems * 2];
                    if constexpr (kIsVariableB) {
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            dB_vals_f[i * 2] = dB_vals[i].real_;
                            dB_vals_f[i * 2 + 1] = dB_vals[i].imag_;
                        }
                        typename Ktraits::BlockExchangeT(smem_exchange).BlockedToStriped(dB_vals_f, dB_vals_f);
                    }
                    if constexpr (kIsVariableC) {
                        #pragma unroll
                        for (int i = 0; i < kNItems; ++i) {
                            dC_vals_f[i * 2] = dC_vals[i].real_;
                            dC_vals_f[i * 2 + 1] = dC_vals[i].imag_;
                        }
                        auto &smem_exchange_C = !kIsVariableB ? smem_exchange : smem_exchange1;
                        typename Ktraits::BlockExchangeT(smem_exchange_C).BlockedToStriped(dC_vals_f, dC_vals_f);
                    }
                    const int seqlen_remaining = (params.seqlen - chunk * kChunkSize) * 2 - threadIdx.x;
                    float *dB_cur = reinterpret_cast<float *>(dB) + state_idx * params.dB_dstate_stride + chunk * kChunkSize * 2 + threadIdx.x;
                    float *dC_cur = reinterpret_cast<float *>(dC) + state_idx * params.dC_dstate_stride + chunk * kChunkSize * 2 + threadIdx.x;
                    #pragma unroll
                    for (int i = 0; i < kNItems * 2; ++i) {
                        if (i * kNThreads < seqlen_remaining) {
                            if constexpr (kIsVariableB) { gpuAtomicAdd(dB_cur + i * kNThreads, dB_vals_f[i]); }
                            if constexpr (kIsVariableC) { gpuAtomicAdd(dC_cur + i * kNThreads, dC_vals_f[i]); }
                        }
                    }
                }
                if constexpr (!kIsVariableB || !kIsVariableC) {
                    float4 dA_dBC_val = make_float4(dA_val.real_, dA_val.imag_, dBC_val.real_, dBC_val.imag_);
                    dA_dBC_val = typename Ktraits::BlockReduceT(smem_reduce).Sum(dA_dBC_val);
                    dA_val = complex_t(dA_dBC_val.x, dA_dBC_val.y);
                    dBC_val = complex_t(dA_dBC_val.z, dA_dBC_val.w);
                    if (threadIdx.x == 0) {
                        smem_dbc[state_idx] = chunk == params.n_chunks - 1 ? dBC_val : dBC_val + smem_dbc[state_idx];
                    }
                } else {
                    dA_val = typename Ktraits::BlockReduceComplexT(smem_reduce_complex).Sum(dA_val);
                }
                if (threadIdx.x == 0) {
                    smem_da[state_idx] = chunk == params.n_chunks - 1 ? dA_val : dA_val + smem_da[state_idx];
                }
            }
        }

        if constexpr (kDeltaSoftplus) {
            __syncthreads();
            input_t delta_vals_load[kNItems];
            load_input<Ktraits>(delta, delta_vals_load, smem_load, params.seqlen - chunk * kChunkSize);
            delta -= kChunkSize;
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float delta_val = float(delta_vals_load[i]) + delta_bias;
                float delta_val_neg_exp = expf(-delta_val);
                ddelta_vals[i] = delta_val <= 20.f
                    ? ddelta_vals[i] / (1.f + delta_val_neg_exp)
                    : ddelta_vals[i];
            }
        }
        for (int i = 0; i < kNItems; ++i) { ddelta_bias_val += ddelta_vals[i]; }

        input_t *du = reinterpret_cast<input_t *>(params.du_ptr) + batch_id * params.du_batch_stride
            + dim_id * params.du_d_stride + chunk * kChunkSize;
        input_t *ddelta = reinterpret_cast<input_t *>(params.ddelta_ptr) + batch_id * params.ddelta_batch_stride
            + dim_id * params.ddelta_d_stride + chunk * kChunkSize;
        __syncthreads();
        store_output<Ktraits>(du, du_vals, smem_store, params.seqlen - chunk * kChunkSize);
        __syncthreads();
        store_output<Ktraits>(ddelta, ddelta_vals, smem_store, params.seqlen - chunk * kChunkSize);

        Bvar -= kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar -= kChunkSize * (!kIsComplex ? 1 : 2);
    }
    if (params.dD_ptr != nullptr) {
        dD_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(dD_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(dD, dD_val); }
    }
    if (params.ddelta_bias_ptr != nullptr) {
        __syncthreads();
        ddelta_bias_val = typename Ktraits::BlockReduceFloatT(smem_reduce_float).Sum(ddelta_bias_val);
        if (threadIdx.x == 0) { gpuAtomicAdd(ddelta_bias, ddelta_bias_val); }
    }
    for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
        gpuAtomicAdd(&(dA[state_idx * params.dA_dstate_stride]), smem_da[state_idx]);
        weight_t dBC_val;
        if (!kIsVariableB || !kIsVariableC) { dBC_val = smem_dbc[state_idx]; }
        if constexpr (!kIsVariableB) {
            gpuAtomicAdd(&(dB[state_idx * params.dB_dstate_stride]),
                         !kIsVariableC ? dBC_val * conj(C[state_idx * params.C_dstate_stride]) : dBC_val);
        }
        if constexpr (!kIsVariableC) {
            gpuAtomicAdd(&(dC[state_idx * params.dC_dstate_stride]),
                        !kIsVariableB ? dBC_val * conj(B[state_idx * params.B_dstate_stride]) : dBC_val);
        }
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_bwd_launch(SSMParamsBwd &params, cudaStream_t stream) {
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
                BOOL_SWITCH(params.delta_softplus, kDeltaSoftplus, [&] {
                    BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
                        using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItems, kIsEvenLen, kIsVariableB, kIsVariableC, kDeltaSoftplus, kHasZ, input_t, weight_t>;
                        // using Ktraits = Selective_Scan_bwd_kernel_traits<kNThreads, kNItems, true, kIsVariableB, kIsVariableC, kDeltaSoftplus, kHasZ, input_t, weight_t>;
                        // TODO: check this
                        constexpr int kSmemSize = Ktraits::kSmemSize + MAX_DSTATE * sizeof(typename Ktraits::scan_t) + (kNThreads + 4 * MAX_DSTATE) * sizeof(typename Ktraits::weight_t);

                        dim3 grid(params.batch, params.dim);
                        
                        auto kernel = &selective_scan_bwd_kernel<Ktraits>;

                        if (kSmemSize >= 48 * 1024) {

                            #ifndef USE_ROCM
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            #else
                            C10_CUDA_CHECK(cudaFuncSetAttribute(
                                (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            std::cerr << "Warning (selective_scan_bwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                            #endif

                        }

                        kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                        C10_CUDA_KERNEL_LAUNCH_CHECK();
                    });
                });
            });
        });
    });
}

template<typename input_t>
void selective_scan_bwd_dense_pq_launch(SSMParamsBwd &params, cudaStream_t stream) {
    dim3 grid(params.batch, params.dim);
    selective_scan_bwd_dense_pq_kernel<input_t><<<grid, 1, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename weight_t>
void selective_scan_bwd_cuda(SSMParamsBwd &params, cudaStream_t stream) {
    if (params.has_pq) {
        selective_scan_bwd_dense_pq_launch<input_t>(params, stream);
        return;
    }

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {
            selective_scan_bwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_bwd_launch<32, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_bwd_launch<32, 16, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_bwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_bwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #else 
        if (params.seqlen <= 256) {
            selective_scan_bwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_bwd_launch<64, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_bwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_bwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #endif
}
