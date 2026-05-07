/******************************************************************************
 * Copyright (c) 2023, MotifMamba Team.
 ******************************************************************************/

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#ifndef USE_ROCM
    #include <cub/block/block_load.cuh>
    #include <cub/block/block_store.cuh>
    #include <cub/block/block_scan.cuh>
#else
    #include <hipcub/hipcub.hpp>
    namespace cub = hipcub;
#endif

#include "selective_scan.h"
#include "selective_scan_common.h"
#include "static_switch.h"

__device__ __forceinline__ void compute_m_n_scalar(
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

constexpr int kMaxDensePQRank = 8;
constexpr int kMaxDenseK = 3 * kMaxDensePQRank;

__device__ __forceinline__ void matmul_small(const float *A, const float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float acc = 0.f;
            for (int t = 0; t < k; ++t) { acc += A[i * k + t] * B[t * n + j]; }
            C[i * n + j] = acc;
        }
    }
}

__device__ __forceinline__ void matvec_small(const float *A, const float *x, float *y, int m, int n) {
    for (int i = 0; i < m; ++i) {
        float acc = 0.f;
        for (int j = 0; j < n; ++j) { acc += A[i * n + j] * x[j]; }
        y[i] = acc;
    }
}

__device__ __forceinline__ bool invert_small(float *A, float *Ainv, int n) {
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

__device__ __forceinline__ void expm_small(float *A, float *Aexp, int n) {
    float norm1 = 0.f;
    for (int c = 0; c < n; ++c) {
        float col = 0.f;
        for (int r = 0; r < n; ++r) { col += fabsf(A[r * n + c]); }
        norm1 = fmaxf(norm1, col);
    }
    int s = 0;
    while (norm1 > 0.5f && s < 6) { norm1 *= 0.5f; ++s; }
    const float scale = ldexpf(1.f, -s);

    float As[kMaxDenseK * kMaxDenseK];
    float term[kMaxDenseK * kMaxDenseK];
    float tmp[kMaxDenseK * kMaxDenseK];
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
        matmul_small(term, As, tmp, n, n, n);
        const float invk = 1.f / float(k);
        for (int i = 0; i < n * n; ++i) {
            term[i] = tmp[i] * invk;
            Aexp[i] += term[i];
        }
    }
    for (int i = 0; i < s; ++i) {
        matmul_small(Aexp, Aexp, tmp, n, n, n);
        for (int j = 0; j < n * n; ++j) { Aexp[j] = tmp[j]; }
    }
}

template<typename input_t>
__global__ void selective_scan_fwd_dense_pq_kernel(SSMParamsBase params) {
    if (threadIdx.x != 0) { return; }
    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    if (batch_id >= params.batch || dim_id >= params.dim) { return; }
    if (params.pq_rank <= 0 || params.pq_rank > kMaxDensePQRank || params.dstate > MAX_DSTATE) { return; }

    const int group_id = dim_id / params.dim_ngroups_ratio;
    const input_t *u = reinterpret_cast<const input_t *>(params.u_ptr) + batch_id * params.u_batch_stride + dim_id * params.u_d_stride;
    const input_t *delta = reinterpret_cast<const input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride + dim_id * params.delta_d_stride;
    const float *A = reinterpret_cast<const float *>(params.A_ptr) + dim_id * params.A_d_stride;
    const float *B_const = reinterpret_cast<const float *>(params.B_ptr) + dim_id * params.B_d_stride;
    const input_t *B_var = reinterpret_cast<const input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    const float *C_const = reinterpret_cast<const float *>(params.C_ptr) + dim_id * params.C_d_stride;
    const input_t *C_var = reinterpret_cast<const input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride + dim_id * params.out_d_stride;
    const input_t *z = params.z_ptr == nullptr ? nullptr : reinterpret_cast<const input_t *>(params.z_ptr) + batch_id * params.z_batch_stride + dim_id * params.z_d_stride;
    input_t *out_z = params.out_z_ptr == nullptr ? nullptr : reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride + dim_id * params.out_z_d_stride;
    float2 *x_chunks = reinterpret_cast<float2 *>(params.x_ptr) + (batch_id * params.dim + dim_id) * params.n_chunks * params.dstate;

    const float *P = reinterpret_cast<const float *>(params.P_ptr) + dim_id * params.P_d_stride;
    const float *Q = reinterpret_cast<const float *>(params.Q_ptr) + dim_id * params.Q_d_stride;
    const float D_val = params.D_ptr == nullptr ? 0.f : reinterpret_cast<const float *>(params.D_ptr)[dim_id];
    const float delta_bias = params.delta_bias_ptr == nullptr ? 0.f : reinterpret_cast<const float *>(params.delta_bias_ptr)[dim_id];

    const int n = params.dstate;
    const int r = params.pq_rank;
    const int chunk_size = 2048;

    float state[MAX_DSTATE] = {0.f};
    float next_state[MAX_DSTATE];
    float q_vec[kMaxDensePQRank];
    float aexp[MAX_DSTATE];
    float b_raw[MAX_DSTATE];
    float c_raw[MAX_DSTATE];

    for (int t = 0; t < params.seqlen; ++t) {
        float dt = float(delta[t]) + delta_bias;
        if (params.delta_softplus) {
            dt = dt <= 20.f ? log1pf(expf(dt)) : dt;
        }
        const float u_val = float(u[t]);

        for (int i = 0; i < n; ++i) {
            aexp[i] = expf(dt * A[i * params.A_dstate_stride]);
            b_raw[i] = !params.is_variable_B
                ? B_const[i * params.B_dstate_stride]
                : float(B_var[i * params.B_dstate_stride + t]);
            c_raw[i] = !params.is_variable_C
                ? C_const[i * params.C_dstate_stride]
                : float(C_var[i * params.C_dstate_stride + t]);
        }

        // Low-rank branch: first Q @ h(t-1), then P @ (...), complexity O(nr + rn).
        for (int j = 0; j < r; ++j) {
            float acc = 0.f;
            for (int i = 0; i < n; ++i) {
                const float q = Q[j * params.Q_r_stride + i * params.Q_dstate_stride];
                acc += q * state[i];
            }
            q_vec[j] = acc;
        }

        float out_val = D_val * u_val;
        for (int i = 0; i < n; ++i) {
            float pq_term = 0.f;
            for (int j = 0; j < r; ++j) {
                const float p = P[i * params.P_dstate_stride + j * params.P_r_stride];
                pq_term += p * q_vec[j];
            }
            const float b_term = dt * u_val * b_raw[i];
            next_state[i] = aexp[i] * state[i] + dt * pq_term + b_term;
            out_val += next_state[i] * c_raw[i];
        }

        for (int i = 0; i < n; ++i) { state[i] = next_state[i]; }

        out[t] = input_t(out_val);
        if (out_z != nullptr) {
            const float z_val = float(z[t]);
            const float silu = z_val / (1.f + expf(-z_val));
            out_z[t] = input_t(out_val * silu);
        }

        if ((t + 1) % chunk_size == 0 || t == params.seqlen - 1) {
            const int chunk_idx = t / chunk_size;
            for (int i = 0; i < n; ++i) {
                x_chunks[chunk_idx * n + i] = make_float2(1.f, state[i]);
            }
        }
    }
}

template<int kNThreads_, int kNItems_, int kNRows_, bool kIsEvenLen_,
         bool kIsVariableB_, bool kIsVariableC_,
         bool kHasZ_, typename input_t_, typename weight_t_>
struct Selective_Scan_fwd_kernel_traits {
    static_assert(kNItems_ % 4 == 0);
    using input_t = input_t_;
    using weight_t = weight_t_;
    static constexpr int kNThreads = kNThreads_;
    // Setting MinBlocksPerMP to be 3 (instead of 2) for 128 threads improves occupancy.
    static constexpr int kMinBlocks = kNThreads < 128 ? 5 : 3;
    static constexpr int kNItems = kNItems_;
    static constexpr int kNRows = kNRows_;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : constexpr_min(8, kNItems);
    static_assert(kNItems % kNElts == 0);
    static constexpr int kNLoads = kNItems / kNElts;
    static constexpr bool kIsComplex = std::is_same_v<weight_t, complex_t>;
    static constexpr bool kIsEvenLen = kIsEvenLen_;
    static constexpr bool kIsVariableB = kIsVariableB_;
    static constexpr bool kIsVariableC = kIsVariableC_;
    static constexpr bool kHasZ = kHasZ_;

    static constexpr bool kDirectIO = kIsEvenLen && kNLoads == 1;

    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    using scan_t = std::conditional_t<!kIsComplex, float2, float4>;
    using BlockLoadT = cub::BlockLoad<input_t, kNThreads, kNItems, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadVecT = cub::BlockLoad<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE : cub::BLOCK_LOAD_DIRECT>;
    using BlockLoadWeightT = cub::BlockLoad<input_t, kNThreads, !kIsComplex ? kNItems : kNItems * 2, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
    using BlockLoadWeightVecT = cub::BlockLoad<vec_t, kNThreads, !kIsComplex ? kNLoads : kNLoads * 2,
        !kDirectIO ? cub::BLOCK_LOAD_WARP_TRANSPOSE  : cub::BLOCK_LOAD_DIRECT>;
    using BlockStoreT = cub::BlockStore<input_t, kNThreads, kNItems, cub::BLOCK_STORE_WARP_TRANSPOSE>;
    using BlockStoreVecT = cub::BlockStore<vec_t, kNThreads, kNLoads,
        !kDirectIO ? cub::BLOCK_STORE_WARP_TRANSPOSE : cub::BLOCK_STORE_DIRECT>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING_MEMOIZE>;
    // using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_RAKING>;
    using BlockScanT = cub::BlockScan<scan_t, kNThreads, cub::BLOCK_SCAN_WARP_SCANS>;
    static constexpr int kSmemIOSize = custom_max({sizeof(typename BlockLoadT::TempStorage),
                                                 sizeof(typename BlockLoadVecT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightT::TempStorage),
                                                 (int(kIsVariableB) + int(kIsVariableC)) * sizeof(typename BlockLoadWeightVecT::TempStorage),
                                                 sizeof(typename BlockStoreT::TempStorage),
                                                 sizeof(typename BlockStoreVecT::TempStorage)});
    static constexpr int kSmemSize = kSmemIOSize + sizeof(typename BlockScanT::TempStorage);
};

template<typename Ktraits>
__global__ __launch_bounds__(Ktraits::kNThreads, Ktraits::kMinBlocks)
void selective_scan_fwd_kernel(SSMParamsBase params) {
    constexpr bool kIsComplex = Ktraits::kIsComplex;
    constexpr bool kIsVariableB = Ktraits::kIsVariableB;
    constexpr bool kIsVariableC = Ktraits::kIsVariableC;
    constexpr bool kHasZ = Ktraits::kHasZ;
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNItems = Ktraits::kNItems;
    constexpr int kNRows = Ktraits::kNRows;
    constexpr bool kDirectIO = Ktraits::kDirectIO;
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
    auto& smem_scan = *reinterpret_cast<typename Ktraits::BlockScanT::TempStorage*>(smem_ + Ktraits::kSmemIOSize);
    // weight_t *smem_a = reinterpret_cast<weight_t *>(smem_ + smem_loadstorescan_size);
    // weight_t *smem_bc = reinterpret_cast<weight_t *>(smem_a + MAX_DSTATE);
    scan_t *smem_running_prefix = reinterpret_cast<scan_t *>(smem_ + Ktraits::kSmemSize);

    const int batch_id = blockIdx.x;
    const int dim_id = blockIdx.y;
    const int group_id = dim_id / (params.dim_ngroups_ratio);
    input_t *u = reinterpret_cast<input_t *>(params.u_ptr) + batch_id * params.u_batch_stride
        + dim_id * kNRows * params.u_d_stride;
    input_t *delta = reinterpret_cast<input_t *>(params.delta_ptr) + batch_id * params.delta_batch_stride
        + dim_id * kNRows * params.delta_d_stride;
    weight_t *A = reinterpret_cast<weight_t *>(params.A_ptr) + dim_id * kNRows * params.A_d_stride;
    weight_t *B = reinterpret_cast<weight_t *>(params.B_ptr) + dim_id * kNRows * params.B_d_stride;
    input_t *Bvar = reinterpret_cast<input_t *>(params.B_ptr) + batch_id * params.B_batch_stride + group_id * params.B_group_stride;
    weight_t *C = reinterpret_cast<weight_t *>(params.C_ptr) + dim_id * kNRows * params.C_d_stride;
    input_t *Cvar = reinterpret_cast<input_t *>(params.C_ptr) + batch_id * params.C_batch_stride + group_id * params.C_group_stride;
    scan_t *x = reinterpret_cast<scan_t *>(params.x_ptr) + (batch_id * params.dim + dim_id * kNRows) * params.n_chunks * params.dstate;

    float D_val[kNRows] = {0};
    if (params.D_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            D_val[r] = reinterpret_cast<float *>(params.D_ptr)[dim_id * kNRows + r];
        }
    }
    float delta_bias[kNRows] = {0};
    if (params.delta_bias_ptr != nullptr) {
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            delta_bias[r] = reinterpret_cast<float *>(params.delta_bias_ptr)[dim_id * kNRows + r];
        }
    }

    // for (int state_idx = threadIdx.x; state_idx < params.dstate; state_idx += blockDim.x) {
    //     smem_a[state_idx] = A[state_idx * params.A_dstate_stride];
    //     smem_bc[state_idx] = B[state_idx * params.B_dstate_stride] * C[state_idx * params.C_dstate_stride];
    // }

    constexpr int kChunkSize = kNThreads * kNItems;
    for (int chunk = 0; chunk < params.n_chunks; ++chunk) {
        input_t u_vals[kNRows][kNItems], delta_vals_load[kNRows][kNItems];
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            load_input<Ktraits>(u + r * params.u_d_stride, u_vals[r], smem_load, params.seqlen - chunk * kChunkSize);
            if constexpr (!kDirectIO) { __syncthreads(); }
            load_input<Ktraits>(delta + r * params.delta_d_stride, delta_vals_load[r], smem_load, params.seqlen - chunk * kChunkSize);
        }
        u += kChunkSize;
        delta += kChunkSize;
    
        float delta_vals[kNRows][kNItems], delta_u_vals[kNRows][kNItems], out_vals[kNRows][kNItems];
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            #pragma unroll
            for (int i = 0; i < kNItems; ++i) {
                float u_val = float(u_vals[r][i]);
                delta_vals[r][i] = float(delta_vals_load[r][i]) + delta_bias[r];
                if (params.delta_softplus) {
                    delta_vals[r][i] = delta_vals[r][i] <= 20.f ? log1pf(expf(delta_vals[r][i])) : delta_vals[r][i];
                }
                delta_u_vals[r][i] = delta_vals[r][i] * u_val;
                out_vals[r][i] = D_val[r] * u_val;
            }
        }

        __syncthreads();
        for (int state_idx = 0; state_idx < params.dstate; ++state_idx) {
            weight_t A_val[kNRows];
            float A_gamma[kNRows];
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                A_val[r] = A[state_idx * params.A_dstate_stride + r * params.A_d_stride];
                // Multiply the real part of A with LOG2E so we can use exp2f instead of expf.
                constexpr float kLog2e = M_LOG2E;
                if constexpr (!kIsComplex) {
                    A_gamma[r] = A_val[r];
                    A_val[r] *= kLog2e;
                } else {
                    A_gamma[r] = A_val[r].real_;
                    A_val[r].real_ *= kLog2e;
                }
            }
            // This variable holds B * C if both B and C are constant across seqlen. If only B varies
            // across seqlen, this holds C. If only C varies across seqlen, this holds B.
            // If both B and C vary, this is unused.
            weight_t BC_val[kNRows];
            weight_t B_vals[kNItems], C_vals[kNItems];
            if constexpr (kIsVariableB) {
                load_weight<Ktraits>(Bvar + state_idx * params.B_dstate_stride, B_vals,
                    smem_load_weight, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableC) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                    }
                }
            }
            if constexpr (kIsVariableC) {
                auto &smem_load_weight_C = !kIsVariableB ? smem_load_weight : smem_load_weight1;
                load_weight<Ktraits>(Cvar + state_idx * params.C_dstate_stride, C_vals,
                    smem_load_weight_C, (params.seqlen - chunk * kChunkSize) * (!kIsComplex ? 1 : 2));
                if constexpr (!kIsVariableB) {
                    #pragma unroll
                    for (int r = 0; r < kNRows; ++r) {
                        BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride];
                    }
                }
            }
            if constexpr (!kIsVariableB && !kIsVariableC) {
                #pragma unroll
                for (int r = 0; r < kNRows; ++r) {
                    BC_val[r] = B[state_idx * params.B_dstate_stride + r * params.B_d_stride] * C[state_idx * params.C_dstate_stride + r * params.C_d_stride];
                }
            }

            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                if (r > 0) { __syncthreads(); }  // Scan could be using the same smem
                scan_t thread_data[kNItems];
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    if constexpr (!kIsComplex) {
                        float m_corr = 1.f, n_corr = 1.f;
                        compute_m_n_scalar(params, dim_id, state_idx, delta_vals[r][i], A_gamma[r], m_corr, n_corr);
                        float log2_a = delta_vals[r][i] * A_val[r];
                        log2_a = fmaxf(-80.f, fminf(80.f, log2_a));
                        thread_data[i] = make_float2(exp2f(log2_a) * m_corr,
                                                     n_corr * (!kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i]));
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float2(1.f, 0.f);
                            }
                        }
                    } else {
                        // Pytorch's implementation of complex exp (which calls thrust) is very slow
                        complex_t delta_a_exp = cexp2f(delta_vals[r][i] * A_val[r]);
                        weight_t B_delta_u_val = !kIsVariableB ? delta_u_vals[r][i] : B_vals[i] * delta_u_vals[r][i];
                        thread_data[i] = make_float4(delta_a_exp.real_, delta_a_exp.imag_, B_delta_u_val.real_, B_delta_u_val.imag_);
                        if constexpr (!Ktraits::kIsEvenLen) {  // So that the last state is correct
                            if (threadIdx.x * kNItems + i >= params.seqlen - chunk * kChunkSize) {
                                thread_data[i] = make_float4(1.f, 0.f, 0.f, 0.f);
                            }
                        }
                    }
                }
                // Initialize running total
                scan_t running_prefix;
                if constexpr (!kIsComplex) {
                    // If we use WARP_SCAN then all lane 0 of all warps (not just thread 0) needs to read
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float2(1.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float2(1.f, 0.f);
                } else {
                    running_prefix = chunk > 0 && threadIdx.x % 32 == 0 ? smem_running_prefix[state_idx + r * MAX_DSTATE] : make_float4(1.f, 0.f, 0.f, 0.f);
                    // running_prefix = chunk > 0 && threadIdx.x == 0 ? smem_running_prefix[state_idx] : make_float4(1.f, 0.f, 0.f, 0.f);
                }
                SSMScanPrefixCallbackOp<weight_t> prefix_op(running_prefix);
                typename Ktraits::BlockScanT(smem_scan).InclusiveScan(
                    thread_data, thread_data, SSMScanOp<weight_t>(), prefix_op
                );
                // There's a syncthreads in the scan op, so we don't need to sync here.
                // Unless there's only 1 warp, but then it's the same thread (0) reading and writing.
                if (threadIdx.x == 0) {
                    smem_running_prefix[state_idx] = prefix_op.running_prefix;
                    x[(r * params.n_chunks + chunk) * params.dstate + state_idx] = prefix_op.running_prefix;
                }
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    const weight_t C_val = !kIsVariableC
                        ? BC_val[r]
                        : (!kIsVariableB ? BC_val[r] * C_vals[i] : C_vals[i]);
                    if constexpr (!kIsComplex) {
                        out_vals[r][i] += thread_data[i].y * C_val;
                    } else {
                        out_vals[r][i] += (complex_t(thread_data[i].z, thread_data[i].w) * C_val).real_ * 2;
                    }
                }
            }
        }
        
        input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride
            + dim_id * kNRows * params.out_d_stride + chunk * kChunkSize;
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < kNRows; ++r) {
            if constexpr (!kDirectIO) {
                if (r > 0) { __syncthreads(); }
            }
            store_output<Ktraits>(out + r * params.out_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
        }

        if constexpr (kHasZ) {
            input_t *z = reinterpret_cast<input_t *>(params.z_ptr) + batch_id * params.z_batch_stride
                + dim_id * kNRows * params.z_d_stride + chunk * kChunkSize;
            input_t *out_z = reinterpret_cast<input_t *>(params.out_z_ptr) + batch_id * params.out_z_batch_stride
                + dim_id * kNRows * params.out_z_d_stride + chunk * kChunkSize;
            #pragma unroll
            for (int r = 0; r < kNRows; ++r) {
                input_t z_vals[kNItems];
                __syncthreads();
                load_input<Ktraits>(z + r * params.z_d_stride, z_vals, smem_load, params.seqlen - chunk * kChunkSize);
                #pragma unroll
                for (int i = 0; i < kNItems; ++i) {
                    float z_val = z_vals[i];
                    out_vals[r][i] *= z_val / (1 + expf(-z_val));
                }
                __syncthreads();
                store_output<Ktraits>(out_z + r * params.out_z_d_stride, out_vals[r], smem_store, params.seqlen - chunk * kChunkSize);
            }
        }

        Bvar += kChunkSize * (!kIsComplex ? 1 : 2);
        Cvar += kChunkSize * (!kIsComplex ? 1 : 2);
    }
}

template<int kNThreads, int kNItems, typename input_t, typename weight_t>
void selective_scan_fwd_launch(SSMParamsBase &params, cudaStream_t stream) {
    // Only kNRows == 1 is tested for now, which ofc doesn't differ from previously when we had each block
    // processing 1 row.
    constexpr int kNRows = 1;
    BOOL_SWITCH(params.seqlen % (kNThreads * kNItems) == 0, kIsEvenLen, [&] {
        BOOL_SWITCH(params.is_variable_B, kIsVariableB, [&] {
            BOOL_SWITCH(params.is_variable_C, kIsVariableC, [&] {
                BOOL_SWITCH(params.z_ptr != nullptr , kHasZ, [&] {
                    using Ktraits = Selective_Scan_fwd_kernel_traits<kNThreads, kNItems, kNRows, kIsEvenLen, kIsVariableB, kIsVariableC, kHasZ, input_t, weight_t>;
                    
                    constexpr int kSmemSize = Ktraits::kSmemSize + kNRows * MAX_DSTATE * sizeof(typename Ktraits::scan_t);
                    dim3 grid(params.batch, params.dim / kNRows);

                    // Had to change this substantially since potentially the hip 
                    // interface for setting kernel launch attributes is slightly different from 
                    // cuda's. In particualar, it seems to expect a plain const void * pointer.

                    auto kernel = &selective_scan_fwd_kernel<Ktraits>;

                    
                    if (kSmemSize >= 48 * 1024) {
                        #ifndef USE_ROCM
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                        #else
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            (void *) kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
                            std::cerr << "Warning (selective_scan_fwd_kernel): attempting to set maxDynamicSharedMemorySize on an AMD GPU which is currently a non-op (in ROCm versions <= 6.1). This might lead to undefined behavior. \n" << std::endl;
                        #endif
                    }

                    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename input_t>
void selective_scan_fwd_dense_pq_launch(SSMParamsBase &params, cudaStream_t stream) {
    dim3 grid(params.batch, params.dim);
    selective_scan_fwd_dense_pq_kernel<input_t><<<grid, 1, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<typename input_t, typename weight_t>
void selective_scan_fwd_cuda(SSMParamsBase &params, cudaStream_t stream) {
    if (params.has_pq) {
        selective_scan_fwd_dense_pq_launch<input_t>(params, stream);
        return;
    }

    #ifndef USE_ROCM
        if (params.seqlen <= 128) {           
            selective_scan_fwd_launch<32, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 256) {
            selective_scan_fwd_launch<32, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<32, 16, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #else
        if (params.seqlen <= 256) {
            selective_scan_fwd_launch<64, 4, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 512) {
            selective_scan_fwd_launch<64, 8, input_t, weight_t>(params, stream);
        } else if (params.seqlen <= 1024) {
            selective_scan_fwd_launch<64, 16, input_t, weight_t>(params, stream);
        } else {
            selective_scan_fwd_launch<128, 16, input_t, weight_t>(params, stream);
        }
    #endif
}
