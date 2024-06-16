#include "gemm.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

const int bM = 128;
const int bN = 128;
const int bK = 32;
const int wM = 64;
const int wN = 64;
const int tM = 16;
const int tN = 16;
const int tK = 16;

// number of pipeline stages
const int num_stages = 2;

const int num_wM = bM / wM;
const int num_wN = bN / wN;

const int num_tM = wM / tM;
const int num_tN = wN / tN;

// use for load global memory -> shared memory
const int ldWidth = 128 / 8 / sizeof(half);
const int bK2 = bK / ldWidth;
const int bN2 = bN / ldWidth;

#define FLOAT4(ptr) ((float4*)ptr)
#define CFLOAT4(ptr) ((const float4*)ptr)

// padding shared memory to avoid bank conflict
const int pad_half = 8;
const int pad_float4 = 1;

using fragA_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, tM, tN, tK, half, nvcuda::wmma::row_major>;
using fragB_t = nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, tM, tN, tK, half, nvcuda::wmma::row_major>;
using accum_t = nvcuda::wmma::fragment<nvcuda::wmma::accumulator, tM, tN, tK, float>;

using pipeline_t = cuda::pipeline<cuda::thread_scope::thread_scope_block>;

__device__ static void load_shared_A(void *dst, const void *src, int K, pipeline_t &pipeline,
                                     int stride_A_m, int offset_A_m, int offset_A_k) {
    for (int i = 0; i < bM; i += stride_A_m) {
        int sAi = (i + offset_A_m) * (bK2 + pad_float4) + offset_A_k;
        int gAi = (i + offset_A_m) * (K / ldWidth) + offset_A_k;
        cuda::memcpy_async(&FLOAT4(dst)[sAi], &CFLOAT4(src)[gAi], sizeof(float4), pipeline);
    }
}

__device__ static void load_shared_B(void *dst, const void *src, int N, pipeline_t &pipeline,
                                     int stride_B_k, int offset_B_k, int offset_B_n) {
    for (int i = 0; i < bK; i += stride_B_k) {
        int sBi = (i + offset_B_k) * (bN2 + pad_float4) + offset_B_n;
        int gBi = (i + offset_B_k) * (N / ldWidth) + offset_B_n;
        cuda::memcpy_async(&FLOAT4(dst)[sBi], &CFLOAT4(src)[gBi], sizeof(float4), pipeline);
    }
}

__device__ static void compute(const half *sA, const half *sB, fragA_t &fragA, fragB_t *fragB, accum_t *accum) {
    const int wi = threadIdx.y;
    const int wj = threadIdx.z;

    const half *sAw = &sA[wi * wM * (bK + pad_half)];
    const half *sBw = &sB[wj * wN];
    for (int k = 0; k < bK; k += tK) {
        for (int j = 0; j < wN; j += tN) {
            nvcuda::wmma::load_matrix_sync(fragB[j / tN], &sBw[k * (bN + pad_half) + j], bN + pad_half);
        }
        for (int i = 0; i < wM; i += tM) {
            nvcuda::wmma::load_matrix_sync(fragA, &sAw[i * (bK + pad_half) + k], bK + pad_half);
            for (int j = 0; j < wN; j += tN) {
                auto &acc = accum[i / tM * num_tN + j / tN];
                nvcuda::wmma::mma_sync(acc, fragA, fragB[j / tN], acc);
            }
        }
    }
}

__global__ static void kernel(GEMM_FUNC_SIGNITURE) {
    __shared__ half sA[num_stages][bM * (bK + pad_half)];
    __shared__ half sB[num_stages][bK * (bN + pad_half)];

    int bi = blockIdx.x;
    int bj = blockIdx.y;

    A += bi * bM * K;
    B += bj * bN;
    C += bi * bM * N + bj * bN;

    fragA_t fragA;
    fragB_t fragB[num_tN];
    accum_t accum[num_tM * num_tN];

    for (int i = 0; i < num_tM * num_tN; i++) {
        nvcuda::wmma::fill_fragment(accum[i], 0.0);
    }

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    const int nthreads = blockDim.x * blockDim.y * blockDim.z;

    const int wi = threadIdx.y;
    const int wj = threadIdx.z;

    C += wi * wM * N + wj * wN;

    const int stride_A_m = nthreads / bK2;
    const int offset_A_m = tid / bK2;
    const int offset_A_k = tid % bK2;
    const int stride_B_k = nthreads / bN2;
    const int offset_B_k = tid / bN2;
    const int offset_B_n = tid % bN2;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, num_stages> shared_state;
    auto pipeline = cuda::make_pipeline(cooperative_groups::this_thread_block(), &shared_state);

    for (int stage = 0; stage < num_stages - 1; stage++) {
        pipeline.producer_acquire();
        load_shared_A(sA[stage], A, K, pipeline, stride_A_m, offset_A_m, offset_A_k);
        load_shared_B(sB[stage], B, N, pipeline, stride_B_k, offset_B_k, offset_B_n);
        pipeline.producer_commit();
        A += bK;
        B += bK * N;
    }

    for (int s = num_stages - 1; s < K / bK; s++) {
        int comp_stage = (s - num_stages + 1) % num_stages;
        int copy_stage = s % num_stages;

        pipeline.producer_acquire();
        load_shared_A(sA[copy_stage], A, K, pipeline, stride_A_m, offset_A_m, offset_A_k);
        load_shared_B(sB[copy_stage], B, N, pipeline, stride_B_k, offset_B_k, offset_B_n);
        pipeline.producer_commit();

        pipeline.consumer_wait();
        compute(sA[comp_stage], sB[comp_stage], fragA, fragB, accum);
        pipeline.consumer_release();

        // advance to next tile
        A += bK;
        B += bK * N;
    }

    for (int s = 0; s < num_stages - 1; s++) {
        int stage = (K / bK + 1 + s) % num_stages;
        pipeline.consumer_wait();
        compute(sA[stage], sB[stage], fragA, fragB, accum);
        pipeline.consumer_release();
    }

    // write to C
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            auto &acc = accum[i / tM * num_tN + j / tN];
            nvcuda::wmma::store_matrix_sync(&C[i * N + j], acc, N, nvcuda::wmma::mem_row_major);
        }
    }
}

void gemm_05_wmma_pipeline(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    if (alpha != 1 || beta != 0) {
        std::cout << "gemm_05_wmma_pipeline kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32, num_wM, num_wN);
    kernel<<<gridDim, blockDim>>>(GEMM_FUNC_PARAM);
}