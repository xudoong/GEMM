#include "gemm.cuh"
#include "utils.cuh"
#include <cuda/pipeline>
#include <cooperative_groups.h>

const int bM = 128;
const int bN = 128;
const int bK = 32;
const int wM = 64;
const int wN = 64;
const int tM = 16;
const int tN = 8;
const int tK = 16;

// number of pipeline stages
const int num_stages = 2;

const int num_wM = bM / wM;
const int num_wN = bN / wN;

const int num_tM = wM / tM;
const int num_tN = wN / tN;

const int nthreads = 32 * num_wM * num_wN;

// use for load global memory -> shared memory
const int ldWidth = 128 / 8 / sizeof(half);
const int bK2 = bK / ldWidth;
const int bN2 = bN / ldWidth;
const int stride_A_m = nthreads / bK2;
const int stride_B_k = nthreads / bN2;

#define FLOAT4(ptr) ((float4*)ptr)
#define CFLOAT4(ptr) ((const float4*)ptr)

using pipeline_t = cuda::pipeline<cuda::thread_scope::thread_scope_block>;

__device__ static void load_shared_A(void *dst, const void *src, int K, pipeline_t &pipeline,
                                     int tid, int offset_A_m, int offset_A_k) {
    for (int i = 0; i < bM; i += stride_A_m) {
        int row = (i * bK2 + tid) / 8;
        int col = (i * bK2 + tid) % 8;
        col = col ^ (row % bK2);

        int sAi = row * 8 + col;
        int gAi = (i + offset_A_m) * (K / ldWidth) + offset_A_k;
        cuda::memcpy_async(&FLOAT4(dst)[sAi], &CFLOAT4(src)[gAi], sizeof(float4), pipeline);
    }
}

__device__ static void load_shared_B(void *dst, const void *src, int N, pipeline_t &pipeline,
                                     int tid, int offset_B_k, int offset_B_n) {
    for (int i = 0; i < bK; i += stride_B_k) {
        int row = (i * bN2 + tid) / bN2;
        int col = (i * bN2 + tid) % bN2;
        col = col ^ (row % bN2);

        int sBi = row * bN2 + col;
        int gBi = (i + offset_B_k) * (N / ldWidth) + offset_B_n;
        cuda::memcpy_async(&FLOAT4(dst)[sBi], &CFLOAT4(src)[gBi], sizeof(float4), pipeline);
    }
}

__device__ static void compute(const half *sA, const half *sB, uint32_t *rA, uint32_t *rB, float *rC) {
    const int lane = threadIdx.x;

    const int wi = threadIdx.y;
    const int wj = threadIdx.z;

    const half *sAw = &sA[wi * wM * bK];

    uint32_t smem_ptr;

    for (int k = 0; k < bK; k += tK) {
        for (int j = 0; j < wN; j += tN) {
            // load B from shared memory -> register
            // slow
//            int idx = (k + lane % tK) * bN2 + (wj * wN + j) / 8;
//            int row = idx / bN2;
//            int col = idx % bN2;
            // fast
            int row = k + lane % tK;
            int col = wj * (wN / 8) + j / 8;
            col = col ^ (row % bN2);

            int sBi = row * bN2 + col;
            smem_ptr = __cvta_generic_to_shared(&FLOAT4(sB)[sBi]);
            uint32_t *rBt = &rB[j / tN * 2];

            asm volatile ( "ldmatrix.sync.aligned.x2.m8n8.shared.b16 "
                           "{%0, %1}, [%2]; "
                    : "=r"(rBt[0]), "=r"(rBt[1])
                    : "r"(smem_ptr)
                    );
            asm volatile ( "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
                    : "=r"(rBt[0]) : "r"(rBt[0])
                    );
            asm volatile ( "movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;"
                    : "=r"(rBt[1]) : "r"(rBt[1])
                    );

        }

        for (int i = 0; i < wM; i += tM) {
            // load A from shared memory -> register
            int idx = (i + lane % tM) * bK2 + (k / 8 + lane / tM);
            int row = idx / 8;
            int col = idx % 8;
            col = col ^ (row % bK2);
            int sAi = row * 8 + col;

            smem_ptr = __cvta_generic_to_shared(&FLOAT4(sAw)[sAi]);

            asm volatile ( "ldmatrix.sync.aligned.x4.m8n8.shared.b16 "
                           "{%0, %1, %2, %3}, [%4]; "
                    : "=r"(rA[0]), "=r"(rA[1]), "=r"(rA[2]), "=r"(rA[3])
                    : "r"(smem_ptr)
                    );
            for (int j = 0; j < wN; j += tN) {
                uint32_t *rBt = &rB[j / tN * 2];
                float *rCt = &rC[(i / tM * num_tN + j / tN) * 4];
                asm( "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                     " { %0, %1, %2, %3 }, "
                     " { %4, %5, %6, %7 }, "
                     " { %8, %9 }, "
                     " { %10, %11, %12, %13 };"
                        : "=f"(rCt[0]), "=f"(rCt[1]), "=f"(rCt[2]), "=f"(rCt[3])
                        : "r"(rA[0]), "r"(rA[1]), "r"(rA[2]), "r"(rA[3]),
                          "r"(rBt[0]), "r"(rBt[1]),
                          "f"(rCt[0]), "f"(rCt[1]), "f"(rCt[2]), "f"(rCt[3])
                        );
            }
        }
    }
}

__global__ static void kernel07(GEMM_FUNC_SIGNITURE) {
    __shared__ half sA[num_stages][bM * bK];
    __shared__ half sB[num_stages][bK * bN];

    // m16n8k16 mma
    uint32_t rA[4];
    uint32_t rB[num_tN * 2];
    float rC[num_tM * num_tN * 4] = {0};

    int bi = blockIdx.x;
    int bj = blockIdx.y;

    A += bi * bM * K;
    B += bj * bN;
    C += bi * bM * N + bj * bN;

    const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;

    const int wi = threadIdx.y;
    const int wj = threadIdx.z;

    C += wi * wM * N + wj * wN;

    const int offset_A_m = tid / bK2;
    const int offset_A_k = tid % bK2;
    const int offset_B_k = tid / bN2;
    const int offset_B_n = tid % bN2;

    __shared__ cuda::pipeline_shared_state<cuda::thread_scope::thread_scope_block, num_stages> shared_state;
    auto pipeline = cuda::make_pipeline(cooperative_groups::this_thread_block(), &shared_state);

    for (int stage = 0; stage < num_stages - 1; stage++) {
        pipeline.producer_acquire();
        load_shared_A(sA[stage], A, K, pipeline, tid, offset_A_m, offset_A_k);
        load_shared_B(sB[stage], B, N, pipeline, tid, offset_B_k, offset_B_n);
        pipeline.producer_commit();
        A += bK;
        B += bK * N;
    }

    for (int s = num_stages - 1; s < K / bK; s++) {
        int comp_stage = (s - num_stages + 1) % num_stages;
        int copy_stage = s % num_stages;

        pipeline.producer_acquire();
        load_shared_A(sA[copy_stage], A, K, pipeline, tid, offset_A_m, offset_A_k);
        load_shared_B(sB[copy_stage], B, N, pipeline, tid, offset_B_k, offset_B_n);
        pipeline.producer_commit();

        pipeline.consumer_wait();
        compute(sA[comp_stage], sB[comp_stage], rA, rB, rC);
        pipeline.consumer_release();

        // advance to next tile
        A += bK;
        B += bK * N;
    }

    for (int s = 0; s < num_stages - 1; s++) {
        int stage = (K / bK + 1 + s) % num_stages;
        pipeline.consumer_wait();
        compute(sA[stage], sB[stage], rA, rB, rC);
        pipeline.consumer_release();
    }

    // write to C
    const int tCi = threadIdx.x / 4;
    const int tCj = threadIdx.x % 4;
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            float *rCt = &rC[(i / tM * num_tN + j / tN) * 4];
            C[(i + tCi + 0) * N + j + tCj * 2 + 0] = rCt[0];
            C[(i + tCi + 0) * N + j + tCj * 2 + 1] = rCt[1];
            C[(i + tCi + 8) * N + j + tCj * 2 + 0] = rCt[2];
            C[(i + tCi + 8) * N + j + tCj * 2 + 1] = rCt[3];
        }
    }
}

void gemm_07_mma_permute(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    assert(bK2 <= 8);
    if (alpha != 1 || beta != 0) {
        std::cout << "gemm_07_mma_permute kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    dim3 gridDim(M / bM, N / bN);
    dim3 blockDim(32, num_wM, num_wN);
    kernel07<<<gridDim, blockDim>>>(GEMM_FUNC_PARAM);
}