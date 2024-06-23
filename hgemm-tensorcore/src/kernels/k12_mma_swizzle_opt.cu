#include "gemm.cuh"

const int bM = 256;
const int bN = 128;
const int bK = 32;
const int wM = 64;
const int wN = 64;
const int tM = 16;
const int tN = 8;
const int tK = 16;

const int num_wM = bM / wM;
const int num_wN = bN / wN;
const int num_warps = num_wM * num_wN;

const int num_tM = wM / tM;
const int num_tN = wN / tN;

//const int nthreads = 32 * num_wM * num_wN;

// use for load global memory -> shared memory
const int ldWidth = 128 / 8 / sizeof(half); // 8
const int bK2 = bK / ldWidth;
const int bN2 = bN / ldWidth;
const int smem_stride_ij = 32 / bK2;
const int stride_C_m = 32 / bN2;

#define FLOAT4(ptr) ((float4*)ptr)
#define CFLOAT4(ptr) ((const float4*)ptr)

const int pad_half = 8;
const int pad_float4 = 1;

constexpr int num_stages = 4;

#define sA(stage) (&sA[stage * bM * bK])
#define sB(stage) (&sB[stage * bN * bK])

const int block_j_swizzle_factor = 16;

__device__ static void load_shared_AB(half *shared_ptr, const half *global_ptr, int K, int lane, int wid, int count) {
    const int warp_ij_offset = wid * count / num_warps;
    shared_ptr += warp_ij_offset * bK;
    global_ptr += warp_ij_offset * K;

    const int offset_ij = lane / bK2;
    const int offset_k = lane % bK2;

    for (int i = 0; i < count / num_warps; i += smem_stride_ij) {
        int row = (i * bK2 + lane) / 8;
        int col = (i * bK2 + lane) % 8;
        col = col ^ (row % bK2);

        float4 *dst       = &FLOAT4(shared_ptr)[row * 8 + col];
        const float4 *src = &CFLOAT4(global_ptr)[(i + offset_ij) * (K / ldWidth) + offset_k];
        copy_async<sizeof(float4)>(dst, src);
    }
}

__device__ static void store_shared_C(half *C, const half *sC, int N, int lane, int wid) {
    const int warp_i_offset = wid * bM / num_warps;
    C += warp_i_offset * N;
    sC += warp_i_offset * (bN + pad_half);

    const int offset_C_m = lane / bN2;
    const int offset_C_n = lane % bN2;

    for (int i = 0; i < bM / num_warps; i += stride_C_m) {
        float4 *dst       =   &FLOAT4(C)[(i + offset_C_m) * (N / ldWidth) + offset_C_n];
        const float4 *src = &CFLOAT4(sC)[(i + offset_C_m) * (bN2 + pad_float4) + offset_C_n];
        *dst = *src;
    }
}

__device__ static void load_register(int lane, int wi, int wj, const half *sA, const half *sB,
                                     uint32_t *rA, uint32_t *rB, int k) {
    const half *sAw = &sA[wi * wM * bK];
    const half *sBw = &sB[wj * wN * bK];

    for (int i = 0; i < wM; i += tM) {
        int idx = (i + lane % tM) * bK2 + (k / 8 + lane / tM);
        int row = idx / 8;
        int col = idx % 8;
        col = col ^ (row % bK2);
        ldmatrixA(&FLOAT4(sAw)[row * 8 + col], &rA[i / tM * 4]);
    }

    for (int j = 0; j < wN; j += tN) {
        int idx = (j + lane % tN) * bK2 + (k / 8 + lane / tN);
        int row = idx / 8;
        int col = idx % 8;
        col = col ^ (row % bK2);
        ldmatrixB(&FLOAT4(sBw)[row * 8 + col], &rB[j / tN * 2]);
    }
}

__device__ static void do_mma(uint32_t *rA, uint32_t *rB, uint32_t *rC) {
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            int j1 = i % 2 ? wN - tN - j : j;
            mma_m16n8k16(&rA[i / tM * 4],
                         &rB[j1 / tN * 2],
                         &rC[(i / tM * num_tN + j1 / tN) * 2]);
        }
    }
}

template <int stage_idx>
__device__ static void post_stage(const int lane, const int wi, const int wj, int K, const half *sA, const half *sB,
                                  uint32_t rA[][num_tM * 4], uint32_t rB[][num_tN * 2], uint32_t *rC, int &reg_st, int &reg_ld) {
    if (stage_idx > num_stages) {
        return;
    }

    int stage = (K / bK + stage_idx - 1) % num_stages;
    for (int k = tK; k < bK; k += tK) {
        load_register(lane, wi, wj, sA(stage), sB(stage), rA[reg_st], rB[reg_st], k);
        do_mma(rA[reg_ld], rB[reg_ld], rC);
        reg_st ^= 1;
        reg_ld ^= 1;
    }
    if (num_stages > stage_idx) {
        // prefetch the first tile of the next stage
        copy_async_wait_and_syncthreads<num_stages - 1 - stage_idx>();
        int next_stage = (stage + 1) % num_stages;
        load_register(lane, wi, wj, sA(next_stage), sB(next_stage), rA[reg_st], rB[reg_st], 0);
    }
    // compute the last tile of this stage
    do_mma(rA[reg_ld], rB[reg_ld], rC);
    reg_st ^= 1;
    reg_ld ^= 1;
}

__global__ static void kernel12(GEMM_FUNC_SIGNITURE) {
    extern __shared__ half smem_base[];
    half *sA = smem_base;
    half *sB = &sA[num_stages * bM * bK];

    // m16n8k16 mma
    uint32_t rA[2][num_tM * 4];
    uint32_t rB[2][num_tN * 2];
    uint32_t rC[num_tM * num_tN * 2] = {0};

    int bi = blockIdx.z % 2 ? (gridDim.y - 1 - blockIdx.y) : blockIdx.y;
    int bj = blockIdx.z * block_j_swizzle_factor + blockIdx.x;
    if (bj >= N / bN) {
        return;
    }

    A += bi * bM * K;
    B += bj * bN * K;
    C += bi * bM * N + bj * bN;

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lane = tid % 32;
    const int wi = wid % num_wM;
    const int wj = wid / num_wM;

    // pre-stage
    for (int stage = 0; stage < num_stages - 1; stage++) {
        load_shared_AB(sA(stage), A + stage * bK, K, lane, wid, bM);
        load_shared_AB(sB(stage), B + stage * bK, K, lane, wid, bN);
        copy_async_commit();
    }
    copy_async_wait_and_syncthreads<num_stages - 2>();

    int reg_st = 0;
    int reg_ld = 1;
    load_register(lane, wi, wj, sA(0), sB(0), rA[reg_st], rB[reg_st], 0);
    reg_st ^= 1;
    reg_ld ^= 1;

    // main loop
    for (int stage = num_stages - 1; stage < K / bK; stage++) {
        int comp_stage = (stage - num_stages + 1) % num_stages;
        int copy_stage = stage % num_stages;

        for (int k = tK; k < bK; k += tK) {
            load_register(lane, wi, wj, sA(comp_stage), sB(comp_stage), rA[reg_st], rB[reg_st], k);
            do_mma(rA[reg_ld], rB[reg_ld], rC);
            reg_st ^= 1;
            reg_ld ^= 1;
        }

        load_shared_AB(sA(copy_stage), A + stage * bK, K, lane, wid, bM);
        load_shared_AB(sB(copy_stage), B + stage * bK, K, lane, wid, bN);
        copy_async_commit();
        copy_async_wait_and_syncthreads<num_stages - 2>();

        // prefetch registers for the next compute stage and compute the last tile in this compute stage
        int next_comp_stage = (comp_stage + 1) % num_stages;
        load_register(lane, wi, wj, sA(next_comp_stage), sB(next_comp_stage), rA[reg_st], rB[reg_st], 0);
        do_mma(rA[reg_ld], rB[reg_ld], rC);
        reg_st ^= 1;
        reg_ld ^= 1;
    }

    // compute remaining stages
    post_stage<2>(lane, wi, wj, K, sA, sB, rA, rB, rC, reg_st, reg_ld);
    post_stage<3>(lane, wi, wj, K, sA, sB, rA, rB, rC, reg_st, reg_ld);
    post_stage<4>(lane, wi, wj, K, sA, sB, rA, rB, rC, reg_st, reg_ld);

    __syncthreads(); // don't forget this

    // write back to C
    const int tCi = lane / 4;
    const int tCj = lane % 4;
    half *sC = smem_base;
    half *sCw = sC + wi * wM * (bN + pad_half) + wj * wN;
    for (int i = 0; i < wM; i += tM) {
        for (int j = 0; j < wN; j += tN) {
            uint32_t *rCt = &rC[(i / tM * num_tN + j / tN) * 2];
            *(uint32_t *)&sCw[(i + tCi + 0) * (bN + pad_half) + j + tCj * 2] = rCt[0];
            *(uint32_t *)&sCw[(i + tCi + 8) * (bN + pad_half) + j + tCj * 2] = rCt[1];
        }
    }

    __syncthreads();
    store_shared_C(C, sC, N, lane, wid);
}

void gemm_12_mma_swizzle_opt(GEMM_FUNC_SIGNITURE) {
    assert(M % bM == 0);
    assert(N % bN == 0);
    assert(K % bK == 0);
    static_assert(1 < num_stages && num_stages <= 4);

    if ((float)alpha != 1 || (float)beta != 0) {
        std::cout << "gemm_12_mma_swizzle_opt kernel only supports computing C=A*B (alpha=1, beta=0)\n";
        exit(-1);
    }

    size_t smem_size = std::max(num_stages * (bM + bN) * bK, bM * (bN + pad_half)) * sizeof(half);
    static bool printed = false;
    if (!printed) {
        printed = true;
        std::cout << "dynamic shared memory: " << float(smem_size) / 1024 << " KB" << std::endl;
    }

    cudaFuncSetAttribute(kernel12,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem_size);

    dim3 gridDim(block_j_swizzle_factor, M / bM, CEIL_DIV(N / bN, block_j_swizzle_factor));
    dim3 blockDim(32 * num_wM * num_wN);
    kernel12<<<gridDim, blockDim, smem_size>>>(GEMM_FUNC_PARAM);
}