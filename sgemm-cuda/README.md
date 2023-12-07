# SGEMM on CUDA GPU

This directory is built upon [siboehm-SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA).

The experiments run on one A100 GPU.

First I evaluted the performance of the kernels from the reference implementation ( [siboehm-SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)). The achieved TFLOPS when M=N=K=4096 is as follows.

|         | K0 cuBLAS | K1 Naive | K2 GMEM Coalescing | K3 SMEM Caching | K4 1D Blocktiling | K5 2D Blocktiling | K6 Vectorized Mem Access | K7 Avoid Bank Conflicts (Linearize) | K8 Avoid Bank Conflicts (Offset) | K9 Autotuning | K10 Warptiling |
| ------- | --------- | -------- | ------------------ | --------------- | ----------------- | ----------------- | ------------------------ | ----------------------------------- | -------------------------------- | ------------- | -------------- |
| TFLOPS  | 18.14     | 0.29     | 3.01               | 5.44            | 10.20             | 14.47             | 15.37                    | 15.28                               | 14.95                            | 15.86         | 16.78          |
| %cuBLAS | 100%      | 1.6%     | 16.6%              | 30%             | 56.2%             | 79.8%             | 84.7%                    | 84.2%                               | 82.4%                            | 87.4%         | 92.5%          |

It can be seen that just using shared memroy plus 2D thread-level tiling could achieve 80% efficiency (relative to cuBLAS).

### Huge performance gap with tiny cuda code difference

Seen the reference kernel could obtain 80% efficiency with merely *shared memory plus 2D tiling*,  I re-implemented this  [kernel](./src/kernels/03_gemm_shmem.cu) by myself,  and found that it only achieved **12TFLOPS** (2.5-3 TFLOPS lower than the ref). 

At the first galance, the computation logic and the hyper parameters are the same as the ref kernel, so I modified my code line by line, to make it be identical with the [ref](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh). 

In this way, I found two code pieces where two seemingly same expression leads to several 100GLFOPS performance gap.

#### 1) The for loop index and offset in loading to shared memory

This code part is for loading A and B tiles from the global memory to the shared memory.

The first version (slow):

```c++
for (int i = offset_A_m; i < BM; i += stride_A_m) {
    A_shared[i * BK + offset_A_k] = A[i * K + offset_A_k];
}
(similar for B_shared)
```

The second version (fast):

```c++
for (int i = 0; i < BM; i += stride_A_m) {
    A_shared[(i + offset_A_m) * BK + offset_A_k] = A[(i + offset_A_m) * K + offset_A_k];
}
(similar for B_shared)
```

The first version is 13.5TFLOPS, and the second verison is 15.5TFLOPS.


#### 2) The computation of thread id

When doing C=A*B computations, the threads are organized as 16x16 grids in a block.

In the first version, the blockDim is 2D in 16x16:

```c++
const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

// inside the gemm kernel
{
	const int tid = threadIdx.x + threadIdx.y * blockDim.x;
}

 ... 
// calling the kernel
dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
kernel_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

In the second version, the blockDim is 1D in 256:

```c++
const int BLOCK_DIM_X = 16;
const int BLOCK_DIM_Y = 16;

// inside the gemm kernel
{
	const int tid = threadIdx.x;
}

 ... 
// calling the kernel
dim3 blockDim(BLOCK_DIM_X * BLOCK_DIM_Y);
kernel_gemm<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

To make these two versions more alike, I make the blockDim be 1D in both versions. Thus they only differ in one line of code:

```c++
// version 1 (fast)
const int tid = threadIdx.x;

// version 2 (slow)
const int tid = threadIdx.x + threadIdx.y * blockDim.x;
```

I verified that without altering the remaining code, the first version is 13.5TFLOPS, and the second version is 15.5TFLOPS (MNK=4096). This is rather astonishing, seeing  2TFLOPS performance gap, and the seeming identical two versions of code. In comparison,  *warp-tiling* couldn't offer 2TFLOPS improvement. 

Combining 1) and 2), the 15.5TFLOPS is achieved only when the fast versions are used for both 1) and 2), while in the other cases, the performance is 13.5TFLOPS.

This may be relevant to the compiler optimizations, partly proves the power of the compiler. I haved compared the ptx and sass assembly code of the two versions, but they appears the same to me. So I couldn't explain the reason for this 2TFLOPS performance gap for now.



