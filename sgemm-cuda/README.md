# SGEMM on CUDA GPU

This directory is built upon [siboehm-SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA).

The experiments run on one A100 GPU.

### MY result

|                         | **Median TFLOPS** | **STD** | **%cuBLAS** |
| ----------------------- | ----------------- | ------- | ----------- |
| **K0 cuBLAS**           | 17.6              | 0.41    | 100%        |
| **K1** **ideal (FAKE)** | 18.6              | -       | --          |
| **K3 shmem**            | 15.8              | 0.38    | 90%         |
| **K4 warptiling**       | 15.1              | 0.4     | 86%         |
| **K5 shmem+vectorize**  | 15.9              | 0.46    | 90%         |
| **K6 cutlass (ref)**    | 16.1              | 0.52    | 91%         |

The performance of the simple kernel 03 (shared memory plus 2D tiling on threads) is very close to that of cutlass.
Adding more optimizations to Kernel 03 doesn't provide speedup.

**Re-timing using Nsight**

|                    | TFLOPS | %theoretical |
| ------------------ | ------ | ------------ |
| K0 cuBLAS          | 19.1   | 97.85%       |
| K6 cutlass         | 17.5   | 89.62%       |
| K3 shmem           | 16.7   | 85.70%       |
| K4 warptiling      | 16.6   | 85.24%       |
| K5 shmem+vectorize | 17     | 87.15%       |

From the result from Nsight, K5 does provide speed up over K3, but K4 still doesn't. Moreover, different from the timing result from `cudaEventRecord`, which has a std deviation up to 400GFLOPS, the result in Nsight is very stable, and the durating for each kernel are shorter.

### REF result

First I evaluted the performance of the kernels from the reference implementation ( [siboehm-SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA)). The achieved TFLOPS when M=N=K=4096 is as follows. Due to the performance variation, each kernel test is repeated 30 times, and the meidan and std TFLOPS are reported.

|                                     | Median TFLOPS | Std  | %cuBLAS |
| ----------------------------------- | ------------- | ---- | ------- |
| K0 cuBLAS                           | 17.6          | 0.41 | 100%    |
| K5 2D Blocktiling                   | 13.4          | 0.31 | 76%     |
| K6 Vectorized Mem Access            | 14.2          | 0.4  | 81%     |
| K7 Avoid Bank Conflicts (Linearize) | 14.3          | 0.36 | 81%     |
| K8 Avoid Bank Conflicts (Offset)    | 13.75         | 0.41 | 78%     |
| K9 Autotuning                       | 14.9          | 0.41 | 85%     |
| K10 Warptiling                      | 16.2          | 0.48 | 92%     |

It can be seen that just using shared memroy plus 2D thread-level tiling could achieve 80% efficiency (relative to cuBLAS).

### Huge performance gap with tiny cuda code difference

Seen the reference kernel could obtain nearly 80% efficiency with merely *shared memory plus 2D tiling*,  I re-implemented this  [kernel](./src/kernels/03_gemm_shmem.cu) by myself,  and found that it only achieved **12TFLOPS** (~2 TFLOPS lower than the ref). 

At the first galance, the computation logic and the hyper parameters are the same as the ref kernel, so I modified my code line by line, to make it be identical with the [ref](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/5_kernel_2D_blocktiling.cuh). 

In this way, I found two code pieces where two seemingly same expression leads to several hunderds GLFOPS performance gap.

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

This may be relevant to the compiler optimizations, partly proving the power of the compiler. I haved compared the *ptx* and *sass* assembly code of the two versions, and found that in version1 the FMA loop over *k*  is autometicly unrolled while in version 2 it is not, but after unrolling this loop in version1 using `pragma` unroll, there is still ~1TFLOPS performance gap, and I couldn't observe other noteable difference between the two sass code.