##  Mixed Precision FP16-Fp32 GEMM on A100 Tensor Core

计算：C=alpha\*A\*B+beta*C。

实现的版本为：A, B, C矩阵均为row-major。基于nvcuda::wmma的kernel仅支持C=A*B。

默认使用的算例大小为4096x4096x4096。

| Kernel             | TFLOPS    | 简述                                                         |
| ------------------ | --------- | ------------------------------------------------------------ |
| theoretical max    | 312       | A100理论半精度峰值性能                                       |
| cuBLAS             | 238.8     | cuBLAS                                                       |
| k03_wmma_shmem     | 39.2      | 基于nvcuda::wmma API，并使用了shared memory。                |
| k04_wmma_shmem_opt | 88.7      | 基于k03，128b Load，使用padding来避免bank conflict。cta tile=128x128x32。 |
|                    | 130.3     | cta tile=128x128x64。                                        |
| k05_wmma_pipeline  | 164.3     | 基于k04，使用cuda::pipeline实现了 pipeline global -> shared load，重叠计算和load。(bK=32, stage=2) |
|                    | 150.8     | bK=64, stage=1。                                             |
|                    | ~~194.5~~ | 伪kernel，注释掉了gmem->smem的代码，可作为upper bound的参考。(bK=32, stage=2) |
| k06_mma            | 143.4     | 基于k05，使用PTX mma, ldmatrix等实现gemm。                   |

#### kernel 05 bottleneck分析

通过注释或修改代码，来制造理想情景，从而得到理论上的性能上限。例如，注释掉cp_asyc代码来获得消除global memory到shared memory load情况下的计算性能。

|                                     | TFLOPS （%MAX) |
| ----------------------------------- | -------------- |
| Baseline                            | 52.7%          |
| load A only                         | 66.7%          |
| load B only                         | 65.6%          |
| rm bank conflict in load gmem->smem | 63.4%          |
| skip smem->reg+comp                 | 56.6%          |

以上数据说明当前的kernel瓶颈主要在global memory到shared memory的load。当前的实现存在store shared bank conflict，如果消除的话则预期能有10%Max TFLOPS的性能上升。

目前的实现使用了padding来减少bank conflict，但并不能完全消除。cuda官方建议使用permuted layout。由于wmma的load_fragment要求矩阵为规则的row-major或col-major，不适用于permuted layout，因此只能在mma版本的kernel中实现。
