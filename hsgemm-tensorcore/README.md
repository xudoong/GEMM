##  Mixed Precision FP16-Fp32 GEMM on A100 Tensor Core

计算：C=alpha\*A\*B+beta*C。

实现的版本为：A, B, C矩阵均为row-major。基于nvcuda::wmma的kernel仅支持C=A*B。

默认使用的算例大小为4096x4096x4096。

| Kernel             | TFLOPS | 简述                                                         |
| ------------------ | ------ | ------------------------------------------------------------ |
| theoretical max    | 312    | A100理论半精度峰值性能                                       |
| cuBLAS             | 238.8  | cuBLAS                                                       |
| k03_wmma_shmem     | 39.2   | 基于nvcuda::wmma API，并使用了shared memory。                |
| k04_wmma_shmem_opt | 88.7   | 基于k03，128b Load，使用padding来避免bank conflict。cta tile=128x128x32。 |
|                    | 130.3  | cta tile=128x128x64。                                        |
| K05_wmma_pipeline  | 161.9  | 基于k04，使用cuda::pipeline实现了 pipeline global -> shared load，重叠计算和load。(bK=32, stage=2) |
|                    | 150.8  | bK=64, stage=1。                                             |
