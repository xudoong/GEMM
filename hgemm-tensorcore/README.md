##  HGEMM on A100 Tensor Core

计算：C=alpha\*A\*B+beta*C。

实现的版本为：A, C矩阵均为row-major，B矩阵为col-major。

默认使用的算例大小为4096x4096x4096。

| Kernel             | TFLOP（%MAX)  | 简述                                                         |
| ------------------ | ------------- | ------------------------------------------------------------ |
| theoretical max    | 312           | A100理论半精度峰值性能                                       |
| cuBLAS             | 210.7 (67.5%) | cuBLAS                                                       |
| k03_wmma_shmem     | 34.4 (11%)    | 基于nvcuda::wmma API，并使用了shared memory。                |
| k04_wmma_shmem_opt | 112.5 (36%)   | 基于k03，128b Load，使用padding来避免bank conflict。cta tile=128x128x32。 |
|                    | 143.4 (46%)   | cta tile=128x128x64。                                        |

