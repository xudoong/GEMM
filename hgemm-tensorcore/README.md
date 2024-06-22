##  HGEMM on A100 Tensor Core

计算：C=alpha\*A\*B+beta*C。

实现的版本为：A, C矩阵均为row-major，B矩阵为col-major。

默认使用的算例大小为4096x4096x4096。

| Kernel               | TFLOP（%MAX)  | 简述                                                         |
| -------------------- | ------------- | ------------------------------------------------------------ |
| theoretical max      | 312           | A100理论半精度峰值性能                                       |
| cuBLAS               | 210.7 (67.5%) | cuBLAS                                                       |
| k03_wmma_shmem       | 34.4 (11%)    | 基于nvcuda::wmma API，并使用了shared memory。                |
| k04_wmma_shmem_opt   | 112.5 (36%)   | 基于k03，128b Load，使用padding来避免bank conflict。cta tile=128x128x32。 |
|                      | 143.4 (46%)   | cta tile=128x128x64。                                        |
| k05_wmma_stage       | 150.1 (48.1%) | 基于k04，实现了stage=3的pipeline。cta tile=256x128x32。      |
| k06_wmma_stage_dbreg | 172.1 (55.2%) | 基于k05，对shared memory到register的load进行了double buffer。 |
| k07_mma_padding      | 145.7 (46.7%) | 基于k04，实现了mma版本。cta tile=128x128x64。                |

##### K05_wmma_stage：global memory访问是否进行warp切分

将一个bM x bN大小的块从global memory搬运到shared memory时，可以将整个cta中的所有线程看作一个整体，也可以先将bMxbN的块进行一维的切分，分配给各个warp。实验发现这两个策略对性能有不可忽略的影响：

* A, B, C三个矩阵的global memory访问均不进行warp切分时，性能150.1TFLOPS；

* A, B, C三个矩阵的global memory访问均进行warp切分时，性能145.9TFLOPS；
* A, B矩阵不进行warp切分，C矩阵不进行warp切分时，性能158.5TFLOPS。

产生上述性能影响的原因尚不清楚。

