# DGEMM on Intel CPU with OMP Threading
This directory implements OpenMP threaded DGEMM on Intel CPU, and the achieved GFLOPS is compared to MKL.

## Platform
Our platform is a single node Inel Xeon Cascade Lake @2.5GHz, with the following configuations:
* 2 socket with 20 cores per socket
* L1D 32K per core, L2 1M per core, L3 27.5M (20x1.375M)
* DRAM: DDR4-2933, 6 channels per socket. theoretical Bandwidth 137.6GB/s
* GCC 9.3.0, Intel OneAPI MKL 2022.1.0

**Note**: For simplicity, we only use a single socket thereby avoid bothering NUMA behavior.

## MKL Performance
We fix M=N=K=3840 and iterate #threads from 1 to 20. The result is as follows:
| #cores     | 1  | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   | 10  | 11  | 12   | 13   | 14   | 15   | 16   | 17   | 18   | 19   | 20   |
|------------|----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|------|------|------|------|------|------|------|------|------|
| GFLOPS     | 97 | 217 | 316 | 420 | 514 | 616 | 717 | 808 | 794 | 874 | 965 | 1039 | 1022 | 1104 | 1175 | 1213 | 1230 | 1288 | 1330 | 1308 |
| Efficiency | 80 | 89  | 91  | 91  | 92  | 92  | 91  | 90  | 92  | 91  | 91  | 90   | 91   | 91   | 91   | 88   | 90   | 89   | 87   | 82   |

The last row "Efficiency" is the percentage between the achieved GFLOPS and the max theoretical GFLOPS. It can be seen that MKL achieves very high efficiency (87%-92%) when #cores ranges from 2 to 19. A noteable thing is that when #cores is 1 or 20, the efficiency is only around 80%, nearly degrading 10%.

Note that the max frequency varies when #cores changes with AVX-512 on. Therefore we can not directly compare the efficiency between different cores.

## V1
The most straight forward way to omp parallize the serial code is to split the Matrix C. Each thread is responsable for a submatrix of C. The result is as follows: (M=N=K=3840)
| #cores     | 1  | 2  | 4  | 8  | 16 |
|------------|----|----|----|----|----|
| Efficiency | 84 | 84 | 80 | 74 | 73 |

The efficiency degrades when #cores increases. This is probably because this strategy push too much pressure on the shared resources: L3 and memory: the whole A, B, C matrix is read and written in parallel. (*But a question is: is memory bandwidth really a bottleneck? If not, then the performance degration must due to the memory latency. Can we hide this?*)

## V2
This version is a more fine-grained parallel strategy. During the phase to process 8192x384 A and 384xN B, the serial code (v4) split the N dimension into length 192 tiles and iterate among these tiles. This version to parallize is to dispatch thest tiles to the threads. The result is as follows: (M=N=K=3840)
| #cores     | 1  | 2  | 4  | 10 | 18 | 19 | 20 |
|------------|----|----|----|----|----|----|----|
| Efficiency | 85 | 80 | 80 | 81 | 47 | 83 | 78 |

It can be seen that the efficiency scales better than V1. But a severe problem is that there are not so much work to dispatch (3840/192 - 1 = 19), which results in work unbalance and leads to the performance drop when #cores=18.

