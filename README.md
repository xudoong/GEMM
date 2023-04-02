# Matmul
implementation of matrix multiplication on intel processors

# Result
MKL: 102.4 GFLOPs, 84%


## The ideal case
The ideal case is a tight loop with only AVX-512 FMADD instructions, without any memory references.
```
.L8:
	vfmadd132pd	%zmm0, %zmm0, %zmm1
	vfmadd132pd	%zmm0, %zmm0, %zmm2
	vfmadd132pd	%zmm0, %zmm0, %zmm3
	vfmadd132pd	%zmm0, %zmm0, %zmm4
	vfmadd132pd	%zmm0, %zmm0, %zmm5
	vfmadd132pd	%zmm0, %zmm0, %zmm6
	vfmadd132pd	%zmm0, %zmm0, %zmm7
	vfmadd132pd	%zmm0, %zmm0, %zmm8
	nop
	addl	$64, %edx
.L6:

	cmpl	%eax, %edx
	jl	.L8
```
This can achieve 74%~110% of the max GFLOPs. The unstable result may be due to the frequency variation.

Also this test also shows loop unrooling is useful: 50% (unrool 4), 15% (unrool 1).

## Vectorize + Pack (MYPACK)

16x8x32 / 16x8x64 kernel: 37%~44%.

## V1

The implementation form is borrowed from OpenBLAS. 
* The outer loop is over tiles of C
* The kernel is of size m = 8, n = 24, k = K. It calculates the 8x24 block of C, iterating over all K dim.

The result is better than MYPACK (former version), with efficiency 68~72% when (MNK=1024x960x1024).

TODO: Support any matrix size (currently only support M a multiple of 8 and N a multiple of 24).

Add prefetch: 70%~74%.

Can not scale well when M or K grow. But scale well when N grows. So need to tile over M & K (e.g. 1024 * 1024). The objective is to make tiled A fit in the cache size: L2 1MB, L3 1.375MB.

Outperm MKL when K is large and M is small. Performance degrades when the opposite happens: M is large and K is small or N is small (N < 256). This suggest to change the implementation style (e.g. dot-product or out-product) according to the relative size of MNK.
* When K is large, choose dot-product style (this version);
* When M,N is large and K is small, choose out-product style.

## V2

Optimize packing from V1.
* Delete C buffer. The kernel directly write result to C.
* Split the packing of A and B into block pieces and make them close to the kernel call.

Result: 99 GFLOPs (81%) when (MNK=1024x960x1024).

## V3

Tiling + V2. Tile Size (TM, TN, TK) = (1024, 1920, 1024).

Result compare with MKL, square size (M=N=K):

| MNK  | 240 | 384 | 480 | 768 | 960 | 1200 | 1440 | 1920 | 2400 | 3840 |
|------|-----|-----|-----|-----|-----|------|------|------|------|------|
| MKL  | 34  | 70  | 78  | 77  | 84  | 81   | 86   | 80   | 84   | 85   |
| V3   | 68  | 70  | 68  | 77  | 82  | 75   | 76   | 80   | 77   | 81   |
| V3.1 | 68  | 70  | 78  | 80  | 81  | 79   | 79   | 80   | 79   | 82   |

The value is the achieved GFLOPs represented in the format of the percentage of MAX theoretical GFLOPs (3.8GHz * 2FMadd * 2 AVX-512 * 512 / 64 = 121.6GHz).

From the table it can be seen that when there are small marginal tiles, the performance is bad (1200, 2400). So a further optimization is to choose the tile size based on the input MNK value.

Update: row V3.1 is the result after adjusting the tile size based on input MNK size. 

## V4

Updates from V3:
* Tiling based on OpenBLAS: smaller M (=192) and K (=384);
* The buffer size is no longer equal to input size. Instead the max buffer size is tile A/B size.
* The inner tile loop on N is splited to two steps. An extra first step (tn = 0)is added to load tile A to buffer.

| MNK  | 240 | 384 | 480 | 768 | 960 | 1200 | 1440 | 1920 | 2400 | 3840 |
|------|-----|-----|-----|-----|-----|------|------|------|------|------|
| V4   | 34  | 93  | 78  | 77  | 75  | 75   | 77   | 80   | 77   | 75   |

From the above table it seems V4 is no better than V3. But V3 performs poorly when M,N is far from square shape, especially when M,K are large. In this case A is read many times from memory.
* (M, N, K)=(256,3840,1024): (V3, V4)=(62%, 68%)
* (M, N, K)=(4096,384,1024): (V3, V4)=(67%, 72%)
* (M, N, K)=(4096,384,2048): (V3, V4)=(63%, 71%)
* (M, N, K)=(4096,384,4096): (V3, V4)=(49%, 70%)
