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