# Matmul
implementation of matrix multiplication on intel processors

# Result
MKL: 89.5GFLOPs, 73.6%

Initial AVX-512: 2.1GFLOPs, 1.7%

## ideal case
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
This can achieve 74%~110% of the max GFLOPs. Also loop unrooling is useful: 50% (unrool 4), 15% (unrool 1).

## MYPACK
8x8x8 kernel: 13%; 16x8x8 kernel: 18%; 32x8x8 kernel: 19%~20%; 16x8x16 kernel: 20%;