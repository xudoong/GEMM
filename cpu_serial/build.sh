#!/bin/bash

MKL_FLAGS="-DMKL_ILP64 -m64 -I"${MKLROOT}/include" -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
OPENBLAS_FLAGS="-lopenblas"

BLAS_FLAGS=${MKL_FLAGS}

SRC="src/dgemm_mkl.c src/util.c src/dgemm_v3.c src/dgemm_v4.c"
# SRC="src/dgemm_openblas.c src/util.c src/dgemm_v3.c src/dgemm_v4.c"

gcc -o main.x -O3 ${BLAS_FLAGS} ${SRC} main.c -I. -march=native
gcc -o correctness.x -O3 ${BLAS_FLAGS} ${SRC} correctness.c -I. -march=native

