#!/bin/bash

MKL_FLAGS="-DMKL_ILP64 -m64 -I"${MKLROOT}/include" -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl"
SRC="src/dgemm_mkl.c src/dgemm_naive.c src/dgemm_vec.c src/util.c src/dgemm_mypack.c src/dgemm_ideal.c"

gcc -o main.x -O0 ${MKL_FLAGS} ${SRC} main.c -I. -march=native
gcc -o correctness.x -O0 ${MKL_FLAGS} ${SRC} correctness.c -I. -march=native

