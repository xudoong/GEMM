#!/bin/bash

MKL_FLAGS="-DMKL_ILP64 -m64 -I"${MKLROOT}/include" -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm -ldl"

SRC="src/dgemm_omp_mkl.c src/dgemm_omp_v1.c"

gcc -o main.x -O3 ${MKL_FLAGS} ${SRC} main.c -I. -march=native -fopenmp
gcc -o correctness.x -O3 ${MKL_FLAGS} ${SRC} correctness.c -I. -march=native -fopenmp
