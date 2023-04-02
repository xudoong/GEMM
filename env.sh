#!/bin/bash

module load gcc/9.3.0
module load intel-oneapi-mkl/2022.1.0

# openblas env
OPENBLAS_ROOT=/lustre/home/acct-hpc/asc/XDWang/packages/OpenBLAS/install
export CPATH=${OPENBLAS_ROOT}/include:${CPATH}
export LD_LIBRARY_PATH=${OPENBLAS_ROOT}/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=${OPENBLAS_ROOT}/lib:${LIBRARY_PATH}
