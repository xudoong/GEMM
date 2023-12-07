#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <iomanip>

#include "gemm.cuh"

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(float *mat, int N);
void randomize_matrix(float *mat, int N);
void zero_init_matrix(float *mat, int N);
void copy_matrix(const float *src, float *dest, int N);
void print_matrix(const float *A, int M, int N, std::ofstream &fs);
bool verify_matrix(float *mat1, float *mat2, int N);

double get_current_sec();                        // Get the current moment
double cpu_elapsed_time(double &beg, double &end); // Calculate time difference

void run_kernel(int kernel_num, cublasHandle_t handle, GEMM_FUNC_SIGNITURE);