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

#include "dgemm.cuh"

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

void range_init_matrix(double *mat, int N);
void randomize_matrix(double *mat, int N);
void zero_init_matrix(double *mat, int N);
void copy_matrix(const double *src, double *dest, int N);
void print_matrix(const double *A, int M, int N, std::ofstream &fs);
bool verify_matrix(double *mat1, double *mat2, int N);

double get_current_sec();                        // Get the current moment
double cpu_elapsed_time(double &beg, double &end); // Calculate time difference

void run_kernel(int kernel_num, cublasHandle_t handle, DGEMM_FUNC_SIGNITURE);