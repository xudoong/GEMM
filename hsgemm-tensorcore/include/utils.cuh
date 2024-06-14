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
#include <algorithm>

#include "gemm.cuh"

void cudaCheck(cudaError_t error, const char *file,
               int line); // CUDA error check
void CudaDeviceInfo();    // print CUDA information

template <typename T>
void randomize_matrix(T *mat, int N)
{
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time;
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++)
    {
        float tmp = float(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = static_cast<T>(tmp);
    }
}

template <typename T>
void zero_init_matrix(T *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = static_cast<T>(0);
    }
}

template <typename T>
void copy_matrix(const T *src, T *dest, int N)
{
    for (int i = 0; i < N; i++)
        *(dest + i) = *(src + i);
}

template <typename T>
void print_matrix(const T *A, int M, int N, std::ofstream &fs)
{
    int i;
    fs << std::setprecision(2)
       << std::fixed; // Set doubleing-point precision and fixed notation
    fs << "[";
    for (i = 0; i < M * N; i++)
    {
        if ((i + 1) % N == 0)
            fs << std::setw(5) << float(A[i]); // Set field width and write the value
        else
            fs << std::setw(5) << float(A[i]) << ", ";
        if ((i + 1) % N == 0)
        {
            if (i + 1 < M * N)
                fs << ";\n";
        }
    }
    fs << "]\n";
}

bool verify_matrix(float *mat1, float *mat2, int N);

double get_current_sec();                        // Get the current moment
double cpu_elapsed_time(double &beg, double &end); // Calculate time difference

void run_kernel(int kernel_num, cublasHandle_t handle, GEMM_FUNC_SIGNITURE);