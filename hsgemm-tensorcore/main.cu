#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "utils.cuh"

#define cudaCheck(err) (cudaCheck(err, __FILE__, __LINE__))

const std::string errLogFile = "matrixValidationFailure.txt";

const int theoretical_max_tflops = 312;
int default_size = 4096;
int repeat_times = 50;
int n_warmup = 5;

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: <this-exe> <kernel-number> [[m <m_size>] [n <n_size>] [k <k_size>] [r <repeat_times>]" << std::endl;
        exit(EXIT_FAILURE);
    }

    int m, n, k;
    m = n = k = default_size;
    if (argc > 2) {
        for (int i = 2; i < argc; i++) {
            if (argv[i][0] == 'm' && i < argc - 1) {
                m = std::stoi(argv[i + 1]);
            }
            else if (argv[i][0] == 'n' && i < argc - 1) {
                n = std::stoi(argv[i + 1]);
            }
            else if (argv[i][0] == 'k' && i < argc - 1) {
                k = std::stoi(argv[i + 1]);
            }
            else if (argv[i][0] == 'r' && i < argc - 1) {
                repeat_times = std::stoi(argv[i + 1]);
            }
        }
    }

    // get kernel number
    int kernel_num = std::stoi(argv[1]);

    // get environment variable for device
    int deviceIdx = 0;
    if (getenv("DEVICE") != NULL) {
        deviceIdx = atoi(getenv("DEVICE"));
    }
    cudaCheck(cudaSetDevice(deviceIdx));

    printf("Running kernel %d on device %d.\n", kernel_num, deviceIdx);

    // print some device info
    // CudaDeviceInfo();

    // Declare the handle, create the handle, cublasCreate will return a value of
    // type cublasStatus_t to determine whether the handle was created
    // successfully (the value is 0)
    cublasHandle_t handle;
    if (cublasCreate(&handle)) {
        std::cerr << "Create cublas handle error." << std::endl;
        exit(EXIT_FAILURE);
    };

    // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
    // publishing event tasks in the target stream
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);


//    float alpha = 0.5, beta = 3.0; // GEMM input parameters, C=α*AB+β*C
    float alpha = 1, beta = 0; // GEMM input parameters, C=α*AB+β*C

    half *A = nullptr;
    half *B = nullptr;
    float *C = nullptr;
    float *C_ref = nullptr;
    half *dA = nullptr;
    half *dB = nullptr;
    float *dC = nullptr;
    float *dC_ref = nullptr;

    int max_size = std::max(m, n);
    max_size = std::max(max_size, k);
    A = (half *) malloc(sizeof(half) * max_size * max_size);
    B = (half *) malloc(sizeof(half) * max_size * max_size);
    C = (float *) malloc(sizeof(float) * max_size * max_size);
    C_ref = (float *) malloc(sizeof(float) * max_size * max_size);

    randomize_matrix(A, max_size * max_size);
    randomize_matrix(B, max_size * max_size);
    randomize_matrix(C, max_size * max_size);

    cudaCheck(cudaMalloc((void **) &dA, sizeof(half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dB, sizeof(half) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC, sizeof(float) * max_size * max_size));
    cudaCheck(cudaMalloc((void **) &dC_ref, sizeof(float) * max_size * max_size));

    cudaCheck(cudaMemcpy(dA, A, sizeof(half) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dB, B, sizeof(half) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC, C, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dC_ref, C, sizeof(float) * max_size * max_size,
                         cudaMemcpyHostToDevice));

    printf("Size: (m, n, k) = (%d, %d, %d).\n", m, n, k);
    printf("Repeat %d times.\n", repeat_times);
    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors
    if (kernel_num != 0 && kernel_num != FAKE_KERNEL_NUMBER) {
        run_kernel(0, handle, m, n, k, alpha, dA, dB, beta, dC_ref);      // cuBLAS
        run_kernel(kernel_num, handle, m, n, k, alpha, dA, dB, beta,
                   dC); // Executes the kernel, modifies the result matrix
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
        cudaMemcpy(C, dC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
        cudaMemcpy(C_ref, dC_ref, sizeof(float) * m * n, cudaMemcpyDeviceToHost);

        if (!verify_matrix(C_ref, C, m * n)) {
            std::cout
                    << "Failed to pass the correctness verification against NVIDIA "
                       "cuBLAS."
                    << std::endl;
            if (m <= 128) {
                std::cout << " Logging faulty output into " << errLogFile << "\n";
                std::ofstream fs;
                fs.open(errLogFile);
                fs << "A:\n";
                print_matrix(A, m, n, fs);
                fs << "B:\n";
                print_matrix(B, m, n, fs);
                fs << "C:\n";
                print_matrix(C, m, n, fs);
                fs << "Should:\n";
                print_matrix(C_ref, m, n, fs);
            }
            exit(EXIT_FAILURE);
        }
    }

    // warmup
    for (int j = 0; j < n_warmup; j++) {
        run_kernel(kernel_num, handle, m, n, k, alpha, dA, dB, beta, dC);
    }

    // benchmark region
    std::vector<float> elapsed_times;
    for (int j = 0; j < repeat_times; j++) {
        float elapsed_time;
        cudaEventRecord(beg);
        // We don't reset dC between runs to save time
        run_kernel(kernel_num, handle, m, n, k, alpha, dA, dB, beta, dC);
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_times.push_back(elapsed_time / 1000);  // Convert to seconds
    }
    // report performance

    std::sort(elapsed_times.begin(), elapsed_times.end());
    std::reverse(elapsed_times.begin(), elapsed_times.end());
    float tile05_time = elapsed_times[int(repeat_times * 0.05)];
    float tile50_time = elapsed_times[int(repeat_times * 0.50)];
    float tile95_time = elapsed_times[int(repeat_times * 0.95)];

    float flops = 2.0 * float(m) * float(n) * k;

    float tflops05 = flops / tile05_time / 1e12;
    float tflops50 = flops / tile50_time / 1e12;
    float tflops95 = flops / tile95_time / 1e12;

    printf("TFLOPS: %5.1f (5%%)  %5.1f (50%%)  %5.1f (95%%). %.1f%% of theoretical.\n",
           tflops05, tflops50, tflops95, tflops95 / theoretical_max_tflops * 100);
    // make dC and dC_ref equal again (we modified dC while calling our kernel
    // for benchmarking)
    cudaCheck(cudaMemcpy(dC, dC_ref, sizeof(float) * m * n,
                         cudaMemcpyDeviceToDevice));

    // Free up CPU and GPU space
    free(A);
    free(B);
    free(C);
    free(C_ref);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    cudaFree(dC_ref);
    cublasDestroy(handle);

    return 0;
};