#include "utils.cuh"

double get_sec()
{
    struct timeval time;
    gettimeofday(&time, NULL);
    return (1e6 * time.tv_sec + time.tv_usec);
}

double cpu_elapsed_time(double &beg, double &end) { return 1.0e-6 * (end - beg); }

void cudaCheck(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

void CudaDeviceInfo()
{
    int deviceId;

    cudaGetDevice(&deviceId);

    cudaDeviceProp props{};
    cudaGetDeviceProperties(&props, deviceId);

    printf("Device ID: %d\n\
        Name: %s\n\
        Compute Capability: %d.%d\n\
        memoryBusWidth: %d\n\
        maxThreadsPerBlock: %d\n\
        maxThreadsPerMultiProcessor: %d\n\
        maxRegsPerBlock: %d\n\
        maxRegsPerMultiProcessor: %d\n\
        totalGlobalMem: %zuMB\n\
        sharedMemPerBlock: %zuKB\n\
        sharedMemPerMultiprocessor: %zuKB\n\
        totalConstMem: %zuKB\n\
        multiProcessorCount: %d\n\
        Warp Size: %d\n",
           deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
           props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
           props.regsPerBlock, props.regsPerMultiprocessor,
           props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
           props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
           props.multiProcessorCount, props.warpSize);
};

bool verify_matrix(half *matRef, half *matOut, int N)
{
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++)
    {
        diff = fabs(matRef[i] - matOut[i]);
        float max_abs = std::max(fabs(matRef[i]), fabs(matOut[i]));
        auto rerr = diff / (max_abs + 1e-8);
        if (diff > 0.1 || rerr > 0.1 || std::isnan(float(matOut[i])))
        {
            printf("Divergence! Should %5.2f, Is %5.2f (diff %5.2f, rerr %5.2f) at %d\n",
                   matRef[i], matOut[i], diff, rerr, i);
            return false;
        }
    }
    return true;
}

int div_ceil(int numerator, int denominator)
{
    std::div_t res = std::div(numerator, denominator);
    return res.rem ? (res.quot + 1) : res.quot;
}

void run_kernel(int kernel_num, cublasHandle_t handle, GEMM_FUNC_SIGNITURE)
{
    switch (kernel_num)
    {
        case 0:
            gemm_00_cublas(handle, GEMM_FUNC_PARAM);
            break;
        case 1:
            gemm_01_fake(GEMM_FUNC_PARAM);
            break;
        case 2:
            gemm_02_naive(GEMM_FUNC_PARAM);
            break;
        case 3:
            gemm_03_wmma_shmem(GEMM_FUNC_PARAM);
            break;
        case 4:
            gemm_04_wmma_shmem_opt(GEMM_FUNC_PARAM);
            break;
       case 5:
            gemm_05_wmma_stage(GEMM_FUNC_PARAM);
            break;
       case 6:
            gemm_06_wmma_stage_dbreg(GEMM_FUNC_PARAM);
            break;
        default:
            std::cout << "Error: invalid kernel number " << kernel_num << std::endl;
            throw std::invalid_argument("Unknown kernel number");
    }
}