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

void randomize_matrix(float *mat, int N)
{
    // NOTICE: Use gettimeofday instead of srand((unsigned)time(NULL)); the time
    // precision is too low and the same random number is generated.
    struct timeval time
    {
    };
    gettimeofday(&time, nullptr);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++)
    {
        float tmp = (float)(rand() % 5) + 0.01 * (rand() % 5);
        tmp = (rand() % 2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void range_init_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = i;
    }
}

void zero_init_matrix(float *mat, int N)
{
    for (int i = 0; i < N; i++)
    {
        mat[i] = 0.0;
    }
}

void copy_matrix(const float *src, float *dest, int N)
{
    int i;
    for (i = 0; src + i && dest + i && i < N; i++)
        *(dest + i) = *(src + i);
    if (i != N)
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
}

void print_matrix(const float *A, int M, int N, std::ofstream &fs)
{
    int i;
    fs << std::setprecision(2)
       << std::fixed; // Set doubleing-point precision and fixed notation
    fs << "[";
    for (i = 0; i < M * N; i++)
    {
        if ((i + 1) % N == 0)
            fs << std::setw(5) << A[i]; // Set field width and write the value
        else
            fs << std::setw(5) << A[i] << ", ";
        if ((i + 1) % N == 0)
        {
            if (i + 1 < M * N)
                fs << ";\n";
        }
    }
    fs << "]\n";
}

bool verify_matrix(float *matRef, float *matOut, int N)
{
    double diff = 0.0;
    int i;
    for (i = 0; i < N; i++)
    {
        diff = std::fabs(matRef[i] - matOut[i]);
        if (diff > 0.01)
        {
            printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
                   matRef[i], matOut[i], diff, i);
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
        gemm_01_naive(GEMM_FUNC_PARAM);
        break;
    case 2:
        gemm_02_ideal(GEMM_FUNC_PARAM);
        break;
    case 3:
        gemm_03_shmem(GEMM_FUNC_PARAM);
        break;
    case 4:
        gemm_04_warp_tiling(GEMM_FUNC_PARAM);
        break;
    case 5:
        gemm_05_shmem_plus_vectorize(GEMM_FUNC_PARAM);
        break;
    case 6:
        gemm_06_cutlass(GEMM_FUNC_PARAM);
        break;
    case 7:
        gemm_07_rm_bank_conflict(GEMM_FUNC_PARAM);
        break;
    default:
        throw std::invalid_argument("Unknown kernel number");
    }
}