#include "gemm.cuh"
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

void gemm_06_cutlass(GEMM_FUNC_SIGNITURE) {
    // cublas uses column-major order, while we use row-major order.
    // So we compute C^T=alpha * B^T * A^T + beta * C^T.
    // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N,
    //     A, K, &beta, C, N);

    // Define the GEMM operation
    using Gemm = cutlass::gemm::device::Gemm<
        float,                                     // ElementA
        cutlass::layout::ColumnMajor,              // LayoutA
        float,                                     // ElementB
        cutlass::layout::ColumnMajor,              // LayoutB
        float,                                     // ElementOutput
        cutlass::layout::ColumnMajor,              // LayoutOutput
        float,                                     // ElementAccumulator
        cutlass::arch::OpClassSimt,                // tag indicating OpClass
        cutlass::arch::Sm80                        // tag indicating target GPU compute architecture
    >;

    Gemm gemm_op;
    cutlass::Status status;

    status = gemm_op({
        {N, M, K},
        {B, N},            // TensorRef to A device tensor
        {A, K},            // TensorRef to B device tensor
        {C, N},            // TensorRef to C device tensor
        {C, N},            // TensorRef to D device tensor - may be the same as C
        {alpha, beta}           // epilogue operation arguments
    });
}
