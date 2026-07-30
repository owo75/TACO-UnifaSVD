// cuda_checks.h
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <curand.h>

// 把状态码转成可读字符串
inline const char* cublasGetStatusString(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
    #if defined(CUBLAS_STATUS_NOT_SUPPORTED)
        case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
    #endif
    #if defined(CUBLAS_STATUS_LICENSE_ERROR)
        case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
    #endif
        default: return "CUBLAS_STATUS_<unknown>";
    }
}

inline const char* cusolverGetStatusString(cusolverStatus_t s) {
    switch (s) {
        case CUSOLVER_STATUS_SUCCESS: return "CUSOLVER_STATUS_SUCCESS";
        case CUSOLVER_STATUS_NOT_INITIALIZED: return "CUSOLVER_STATUS_NOT_INITIALIZED";
        case CUSOLVER_STATUS_ALLOC_FAILED: return "CUSOLVER_STATUS_ALLOC_FAILED";
        case CUSOLVER_STATUS_INVALID_VALUE: return "CUSOLVER_STATUS_INVALID_VALUE";
        case CUSOLVER_STATUS_ARCH_MISMATCH: return "CUSOLVER_STATUS_ARCH_MISMATCH";
        case CUSOLVER_STATUS_EXECUTION_FAILED: return "CUSOLVER_STATUS_EXECUTION_FAILED";
        case CUSOLVER_STATUS_INTERNAL_ERROR: return "CUSOLVER_STATUS_INTERNAL_ERROR";
    #if defined(CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED: return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    #endif
        default: return "CUSOLVER_STATUS_<unknown>";
    }
}

inline const char* curandGetStatusString(curandStatus_t s) {
    switch (s) {
        case CURAND_STATUS_SUCCESS: return "CURAND_STATUS_SUCCESS";
        case CURAND_STATUS_VERSION_MISMATCH: return "CURAND_STATUS_VERSION_MISMATCH";
        case CURAND_STATUS_NOT_INITIALIZED: return "CURAND_STATUS_NOT_INITIALIZED";
        case CURAND_STATUS_ALLOCATION_FAILED: return "CURAND_STATUS_ALLOCATION_FAILED";
        case CURAND_STATUS_TYPE_ERROR: return "CURAND_STATUS_TYPE_ERROR";
        case CURAND_STATUS_OUT_OF_RANGE: return "CURAND_STATUS_OUT_OF_RANGE";
        case CURAND_STATUS_LENGTH_NOT_MULTIPLE: return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
        case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
        case CURAND_STATUS_LAUNCH_FAILURE: return "CURAND_STATUS_LAUNCH_FAILURE";
        case CURAND_STATUS_PREEXISTING_FAILURE: return "CURAND_STATUS_PREEXISTING_FAILURE";
        case CURAND_STATUS_INITIALIZATION_FAILED: return "CURAND_STATUS_INITIALIZATION_FAILED";
        case CURAND_STATUS_ARCH_MISMATCH: return "CURAND_STATUS_ARCH_MISMATCH";
        case CURAND_STATUS_INTERNAL_ERROR: return "CURAND_STATUS_INTERNAL_ERROR";
        default: return "CURAND_STATUS_<unknown>";
    }
}

#define CHECK_CUDA(call) do {                                   \
    cudaError_t _e = (call);                                     \
    if (_e != cudaSuccess) {                                     \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n",          \
            cudaGetErrorString(_e), __FILE__, __LINE__, #call);  \
        std::exit(EXIT_FAILURE);                                 \
    }                                                            \
} while (0)

#define CUDA_CHECK(call) CHECK_CUDA(call)

#define CUBLAS_CHECK(call) do {                                  \
    cublasStatus_t _s = (call);                                   \
    if (_s != CUBLAS_STATUS_SUCCESS) {                            \
        fprintf(stderr, "cuBLAS error %s at %s:%d: %s\n",         \
            cublasGetStatusString(_s), __FILE__, __LINE__, #call);\
        std::exit(EXIT_FAILURE);                                  \
    }                                                             \
} while (0)

#define CUSOLVER_CHECK(call) do {                                 \
    cusolverStatus_t _s = (call);                                  \
    if (_s != CUSOLVER_STATUS_SUCCESS) {                           \
        fprintf(stderr, "cuSOLVER error %s at %s:%d: %s\n",        \
            cusolverGetStatusString(_s), __FILE__, __LINE__, #call);\
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)

#define CURAND_CHECK(call) do {                                   \
    curandStatus_t _s = (call);                                    \
    if (_s != CURAND_STATUS_SUCCESS) {                             \
        fprintf(stderr, "cuRAND error %s at %s:%d: %s\n",          \
            curandGetStatusString(_s), __FILE__, __LINE__, #call); \
        std::exit(EXIT_FAILURE);                                   \
    }                                                              \
} while (0)
