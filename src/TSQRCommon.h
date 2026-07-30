#pragma once
#include <cuda_runtime_api.h>

#define TSQR_BLOCK_SIZE 256
#define TSQR_BLOCK_DIM_Y 32
#define TSQR_BLOCK_DIM_X 32
#define TSQR_NUM_DATA_ROW 8

#ifndef TSQR_SHARED_MEMORY_HELPER_DEFINED
#define TSQR_SHARED_MEMORY_HELPER_DEFINED
template <typename T>
struct shared_memory;
template <>
struct shared_memory<float> {
    __device__ static float *get_pointer() {
        extern __shared__ float shared_mem_float[];
        return shared_mem_float;
    }
};
template <>
struct shared_memory<double> {
    __device__ static double *get_pointer() {
        extern __shared__ double shared_mem_double[];
        return shared_mem_double;
    }
};
#endif

template <typename T>
static __inline__ __device__ T warpAllReduceSum(T val) {
    for (int mask = warpSize / 2; mask > 0; mask /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}
