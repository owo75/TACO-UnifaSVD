#ifndef VERIFICATION_H
#define VERIFICATION_H

#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <cstddef> 

template<typename T>
void verify_Q_matrix(
    cublasHandle_t handle,
    size_t m,
    size_t n,
    size_t rank,
    thrust::device_vector<T>& d_A,
    thrust::device_vector<T>& d_Q
);

template<typename T>
void verify_rsvd_final(
    cublasHandle_t handle,
    size_t m, size_t n, size_t rank,
    thrust::device_vector<T>& d_A,
    thrust::device_vector<T>& d_U,
    thrust::device_vector<T>& d_S_vec, // 奇异值是向量
    thrust::device_vector<T>& d_V
);

template<typename T>
void verify_C_svd(
    cublasHandle_t handle,
    size_t rows, // C 的行数
    size_t cols, // C 的列数
    size_t k,    // 奇异值的数量
    const thrust::device_vector<T>& d_C_original,
    const thrust::device_vector<T>& d_U_c,
    const thrust::device_vector<T>& d_S_c,
    const thrust::device_vector<T>& d_VT_c
);

template<typename T>
void verify_full_reconstruction(
    cublasHandle_t handle,
    size_t m,
    size_t n,
    size_t final_rank, // Q的列数
    size_t k_final,    // 奇异值的数量
    const thrust::device_vector<T>& d_A_original,
    const thrust::device_vector<T>& d_Q,
    const thrust::device_vector<T>& d_U_c,
    const thrust::device_vector<T>& d_S_c,
    const thrust::device_vector<T>& d_VT_c
);

#endif