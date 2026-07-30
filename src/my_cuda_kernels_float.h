#ifndef MY_CUDA_KERNELS_FLOAT_H  
#define MY_CUDA_KERNELS_FLOAT_H  
#include <thrust/device_vector.h> 
#include <thrust/copy.h>         

#include <vector>     
#include <cstdio>    
#include <algorithm>  
#include <cmath>      
#include <type_traits> 
#include <cublas_v2.h> 
#include <curand.h>    
#include <cusolverDn.h> 

__global__ void scale_columns_kernel(
    float* M_out, 
    const float* M_in, 
    const float* vec, 
    size_t rows, 
    size_t cols, 
    size_t ld_out, 
    size_t ld_in
)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        // 获取缩放因子
        float scale_factor = vec[col];
        
        // 执行缩放
        M_out[col * ld_out + row] = M_in[col * ld_in + row] * scale_factor;
    }
}

template <typename T>
void generateNormalMatrix(curandGenerator_t& gen, thrust::device_vector<T>& d_vec, size_t m, size_t n)
{
    // 从device_vector中提取原始指针
    T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());

    // 使用 if constexpr 在编译时选择正确的cuRAND函数
    if constexpr (std::is_same_v<T, float>) 
    {
        curandGenerateNormal(gen, raw_ptr, m * n, 0.0f, 1.0f);
    } 
    else if constexpr (std::is_same_v<T, double>) 
    {
        curandGenerateNormalDouble(gen, raw_ptr, m * n, 0.0, 1.0);
    }
}

// 从 R 块中提取对角线元素
__global__ void extractDiagonal(float* diag_vector, const float* R_block, size_t ld_R, size_t ost, size_t block_width) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < block_width) {
        diag_vector[i] = R_block[(i + ost) * ld_R + (i + ost)];
    }
}

// 默认在对角线元素上原子地或直接地加上1.0，也可以通过参数 alpha 来控制加的值
__global__ void add_identity_diagonal_kernel(
    float* matrix, 
    long int rows, 
    long int cols, 
    long int lda,
    float alpha = 1.0f // **新增参数，并提供默认值**
) {
    long int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // 确保线程索引在对角线范围内
    if (i < rows && i < cols) {
        // 将标量 alpha 加到对角线元素上
        matrix[i * lda + i] += alpha;
    }
}

//提取L下三角矩阵
__global__ void extract_L_factor_kernel(
    float*       L_out, 
    long int      lda_out,
    const float* LU_in, 
    long int      lda_in,
    long int      rows, 
    long int      cols
) {
    long int c = blockIdx.x * blockDim.x + threadIdx.x;
    long int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        long int out_idx = c * lda_out + r;
        
        // L因子只存在于前 min(rows, cols) 列
        if (c >= rows) { // 如果是矮胖矩阵，L因子不存在于这些列中
            L_out[out_idx] = (r == c) ? 1.0f : 0.0f; // 对角线为1，其余为0
            return;
        }

        if (r < c) {
            L_out[out_idx] = 0.0f;
        } 
        else if (r == c) {
            L_out[out_idx] = 1.0f;
        } 
        else { // r > c
            long int in_idx = c * lda_in + r;
            L_out[out_idx] = LU_in[in_idx];
        }
    }
}

// 【新增】自定义CUDA Kernel: 根据索引列表从源矩阵选择列，写入目标矩阵
// 用途: Q_filtered = Q_blk(:, keep_indices)
__global__ void select_columns_kernel(
    float*       dest,           // 目标矩阵指针 (紧凑)
    const int     ld_dest,        // 目标矩阵的leading dimension
    const float* src,            // 源矩阵指针
    const int     ld_src,         // 源矩阵的leading dimension
    const int*    indices,        // 要保留的列的索引数组 (在GPU上)
    const int     rows,           // 要复制的行数
    const int     num_new_cols)   // 新的列数 (要保留的列的数量)
{
    // 计算当前线程要处理的目标矩阵中的元素 (row, dest_col)
    int dest_col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // 边界检查，确保不越界
    if (row < rows && dest_col < num_new_cols) {
        // 从索引数组中查出需要从源矩阵的哪一列读取数据
        int src_col = indices[dest_col];
        
        // 计算源和目标元素的线性地址
        size_t src_idx = (size_t)row + (size_t)src_col * ld_src;
        size_t dest_idx = (size_t)row + (size_t)dest_col * ld_dest;
        
        // 执行复制
        dest[dest_idx] = src[src_idx];
    }
}

// 根据行索引选择行，列保持不变
__global__ void select_rows_kernel(
    float*       dest,        // 目标矩阵指针 (紧凑)
    const int     ld_dest,     // 目标矩阵的 ld
    const float* src,         // 源矩阵指针
    const int     ld_src,      // 源矩阵的 ld
    const int*    indices,     // 要保留的行索引数组 (在GPU上)
    const int     rows_keep,   // 保留的行数
    const int     cols)        // 列数
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows_keep && col < cols) {
        int src_row = indices[row];
        size_t src_idx  = (size_t)col * ld_src  + src_row;
        size_t dest_idx = (size_t)col * ld_dest + row;
        dest[dest_idx] = src[src_idx];
    }
}

template <typename T>
void recursive_compute_w(cublasHandle_t handle,
                         size_t m,
                         size_t total_cols,
                         size_t block_size,
                         T* d_W, size_t ldw,
                         const T* d_Y, size_t ldy,
                         T* d_work_C)
{
    const T one = 1.0;
    const T zero = 0.0;
    const T minus_one = -1.0;

    for (size_t i = block_size; i < total_cols; i += block_size) {
        
        size_t left_width = i;
        size_t right_width = std::min(block_size, total_cols - i);
        if (right_width == 0) continue;

        const T* Y_left = d_Y;
        const T* W_left = d_W;
        T*       W_right = d_W + left_width * ldw;
        
        // C = Y_left^T * W_right
        if constexpr (std::is_same_v<T, float>) {
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        left_width, right_width, m,
                        &one, Y_left, ldy, W_right, ldw,
                        &zero, d_work_C, left_width);

            // W_right = W_right - W_left * C
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m, right_width, left_width,
                        &minus_one, W_left, ldw, d_work_C, left_width,
                        &one, W_right, ldw);
        } else {
            cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        left_width, right_width, m,
                        &one, Y_left, ldy, W_right, ldw,
                        &zero, d_work_C, left_width);

            // W_right = W_right - W_left * C
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        m, right_width, left_width,
                        &minus_one, W_left, ldw, d_work_C, left_width,
                        &one, W_right, ldw);
        }
    }
}

// 放在 print_device_matrix 函数之后
float verify_orthogonality(cublasHandle_t handle, int rows, int cols, float* d_X_ptr, int ldx) {
    thrust::device_vector<float> d_XTX(cols * cols);
    float* d_XTX_ptr = thrust::raw_pointer_cast(d_XTX.data());
    
    float one = 1.0f, zero = 0.0f;
    
    // 1. 计算 XTX = X^T * X
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                cols, cols, rows,
                &one, d_X_ptr, ldx,
                d_X_ptr, ldx,
                &zero, d_XTX_ptr, cols);
                
    // 2. 计算 XTX - I
    // 更简单的办法: 直接在CPU上做
    std::vector<float> h_XTX(cols * cols);
    cudaMemcpy(h_XTX.data(), d_XTX_ptr, cols * cols * sizeof(float), cudaMemcpyDeviceToHost);
    
    float frobenius_norm_sq = 0.0f;
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < cols; ++j) {
            float val = h_XTX[j * cols + i];
            if (i == j) {
                val -= 1.0f;
            }
            frobenius_norm_sq += val * val;
        }
    }
    return sqrt(frobenius_norm_sq);
}

// 提取上三角矩阵R
__global__ void extract_upper_triangle_kernel(
    float*       dest,       // 目标矩阵 (紧凑)
    const int     ld_dest,    // 目标矩阵的 ld
    const float* src,        // 源矩阵 (面板)
    const int     ld_src,     // 源矩阵的 ld
    const int     rows,       // 要提取的区域的行数
    const int     cols)       // 要提取的区域的列数
{
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 列索引
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 行索引

    if (i < rows && j < cols) {
        // 计算线性地址
        size_t src_idx = (size_t)i + (size_t)j * ld_src;
        size_t dest_idx = (size_t)i + (size_t)j * ld_dest;

        if (i <= j) {
            // 如果在对角线或上三角，从源拷贝
            dest[dest_idx] = src[src_idx];
        } else {
            // 如果在下三角，强制置为零
            dest[dest_idx] = 0.0f;
        }
    }
}

#endif
