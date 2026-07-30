#ifndef HQB_SVD_LOGIC_H
#define HQB_SVD_LOGIC_H
#include <thrust/device_vector.h>
#include <vector>
#include <string>
#include "my_cuda_kernels.h" 
// 为内联标定工具补充所需 CUDA/CUBLAS 头
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
// 自适应低精选择器（仅在 计算=float 且 存储=half/bfloat16 的调用点使用）
namespace hqbtune {
struct RtTune {
    bool   calibrated = false;   // 是否已完成一次性校准
    double gflops32   = 1e3;     // FP32 有效吞吐（GFLOPS）
    double gflops16   = 2e3;     // FP16/BF16 有效吞吐（GFLOPS）
    double bw_conv    = 600.0;   // 类型转换有效带宽（GB/s）
    int    force      = 0;       // 强制策略：-1 强制 FP32，+1 强制低精，0 自动
};
// 单例获取器，避免多个编译单元重复定义
inline RtTune& tuner() { static RtTune t; return t; }
// 提前初始化 CUDA/cuBLAS，缩短后续第一次调用的冷启动时间
// 建议在程序启动早期调用一次：hqbtune::early_cuda_cublas_init();
inline void early_cuda_cublas_init() {
    static bool done = false; if (done) return; done = true;
    // 触发 CUDA 上下文创建
    (void)cudaFree(0);
    // 创建临时 cuBLAS 句柄并启用 TF32（对后续 FP32 GEMM 更公平）
    cublasHandle_t h = nullptr; cublasCreate(&h);
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    // 做一次极小 GEMM 触发内核选择/JIT
    float *A=nullptr, *B=nullptr, *C=nullptr; const int m=16,n=16,k=16; 
    cudaMalloc(&A, m*k*sizeof(float)); cudaMalloc(&B, k*n*sizeof(float)); cudaMalloc(&C, m*n*sizeof(float));
    const float alpha = 1.0f, beta = 0.0f;
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                 m, n, k,
                 &alpha,
                 A, CUDA_R_32F, m,
                 B, CUDA_R_32F, k,
                 &beta,
                 C, CUDA_R_32F, m,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaDeviceSynchronize();
    cudaFree(A); cudaFree(B); cudaFree(C);
    cublasDestroy(h);
}
// 一次性实测校准：测转换带宽与 GEMM 吞吐
inline void calibrate_once(cublasHandle_t h, cudaStream_t s,
                           int m_ref, int r_ref, int nt_ref) {
    RtTune& rt = tuner();
    if (rt.calibrated) return;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    // 启用 TF32 Tensor Core，以提升 FP32 路径吞吐
    cublasSetMathMode(h, CUBLAS_TF32_TENSOR_OP_MATH);
    // --- 转换带宽标定（按参考尺寸拆 主体/尾部） ---
    int rows_conv = std::max(256, std::min(m_ref, 8192));
    // 确保行步长对齐到 4 的倍数，避免 vec4 读写未对齐
    rows_conv = ((rows_conv + 3) / 4) * 4;
    int cols_conv = std::max(256, std::min(nt_ref, 8192));
    size_t rows_main = (rows_conv / 4) * 4;   // 4 的倍数给 vec4 内核
    size_t rows_rem  = rows_conv - rows_main; // 余下给标量内核
    float*  d_f = nullptr; __half* d_h = nullptr;
    cudaMalloc(&d_f, (size_t)rows_conv * cols_conv * sizeof(float));
    cudaMalloc(&d_h, (size_t)rows_conv * cols_conv * sizeof(__half));
    dim3 threads_vec(32, 8);
    dim3 blocks_vec((cols_conv + threads_vec.x - 1) / threads_vec.x,
                    ((int)rows_main / 4 + threads_vec.y - 1) / threads_vec.y);
    // 轻量预热 1 次
    if (rows_main > 0) {
        convert_kernel_strided_vec4<float, __half><<<blocks_vec, threads_vec, 0, s>>>(
            d_f, rows_conv, d_h, rows_conv, rows_main, cols_conv);
        convert_kernel_strided_vec4<__half, float><<<blocks_vec, threads_vec, 0, s>>>(
            d_h, rows_conv, d_f, rows_conv, rows_main, cols_conv);
    }
    if (rows_rem > 0) {
        float*  f_rem = d_f + rows_main;
        __half* h_rem = d_h + rows_main;
        dim3 threads_scalar(16,16);
        dim3 blocks_scalar((cols_conv + threads_scalar.x - 1) / threads_scalar.x,
                          ((int)rows_rem + threads_scalar.y - 1) / threads_scalar.y);
        convert_kernel_strided_f2h<<<blocks_scalar, threads_scalar, 0, s>>>(
            f_rem, (int)rows_conv, h_rem, (int)rows_conv, (int)rows_rem, (int)cols_conv);
        convert_kernel_strided_h2f<<<blocks_scalar, threads_scalar, 0, s>>>(
            h_rem, (int)rows_conv, f_rem, (int)rows_conv, (int)rows_rem, (int)cols_conv);
    }
    cudaStreamSynchronize(s);
    // 多次测量取中位数（samples=3）
    auto median3 = [](float a, float b, float c){
        if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
        if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
        return c;
    };
    float t_conv_ms[3];
    for (int si = 0; si < 3; ++si) {
        cudaEventRecord(e0, s);
        if (rows_main > 0) {
            convert_kernel_strided_vec4<float, __half><<<blocks_vec, threads_vec, 0, s>>>(
                d_f, rows_conv, d_h, rows_conv, rows_main, cols_conv);
        }
        if (rows_rem > 0) {
            float*  f_rem = d_f + rows_main;
            __half* h_rem = d_h + rows_main;
            dim3 threads_scalar(16,16);
            dim3 blocks_scalar((cols_conv + threads_scalar.x - 1) / threads_scalar.x,
                              ((int)rows_rem + threads_scalar.y - 1) / threads_scalar.y);
            convert_kernel_strided_f2h<<<blocks_scalar, threads_scalar, 0, s>>>(
                f_rem, (int)rows_conv, h_rem, (int)rows_conv, (int)rows_rem, (int)cols_conv);
        }
        if (rows_main > 0) {
            convert_kernel_strided_vec4<__half, float><<<blocks_vec, threads_vec, 0, s>>>(
                d_h, rows_conv, d_f, rows_conv, rows_main, cols_conv);
        }
        if (rows_rem > 0) {
            float*  f_rem = d_f + rows_main;
            __half* h_rem = d_h + rows_main;
            dim3 threads_scalar(16,16);
            dim3 blocks_scalar((cols_conv + threads_scalar.x - 1) / threads_scalar.x,
                              ((int)rows_rem + threads_scalar.y - 1) / threads_scalar.y);
            convert_kernel_strided_h2f<<<blocks_scalar, threads_scalar, 0, s>>>(
                h_rem, (int)rows_conv, f_rem, (int)rows_conv, (int)rows_rem, (int)cols_conv);
        }
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t_conv_ms[si], e0, e1);
    }
    float ms_conv = median3(t_conv_ms[0], t_conv_ms[1], t_conv_ms[2]);
    if (ms_conv <= 0) ms_conv = 1e-3f;
    // 每次往返：float->half + half->float，共 ~12B/元素（一次样本）
    double bytes_total = 12.0 * (double)rows_conv * cols_conv;
    rt.bw_conv = (bytes_total / (ms_conv / 1e3)) / 1e9;
    // GEMM 吞吐标定
    int M = std::max(256, std::min(m_ref, 8192));
    int N = std::max(256, std::min(nt_ref, 8192));
    // 取消对 K 的上限，按传入 r_ref 标定；保留下限 64 以稳定计时
    int K = std::max(64, r_ref);
    // FP32/TF32 路径
    float *d_A32=nullptr, *d_B32=nullptr, *d_C32=nullptr;
    cudaMalloc(&d_A32, (size_t)M * K * sizeof(float));
    cudaMalloc(&d_B32, (size_t)K * N * sizeof(float));
    cudaMalloc(&d_C32, (size_t)M * N * sizeof(float));
    float alpha32 = 1.0f, beta32 = 0.0f;
    // 轻量预热 1 次
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                 M, N, K,
                 &alpha32,
                 d_A32, CUDA_R_32F, M,
                 d_B32, CUDA_R_32F, K,
                 &beta32,
                 d_C32, CUDA_R_32F, M,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaStreamSynchronize(s);
    // 多次测量取中位数（samples=3，单次 GEMM）
    float t32_ms[3];
    for (int si = 0; si < 3; ++si) {
        cudaEventRecord(e0, s);
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, K,
                     &alpha32,
                     d_A32, CUDA_R_32F, M,
                     d_B32, CUDA_R_32F, K,
                     &beta32,
                     d_C32, CUDA_R_32F, M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t32_ms[si], e0, e1);
    }
    float ms_gemm32 = median3(t32_ms[0], t32_ms[1], t32_ms[2]);
    if (ms_gemm32 <= 0) ms_gemm32 = 1e-3f;
    {
        double flops_one = 2.0 * (double)M * N * K;
        rt.gflops32 = flops_one / (ms_gemm32 / 1e3) / 1e9;
    }
    // FP16 路径（compute=FP32）
    __half *d_A16=nullptr, *d_B16=nullptr; float *d_C16=nullptr;
    cudaMalloc(&d_A16, (size_t)M * K * sizeof(__half));
    cudaMalloc(&d_B16, (size_t)K * N * sizeof(__half));
    cudaMalloc(&d_C16, (size_t)M * N * sizeof(float));
    // 轻量预热 1 次
    cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                 M, N, K,
                 &alpha32,
                 d_A16, CUDA_R_16F, M,
                 d_B16, CUDA_R_16F, K,
                 &beta32,
                 d_C16, CUDA_R_32F, M,
                 CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    cudaStreamSynchronize(s);
    // 多次测量取中位数（samples=3，单次 GEMM）
    float t16_ms[3];
    for (int si = 0; si < 3; ++si) {
        cudaEventRecord(e0, s);
        cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, K,
                     &alpha32,
                     d_A16, CUDA_R_16F, M,
                     d_B16, CUDA_R_16F, K,
                     &beta32,
                     d_C16, CUDA_R_32F, M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        cudaEventElapsedTime(&t16_ms[si], e0, e1);
    }
    float ms_gemm16 = median3(t16_ms[0], t16_ms[1], t16_ms[2]);
    if (ms_gemm16 <= 0) ms_gemm16 = 1e-3f;
    {
        double flops_one = 2.0 * (double)M * N * K;
        rt.gflops16 = flops_one / (ms_gemm16 / 1e3) / 1e9;
    }
    // 清理资源
    cudaFree(d_f); cudaFree(d_h);
    cudaFree(d_A32); cudaFree(d_B32); cudaFree(d_C32);
    cudaFree(d_A16); cudaFree(d_B16); cudaFree(d_C16);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    rt.calibrated = true;
    // 可选一次性输出标定结果（设置环境变量 HQB_DEBUG=1 生效）
    {
        const char* dbg = std::getenv("HQB_DEBUG");
        if (dbg && dbg[0] != '0') {
            printf("[hqbtune] calib: gflops32=%.1f, gflops16=%.1f, bw_conv=%.1f GB/s\n",
                   rt.gflops32, rt.gflops16, rt.bw_conv);
        }
    }
}
// 是否选择低精（half/bfloat16 输入，FP32 累加）
inline bool decide_lowp(long m_h, long m_r, long n_t, double margin = 0.15) {
    RtTune& rt = tuner();
    if (rt.force != 0) return rt.force > 0;  // 外部强制优先
    if (!rt.calibrated) return false;        // 未校准则保守走 FP32
    // 窄维度保护：默认 m_r,n_t 小于 64 时不走低精；可通过环境变量覆盖
    long min_r = 64, min_nt = 64;
    if (const char* s = std::getenv("HQB_MIN_R")) {
        long v = std::atol(s); if (v > 0) min_r = v;
    }
    if (const char* s = std::getenv("HQB_MIN_NT")) {
        long v = std::atol(s); if (v > 0) min_nt = v;
    }
    if (m_r < min_r || n_t < min_nt) return false; // 窄维度：转换不划算
    // 允许通过环境变量 HQB_MARGIN 覆盖判决裕量（例如 0 或 0.05）
    const char* m_env = std::getenv("HQB_MARGIN");
    if (m_env && m_env[0] != '\0') {
        double v = atof(m_env);
        if (v >= 0.0 && v < 1.0) margin = v;
    }
    const double flops = 4.0 * (double)m_h * (double)m_r * (double)n_t; // 两次 GEMM
    const double t32 = flops / (rt.gflops32 * 1e9);
    const double t16 = flops / (rt.gflops16 * 1e9);
    double a_factor = 1.0;
    if (const char* lo = std::getenv("HQB_LOWPOUT"); lo && (lo[0]=='c' || lo[0]=='C')) a_factor = 2.0;
    const double bytes_conv = 6.0 * ( 2.0 * (double)m_h * (double)m_r
                                    + a_factor * (double)m_h * (double)n_t );
    const double tconv = bytes_conv / (rt.bw_conv * 1e9);
    return (t32 - t16) > tconv * (1.0 + margin);
}
} // namespace hqbtune
// 打印设备端矩阵左上角子块（列主序）
template<typename T>
inline void print_device_matrix_topleft(const T* d_ptr, int ld, int rows, int cols,
                                        const char* name, int max_r, int max_c,
                                        cudaStream_t s) {
    int pr = std::min(rows, max_r);
    int pc = std::min(cols, max_c);
    if (pr <= 0 || pc <= 0) {
        printf("[%s] empty matrix (%d x %d)\n", name, rows, cols);
        return;
    }
    std::vector<T> h((size_t)pr * pc);
    // 将设备端左上 pr x pc 子块复制为紧致列主序到主机
    cudaMemcpy2DAsync(h.data(), (size_t)pr * sizeof(T),
                                 d_ptr, (size_t)ld * sizeof(T),
                                 (size_t)pr * sizeof(T), pc,
                                 cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
    printf("%s (showing %d x %d of %d x %d):\n", name, pr, pc, rows, cols);
    for (int i = 0; i < pr; ++i) {
        for (int j = 0; j < pc; ++j) {
            double v = (double)h[(size_t)j * pr + i];
            printf(" % .6e", v);
        }
        printf("\n");
    }
}
// 通用类型转换核函数
template<typename Tin, typename Tout>
__global__ void convert_kernel(const Tin* src, Tout* dst, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = static_cast<Tout>(src[idx]);
    }
}
// 创建低秩矩阵 A = U * S * V^T 
template<typename T_storage, typename T_compute>
void create_low_rank_matrix(
    thrust::device_vector<T_storage>& d_A_storage,      
    std::vector<T_compute>& h_S_expected,               
    long int m, long int n, long int real_rank,
    const std::string& spectrum_type, T_compute s_max, T_compute s_min,
    cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle, cudaStream_t stream,
    long int& ld_storage_out)
{
    const T_compute one = T_compute(1.0);
    const T_compute zero = T_compute(0.0);
    
    // 临时向量和矩阵都使用 T_compute 精度
    thrust::device_vector<T_compute> factor1(m * real_rank);
    thrust::device_vector<T_compute> factor2(n * real_rank);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    generateNormalMatrix<T_compute>(gen, factor1, m, real_rank); 
    curandSetPseudoRandomGeneratorSeed(gen, 9999ULL);
    generateNormalMatrix<T_compute>(gen, factor2, n, real_rank);
    curandDestroyGenerator(gen);
    
    // --- 对 factor1 (U) 进行正交化 ---
    {
        int lwork_geqrf, lwork_orgqr, lwork;
        SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, m, real_rank, thrust::raw_pointer_cast(factor1.data()), m, &lwork_geqrf);
        SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, m, real_rank, real_rank, thrust::raw_pointer_cast(factor1.data()), m, nullptr, &lwork_orgqr);
        lwork = std::max(lwork_geqrf, lwork_orgqr);
        
        thrust::device_vector<T_compute> work(lwork);
        thrust::device_vector<T_compute> tau(real_rank);
        thrust::device_vector<int> devInfo(1);
        SolverCaller<T_compute>::geqrf(cusolverHandle, m, real_rank, thrust::raw_pointer_cast(factor1.data()), m,
                                       thrust::raw_pointer_cast(tau.data()), thrust::raw_pointer_cast(work.data()), lwork, thrust::raw_pointer_cast(devInfo.data()));
        SolverCaller<T_compute>::orgqr(cusolverHandle, m, real_rank, real_rank, thrust::raw_pointer_cast(factor1.data()), m,
                                       thrust::raw_pointer_cast(tau.data()), thrust::raw_pointer_cast(work.data()), lwork, thrust::raw_pointer_cast(devInfo.data()));
    }
    // --- 对 factor2 (V) 进行正交化 ---
    {
        int lwork_geqrf, lwork_orgqr, lwork;
        SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, n, real_rank, thrust::raw_pointer_cast(factor2.data()), n, &lwork_geqrf);
        SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, n, real_rank, real_rank, thrust::raw_pointer_cast(factor2.data()), n, nullptr, &lwork_orgqr);
        lwork = std::max(lwork_geqrf, lwork_orgqr);
        
        thrust::device_vector<T_compute> work(lwork);
        thrust::device_vector<T_compute> tau(real_rank);
        thrust::device_vector<int> devInfo(1);
        SolverCaller<T_compute>::geqrf(cusolverHandle, n, real_rank, thrust::raw_pointer_cast(factor2.data()), n,
                                       thrust::raw_pointer_cast(tau.data()), thrust::raw_pointer_cast(work.data()), lwork, thrust::raw_pointer_cast(devInfo.data()));
        SolverCaller<T_compute>::orgqr(cusolverHandle, n, real_rank, real_rank, thrust::raw_pointer_cast(factor2.data()), n,
                                       thrust::raw_pointer_cast(tau.data()), thrust::raw_pointer_cast(work.data()), lwork, thrust::raw_pointer_cast(devInfo.data()));
    }
    // --- 生成奇异值 ---
    generate_singular_values<T_compute>(spectrum_type, s_max, s_min, h_S_expected, real_rank);
    thrust::device_vector<T_compute> d_S = h_S_expected;
    // if (real_rank <= 64) {
    //     printf("Constructed singular values:\n");
    //     for (size_t i = 0; i < real_rank; ++i) {
    //         printf("  S_expected[%2zu] = %e\n", i, (double)h_S_expected[i]);
    //     }
    // }
    // --- 组装矩阵 A = U * diag(S) * V^T ---
    // 1. 计算 Temp = U * diag(S)
    thrust::device_vector<T_compute> d_Temp(m * real_rank);
    dim3 threads_scale(16, 16);
    dim3 blocks_scale((real_rank + 15) / 16, (m + 15) / 16);
    if constexpr (std::is_same_v<T_compute, float>) {
        scale_columns_kernel_f<<<blocks_scale, threads_scale, 0, stream>>>(
            thrust::raw_pointer_cast(d_Temp.data()),
            thrust::raw_pointer_cast(factor1.data()),
            thrust::raw_pointer_cast(d_S.data()),
            (int)m, (int)real_rank, (int)m, (int)m);
    } else if constexpr (std::is_same_v<T_compute, double>) {
        scale_columns_kernel_d<<<blocks_scale, threads_scale, 0, stream>>>(
            reinterpret_cast<double*>(thrust::raw_pointer_cast(d_Temp.data())),
            reinterpret_cast<const double*>(thrust::raw_pointer_cast(factor1.data())),
            reinterpret_cast<const double*>(thrust::raw_pointer_cast(d_S.data())),
            (int)m, (int)real_rank, (int)m, (int)m);
    } else {
        static_assert(std::is_same_v<T_compute, float> || std::is_same_v<T_compute, double>,
                      "scale_columns only supports float/double in this TU");
    }
    // 2. 计算 A_compute = Temp * V^T
    thrust::device_vector<T_compute> d_A_compute(m * n);
    BlasCaller<T_compute>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                m, n, real_rank, &one,
                                thrust::raw_pointer_cast(d_Temp.data()), m,
                                thrust::raw_pointer_cast(factor2.data()), n,
                                &zero, thrust::raw_pointer_cast(d_A_compute.data()), m);
    
    // --- 将计算结果转换为最终的存储精度 T_storage（可选对齐填充） ---
    int align = 0;
    if (const char* al = std::getenv("HQB_ALIGN_M")) {
        align = atoi(al);
        if (align != 16 && align != 32) align = 0;
    }
    long int ld_storage = (align > 0) ? ((m + align - 1) / align) * align : m;
    ld_storage_out = ld_storage;
    d_A_storage.resize((size_t)ld_storage * n);
    // 先置零填充行
    cudaMemsetAsync(thrust::raw_pointer_cast(d_A_storage.data()), 0, (size_t)ld_storage * n * sizeof(T_storage), stream);
    // 按列步长从紧致布局复制到填充布局
    dim3 threads_scalar(16,16);
    dim3 blocks_scalar((n + threads_scalar.x - 1) / threads_scalar.x, (m + threads_scalar.y - 1) / threads_scalar.y);
    if constexpr (std::is_same_v<T_compute, float> && std::is_same_v<T_storage, __half>) {
        convert_kernel_strided_f2h<<<blocks_scalar, threads_scalar, 0, stream>>>(
            reinterpret_cast<const float*>(thrust::raw_pointer_cast(d_A_compute.data())), (int)m,
            reinterpret_cast<__half*>(thrust::raw_pointer_cast(d_A_storage.data())), (int)ld_storage,
            (int)m, (int)n);
    } else if constexpr (std::is_same_v<T_compute, float> && std::is_same_v<T_storage, __nv_bfloat16>) {
        convert_kernel_strided_f2bf16<<<blocks_scalar, threads_scalar, 0, stream>>>(
            reinterpret_cast<const float*>(thrust::raw_pointer_cast(d_A_compute.data())), (int)m,
            reinterpret_cast<__nv_bfloat16*>(thrust::raw_pointer_cast(d_A_storage.data())), (int)ld_storage,
            (int)m, (int)n);
    } else if constexpr (std::is_same_v<T_compute, float> && std::is_same_v<T_storage, float>) {
        cudaMemcpy2DAsync(
            thrust::raw_pointer_cast(d_A_storage.data()), (size_t)ld_storage * sizeof(float),
            thrust::raw_pointer_cast(d_A_compute.data()), (size_t)m * sizeof(float),
            (size_t)m * sizeof(float), (size_t)n,
            cudaMemcpyDeviceToDevice, stream);
    } else if constexpr (std::is_same_v<T_compute, double> && std::is_same_v<T_storage, double>) {
        cudaMemcpy2DAsync(
            thrust::raw_pointer_cast(d_A_storage.data()), (size_t)ld_storage * sizeof(double),
            thrust::raw_pointer_cast(d_A_compute.data()), (size_t)m * sizeof(double),
            (size_t)m * sizeof(double), (size_t)n,
            cudaMemcpyDeviceToDevice, stream);
    } else {
        // 其他组合当前未使用；如出现则报错提醒
        printf("[Error] Unsupported T_compute->T_storage conversion in create_low_rank_matrix.\n");
    }
    
}
// 向后兼容的重载：不关心存储对齐时的行步长，返回紧致布局（ld=m）
template<typename T_storage, typename T_compute>
inline void create_low_rank_matrix(
    thrust::device_vector<T_storage>& d_A_storage,      
    std::vector<T_compute>& h_S_expected,               
    long int m, long int n, long int real_rank,
    const std::string& spectrum_type, T_compute s_max, T_compute s_min,
    cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle, cudaStream_t stream)
{
    long int dummy_ld = m;
    create_low_rank_matrix<T_storage, T_compute>(d_A_storage, h_S_expected, m, n, real_rank, spectrum_type,
                                                 s_max, s_min, cublasHandle, cusolverHandle, stream, dummy_ld);
}
template<typename T_compute>
void verify_qb_decomposition(
    cublasHandle_t handle,
    long int m, long int n, long int final_rank,
    const thrust::device_vector<T_compute>& d_A_backup,
    const T_compute* d_Q_final_ptr,
    const T_compute* d_R_final_ptr,
    long int ld_R,
    cudaStream_t stream)
{
    printf("=== Step 3: Verifying QB Decomposition (A ≈ Q*R) ===\n");
    if (final_rank == 0) {
        printf("Final rank is 0, skipping verification.\n");
        return;
    }
    const T_compute one = T_compute(1.0);
    const T_compute zero = T_compute(0.0);
    const T_compute minus_one = T_compute(-1.0);
    // 验证阶段固定 cuBLAS 指针/数学模式，避免 alpha/beta 与 TF32 干扰
    cublasPointerMode_t old_pm; cublasGetPointerMode(handle, &old_pm);
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    cublasMath_t old_mm; cublasGetMathMode(handle, &old_mm);
    cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    cudaStream_t s = nullptr; cublasGetStream(handle, &s);
    // --- 1. 验证 Q 的正交性: ||Q^T * Q - I||_F ---
    if constexpr (std::is_same<T_compute, float>::value) {
        // 用 double 聚合评估 Q 的正交性（更稳健，避免大 m 下的 FP32 累积误差）
        thrust::device_vector<double> d_Qd((size_t)m * (size_t)final_rank);
        double* d_Qd_ptr = thrust::raw_pointer_cast(d_Qd.data());
        dim3 tqd(16,16);
        dim3 bqd((int)((final_rank + tqd.x - 1)/tqd.x), (int)((m + tqd.y - 1)/tqd.y));
        convert_kernel_strided_f2d<<<bqd, tqd, 0, stream>>>(
            d_Q_final_ptr, (size_t)m,
            d_Qd_ptr,      (size_t)m,
            (size_t)m, (size_t)final_rank);
        double q_ortho_error_d = verify_orthogonality<double>(handle, (int)m, (int)final_rank, d_Qd_ptr, (int)m);
        printf("Orthogonality of Q ||Q^T*Q - I||_F = %.12e\n", q_ortho_error_d);
    } else {
        T_compute q_ortho_error = verify_orthogonality<T_compute>(handle, m, final_rank, d_Q_final_ptr, m);
        printf("Orthogonality of Q ||Q^T*Q - I||_F = %.12e\n", (double)q_ortho_error);
    }
    // --- 2. 直接重构误差（可选，尺寸过大时跳过）：A_hat = Q * R_top ---
    // 避免在巨大矩阵上分配 m*n 的临时缓冲区，默认阈值 1 GiB（可用 HQB_MAX_VERIFY_BYTES 覆盖）。
    size_t elems = (size_t)m * (size_t)n;
    size_t bytes_total = elems * sizeof(T_compute);
    size_t max_bytes = 1ull << 30; // 1 GiB 默认
    if (const char* env = std::getenv("HQB_MAX_VERIFY_BYTES")) {
        long long v = atoll(env); if (v > 0) max_bytes = (size_t)v;
    }
    const T_compute* d_R_top = d_R_final_ptr; // 列主序，逻辑尺寸为 final_rank x n，步长由 ld_R 指定
    if (bytes_total <= max_bytes) {
        printf("Verifying Reconstruction Error ||A - Q*R_top||_F...\n");
        thrust::device_vector<T_compute> d_A_reconstructed(m * n);
        T_compute* d_A_reconstructed_ptr = thrust::raw_pointer_cast(d_A_reconstructed.data());
        BlasCaller<T_compute>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, n, final_rank,
                                    &one,
                                    d_Q_final_ptr, m,
                                    d_R_top, ld_R,
                                    &zero,
                                    d_A_reconstructed_ptr, m);
        // 误差矩阵: A_backup - A_reconstructed
        BlasCaller<T_compute>::geam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    m, n,
                                    &one, thrust::raw_pointer_cast(d_A_backup.data()), m,
                                    &minus_one, d_A_reconstructed_ptr, m,
                                    d_A_reconstructed_ptr, m);
        cudaStreamSynchronize(s);
        // Frobenius 范数
        T_compute reconstruction_error = 0, norm_A_original = 0;
        if constexpr (std::is_same<T_compute, float>::value) {
            cublasSnrm2(handle, (int)((long long)m * n), (const float*)d_A_reconstructed_ptr, 1, (float*)&reconstruction_error);
            cublasSnrm2(handle, (int)((long long)m * n), (const float*)thrust::raw_pointer_cast(d_A_backup.data()), 1, (float*)&norm_A_original);
        } else {
            cublasDnrm2(handle, (int)((long long)m * n), (const double*)d_A_reconstructed_ptr, 1, (double*)&reconstruction_error);
            cublasDnrm2(handle, (int)((long long)m * n), (const double*)thrust::raw_pointer_cast(d_A_backup.data()), 1, (double*)&norm_A_original);
        }
        cudaStreamSynchronize(s);
        printf("Reconstruction error ||A - Q*R||_F = %.12e\n", (double)reconstruction_error);
        T_compute relative_error = (norm_A_original > 1e-12) ? (reconstruction_error / norm_A_original) : reconstruction_error;
        printf("Relative reconstruction error ||A - Q*R||_F / ||A||_F = %.12e\n", (double)relative_error);
    } else {
        printf("[Info] Skipping full reconstruction in verification due to size (%.2f GiB > %.2f GiB). Using cross residual only.\n",
               bytes_total / (1024.0*1024.0*1024.0), max_bytes / (1024.0*1024.0*1024.0));
    }
    // 交叉校验：||Q^T A - R_top||_F（能直接验证 R 的一致性）
    thrust::device_vector<T_compute> d_Bchk(final_rank * n);
    T_compute* d_Bchk_ptr = thrust::raw_pointer_cast(d_Bchk.data());
    BlasCaller<T_compute>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                final_rank, n, m,
                                &one,
                                d_Q_final_ptr, m,
                                thrust::raw_pointer_cast(d_A_backup.data()), m,
                                &zero,
                                d_Bchk_ptr, final_rank);
    BlasCaller<T_compute>::geam(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                final_rank, n,
                                &one, d_Bchk_ptr, final_rank,
                                &minus_one, d_R_top, ld_R,
                                d_Bchk_ptr, final_rank);
    cudaStreamSynchronize(s);
    T_compute cross_err = 0;
    if constexpr (std::is_same<T_compute, float>::value) {
        cublasSnrm2(handle, (int)((long long)final_rank * n), (const float*)d_Bchk_ptr, 1, (float*)&cross_err);
    } else {
        cublasDnrm2(handle, (int)((long long)final_rank * n), (const double*)d_Bchk_ptr, 1, (double*)&cross_err);
    }
    cudaStreamSynchronize(s);
    printf("Cross residual ||Q^T*A - R_top||_F = %.12e\n", (double)cross_err);
    // (preview prints of Q and R_top removed)
    // 恢复句柄原有模式
    cublasSetPointerMode(handle, old_pm);
    cublasSetMathMode(handle, old_mm);
}

template<typename T_compute>
void verify_qb_decomposition(
    cublasHandle_t handle,
    long int m, long int n, long int final_rank,
    const thrust::device_vector<T_compute>& d_A_backup,
    const T_compute* d_Q_final_ptr,
    const T_compute* d_R_final_ptr,
    cudaStream_t stream)
{
    verify_qb_decomposition(handle, m, n, final_rank,
                            d_A_backup, d_Q_final_ptr, d_R_final_ptr,
                            n, stream);
}
#endif
