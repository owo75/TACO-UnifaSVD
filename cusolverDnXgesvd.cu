#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <cusolverDn.h>
#include <type_traits>
#include <cmath>
#include "src/verify_results.h"
#include <vector>
#include <string>
#include <cusolverDn.h>
#include "src/my_cuda_kernels_float.h"
#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "src/cli_utils.h"

long int m, n, rank;
enum class PrecisionMode { Float32, Half16, BFloat16 };
PrecisionMode g_precision = PrecisionMode::Float32;

__global__ void convert_float_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void convert_float_to_bfloat16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

__global__ void convert_half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

__global__ void convert_bfloat16_to_float_kernel(const __nv_bfloat16* __restrict__ src, float* __restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}
std::string spectrum_type; // "geometric", "uniform", "cluster0", "cluster1"，"arithmetic", "normal"
float s_max;
float s_min;
int parseArguments(int argc,char *argv[])
{
    const bool show_help = cli::help_requested(argc, argv);
    if(argc < 7 || show_help)
    {
        printf("Usage: %s <m> <n> <rank> <spectrum_type> <s_max> <s_min> [--precision=<float|half|bfloat16>]\n", argv[0]);
        printf("  <spectrum_type>: 'geometric', 'uniform', 'cluster0', 'cluster1', 'arithmetic', 'normal'\n");
        printf("  <s_max>: Maximum singular value (e.g., 1.0)\n");
        printf("  <s_min>: Minimum singular value (e.g., 1e-10)\n");
        return show_help ? 1 : -1;
    }
    if (!cli::positive_integer(argv[1], "m", m) ||
        !cli::positive_integer(argv[2], "n", n) ||
        !cli::positive_integer(argv[3], "rank", rank) ||
        !cli::positive_real(argv[5], "s_max", s_max) ||
        !cli::positive_real(argv[6], "s_min", s_min)) return -1;
    spectrum_type = argv[4];
    if (m < n || rank > std::min(m, n) || s_max < s_min) {
        std::fprintf(stderr, "Error: require m >= n, rank <= min(m,n), and s_max >= s_min.\n");
        return -1;
    }

    for (int i = 7; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--precision=", 0) == 0) {
            std::string val = arg.substr(strlen("--precision="));
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "float" || val == "fp32") {
                g_precision = PrecisionMode::Float32;
            } else if (val == "half" || val == "fp16" || val == "16f") {
                g_precision = PrecisionMode::Half16;
            } else if (val == "bfloat16" || val == "bf16") {
                g_precision = PrecisionMode::BFloat16;
            } else {
                printf("Error: Unknown precision '%s'. Use float, half, or bfloat16.\n", val.c_str());
                return -1;
            }
        } else {
            std::fprintf(stderr, "Error: Unrecognized argument '%s'. Use --help for usage.\n", arg.c_str());
            return -1;
        }
    }
    
    printf("Parsed arguments:\n");
    printf("  m = %ld, n = %ld, rank = %ld\n", m, n, rank);
    const char* precision_str = (g_precision == PrecisionMode::Float32) ? "float32" : (g_precision == PrecisionMode::Half16 ? "half" : "bfloat16");
    printf("  Precision: %s\n", precision_str);
    printf("  Spectrum Type: %s\n", spectrum_type.c_str());
    printf("  Singular Value Range: [%e, %e]\n\n", s_max, s_min);

    if (spectrum_type != "geometric" && spectrum_type != "uniform" &&
        spectrum_type != "cluster0" && spectrum_type != "cluster1" &&
        spectrum_type != "arithmetic" && spectrum_type != "normal") {
        printf("Error: Invalid spectrum_type '%s'.\n", spectrum_type.c_str());
        return -1;
    }

    return 0;
}

void generate_singular_values(
    const std::string& type,
    float max_val,
    float min_val,
    std::vector<float>& s_values,
    size_t rank)
{
    s_values.resize(rank);
    if (rank == 0) return;

    if (type == "geometric") {
        printf("Generating GEOMETRICALLY distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            float log_max = logf(max_val);
            float log_min = logf(min_val);
            float step = (log_min - log_max) / (float)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = expf(log_max + (float)i * step);
            }
        }
    }
    else if (type == "uniform") {
        printf("Generating UNIFORMLY (linearly) distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            float step = (max_val - min_val) / (float)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (float)i * step;
            }
        }
    }
    else if (type == "cluster0") {
        printf("Generating 'Cluster0' singular values (sharp drop)...\n");
        // 前1/4的奇异值从 s_max 线性衰减到 s_max*0.9，其余的全部为 s_min
        size_t cutoff_rank = rank / 4;
        if (cutoff_rank == 0 && rank > 0) cutoff_rank = 1;
        
        float high_end_val = max_val * 0.9f;
        float step = (cutoff_rank > 1) ? (max_val - high_end_val) / (float)(cutoff_rank - 1) : 0.0f;
        
        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff_rank) {
                s_values[i] = max_val - (float)i * step;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == "cluster1") {
        printf("Generating 'Cluster1' singular values (staircase)...\n");
        // 三个阶梯：s_max, (s_max+s_min)/2, s_min
        size_t cutoff1 = rank / 3;
        size_t cutoff2 = 2 * rank / 3;
        float mid_val = (max_val + min_val) / 2.0f;

        for (size_t i = 0; i < rank; ++i) {
            if (i < cutoff1) {
                s_values[i] = max_val;
            } else if (i < cutoff2) {
                s_values[i] = mid_val;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == "arithmetic") {
        printf("Generating ARITHMETIC progression singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            float step = (max_val - min_val) / (float)(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - (float)i * step;
            }
        }
    }
    else if (type == "normal") {
        printf("Generating NORMAL (Gaussian-like) distributed singular values...\n");
        // 正态分布样式：在中间值附近集中，两端较小
        float mean = (float)rank / 2.0f;
        float sigma = (float)rank / 6.0f; 
        
        for (size_t i = 0; i < rank; ++i) {
            float x = (float)i;
            float gaussian_weight = expf(-0.5f * powf((x - mean) / sigma, 2.0f));
            s_values[i] = min_val + (max_val - min_val) * gaussian_weight;
        }
    }

}
cudaEvent_t begin, end;
void startTimer()
{
    cudaEventCreate(&begin);
    cudaEventRecord(begin);
    cudaEventCreate(&end);
}

float stopTimer()
{
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}

int main(int argc,char *argv[])
{
    const int parse_status = parseArguments(argc, argv);
    if (parse_status != 0) return parse_status > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
    const size_t mn = std::min(m, n);
    std::cout << "Constant mn: " << mn << std::endl;
    const size_t real_rank = rank; 
    printf("Creating a large low-rank matrix of size %ld x %ld with rank %ld.\n", m, n, real_rank);

    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cusolverDnHandle_t cusolverHandle;
    cusolverDnCreate(&cusolverHandle);

    float one = 1.0f;
    float zero = 0.0f;
    float minus_one = -1.0f;
    // 创建唯一的随机数生成器
    curandGenerator_t main_gen;
    curandCreateGenerator(&main_gen, CURAND_RNG_PSEUDO_DEFAULT);
    // 创建低秩矩阵 A , 首先生成两个随机矩阵(m*real_rank)和(real_rank*n),分别对其进行QR分解得到对应的Q1和Q2,
    thrust::device_vector<float> d_A(m * n);
    thrust::device_vector<__half> d_A_half;
    thrust::device_vector<__nv_bfloat16> d_A_bf16;
    std::vector<float> h_S_expected(real_rank);
    { 
        thrust::device_vector<float> factor1(m * real_rank);
        thrust::device_vector<float> factor2(n * real_rank);
        
        // 使用不同的固定种子生成 factor1 和 factor2
        curandSetPseudoRandomGeneratorSeed(main_gen, 1234ULL);
        generateNormalMatrix(main_gen, factor1, m, real_rank); 
        curandSetPseudoRandomGeneratorSeed(main_gen, 9999ULL);
        generateNormalMatrix(main_gen, factor2, n, real_rank); 
        //分别对两个随机矩阵进行正交化
        int lwork_geqrf1 = 0;
        int lwork_orgqr1 = 0;
        cusolverDnSgeqrf_bufferSize(cusolverHandle, m, real_rank, thrust::raw_pointer_cast(factor1.data()), m, &lwork_geqrf1);
        cusolverDnSorgqr_bufferSize(cusolverHandle, m, real_rank, real_rank, thrust::raw_pointer_cast(factor1.data()), m, thrust::raw_pointer_cast(factor1.data()), &lwork_orgqr1);
        int lwork1 = std::max(lwork_geqrf1, lwork_orgqr1);
        printf("Workspace for factor1 (m=%ld): geqrf=%d, orgqr=%d. Using max size: %d\n", m, lwork_geqrf1, lwork_orgqr1, lwork1);
        
        thrust::device_vector<float> work1(lwork1);
        thrust::device_vector<float> tau1(real_rank);
        thrust::device_vector<int> devInfo(1);
        // QR分解，构造Q1
        cusolverDnSgeqrf(cusolverHandle, m, real_rank, thrust::raw_pointer_cast(factor1.data()), m,
                         thrust::raw_pointer_cast(tau1.data()), thrust::raw_pointer_cast(work1.data()), lwork1,
                         thrust::raw_pointer_cast(devInfo.data()));
        cusolverDnSorgqr(cusolverHandle, m, real_rank, real_rank, thrust::raw_pointer_cast(factor1.data()), m,
                         thrust::raw_pointer_cast(tau1.data()), thrust::raw_pointer_cast(work1.data()), lwork1,
                         thrust::raw_pointer_cast(devInfo.data()));
        const thrust::device_vector<float>& Q1 = factor1; // 现在 factor1 存储的是正交矩阵 Q1
        
        int lwork_geqrf2 = 0;
        int lwork_orgqr2 = 0;
        cusolverDnSgeqrf_bufferSize(cusolverHandle, n, real_rank, thrust::raw_pointer_cast(factor2.data()), n, &lwork_geqrf2);
        cusolverDnSorgqr_bufferSize(cusolverHandle, n, real_rank, real_rank, thrust::raw_pointer_cast(factor2.data()), n, thrust::raw_pointer_cast(factor2.data()), &lwork_orgqr2);
        int lwork2 = std::max(lwork_geqrf2, lwork_orgqr2);
        printf("Workspace for factor2 (m=%ld): geqrf=%d, orgqr=%d. Using max size: %d\n", n, lwork_geqrf2, lwork_orgqr2, lwork2);
        thrust::device_vector<float> work2(lwork2);
        thrust::device_vector<float> tau2(real_rank);
        // QR分解，然后构造Q2
        cusolverDnSgeqrf(cusolverHandle, n, real_rank, thrust::raw_pointer_cast(factor2.data()), n,
                         thrust::raw_pointer_cast(tau2.data()), thrust::raw_pointer_cast(work2.data()), lwork2,
                         thrust::raw_pointer_cast(devInfo.data()));
        cusolverDnSorgqr(cusolverHandle, n, real_rank, real_rank, thrust::raw_pointer_cast(factor2.data()), n,
                         thrust::raw_pointer_cast(tau2.data()), thrust::raw_pointer_cast(work2.data()), lwork2,
                         thrust::raw_pointer_cast(devInfo.data()));
        const thrust::device_vector<float>& Q2 = factor2; // 现在 factor2 存储的是正交矩阵 Q2
        //定义奇异值S
        printf("Step 3: Defining a spectrum of singular values...\n");
        generate_singular_values(spectrum_type, s_max, s_min, h_S_expected, real_rank);
        thrust::device_vector<float> d_S = h_S_expected;
        printf("Constructed singular values (top 64):\n");
        size_t print_count = (real_rank < 64) ? real_rank : 64;
        for(size_t i=0; i<print_count; ++i) printf("  S_expected[%zu] = %f\n", i, h_S_expected[i]);

        // A = factor1 * diag(S) * factor2^T
        printf("Step 4: Assembling A = factor1 * diag(S) * factor2^T...\n");
        // Temp = factor1 * diag(S)
        thrust::device_vector<float> d_Temp(m * real_rank);
        dim3 threads(16, 16);
        dim3 blocks((real_rank + 15) / 16, (m + 15) / 16);
        scale_columns_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(d_Temp.data()),thrust::raw_pointer_cast(Q1.data()),
                                                  thrust::raw_pointer_cast(d_S.data()),m, real_rank, m, m);

        // 计算最终的 A = Temp * factor2^T
        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, real_rank,
                    &one, thrust::raw_pointer_cast(d_Temp.data()), m,
                    thrust::raw_pointer_cast(Q2.data()), n,
                    &zero, thrust::raw_pointer_cast(d_A.data()), m);      
        cudaDeviceSynchronize();
        printf("Matrix A constructed successfully using cuSolver.\n\n");
    }
    
    // 创建A的副本用于后续验证，因为cusolverDnXgesvdr会修改输入矩阵
    thrust::device_vector<float> d_A_backup(d_A.size());
    thrust::copy(d_A.begin(), d_A.end(), d_A_backup.begin());
    
    // =================================================================
    // ==== 对比实验 2: 使用 cusolverDnXgesvd (float版本) ====
    // =================================================================
    printf("\n\n===========================================================\n");
    printf("--- Comparison: Calling built-in cusolverDnXgesvd ---\n");
    printf("===========================================================\n");

    const size_t min_mn = std::min(m, n);
    
    signed char jobu_gesvd = 'A';  // 计算完整的U矩阵 (m x m)
    signed char jobvt_gesvd = 'A'; // 计算完整的V^T矩阵 (n x n)

    cusolverDnParams_t params_gesvd;
    cusolverDnCreateParams(&params_gesvd);

    thrust::device_vector<float> d_S_gesvd(min_mn);
    const size_t lda_gesvd = m;
    const size_t ldu_gesvd = m;
    const size_t ldvt_gesvd = n;
    thrust::device_vector<float> d_U_gesvd(ldu_gesvd * m);      // 完整的U矩阵
    thrust::device_vector<float> d_VT_gesvd(ldvt_gesvd * n);     // 完整的V^T矩阵

    size_t workspaceInBytesOnDevice_gesvd = 0;
    size_t workspaceInBytesOnHost_gesvd = 0;
    
    // 恢复A矩阵，因为gesvd会修改输入矩阵
    thrust::copy(d_A_backup.begin(), d_A_backup.end(), d_A.begin());

    cudaDataType dtype_A = CUDA_R_32F;
    void* d_A_input_ptr = nullptr;
    const size_t total_elems_A = static_cast<size_t>(m) * static_cast<size_t>(n);
    float quant_in_ms = 0.0f;
    float quant_out_ms = 0.0f;

    if (g_precision == PrecisionMode::Float32) {
        dtype_A = CUDA_R_32F;
        d_A_input_ptr = (void*)thrust::raw_pointer_cast(d_A.data());
    } else if (g_precision == PrecisionMode::Half16) {
        d_A_half.resize(total_elems_A);
        int threads = 256;
        int blocks = static_cast<int>((total_elems_A + threads - 1) / threads);
        startTimer();
        convert_float_to_half_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_A_half.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        cudaError_t convert_status = cudaGetLastError();
        if (convert_status != cudaSuccess) {
            printf("[Error] float->half conversion failed: %s\n", cudaGetErrorString(convert_status));
            return -1;
        }
        // 将量化后的半精度重新转回 float，模拟半精输入的量化误差
        convert_half_to_float_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_A_half.data()),
            thrust::raw_pointer_cast(d_A.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        convert_status = cudaGetLastError();
        if (convert_status != cudaSuccess) {
            printf("[Error] half->float conversion failed: %s\n", cudaGetErrorString(convert_status));
            return -1;
        }
        quant_in_ms = stopTimer();
        d_A_input_ptr = (void*)thrust::raw_pointer_cast(d_A.data());
    } else {
        d_A_bf16.resize(total_elems_A);
        int threads = 256;
        int blocks = static_cast<int>((total_elems_A + threads - 1) / threads);
        startTimer();
        convert_float_to_bfloat16_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_A_bf16.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        cudaError_t convert_status = cudaGetLastError();
        if (convert_status != cudaSuccess) {
            printf("[Error] float->bfloat16 conversion failed: %s\n", cudaGetErrorString(convert_status));
            return -1;
        }
        convert_bfloat16_to_float_kernel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_A_bf16.data()),
            thrust::raw_pointer_cast(d_A.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        convert_status = cudaGetLastError();
        if (convert_status != cudaSuccess) {
            printf("[Error] bfloat16->float conversion failed: %s\n", cudaGetErrorString(convert_status));
            return -1;
        }
        quant_in_ms = stopTimer();
        d_A_input_ptr = (void*)thrust::raw_pointer_cast(d_A.data());
    }
    dtype_A = CUDA_R_32F;
    
    cusolverDnXgesvd_bufferSize(cusolverHandle, params_gesvd,
                                jobu_gesvd, jobvt_gesvd,
                                m, n,
                                dtype_A, (const void*)d_A_input_ptr, (int64_t)lda_gesvd,
                                CUDA_R_32F, (const void*)thrust::raw_pointer_cast(d_S_gesvd.data()),
                                CUDA_R_32F, (const void*)thrust::raw_pointer_cast(d_U_gesvd.data()), (int64_t)ldu_gesvd,
                                CUDA_R_32F, (const void*)thrust::raw_pointer_cast(d_VT_gesvd.data()), (int64_t)ldvt_gesvd,
                                CUDA_R_32F,
                                &workspaceInBytesOnDevice_gesvd,
                                &workspaceInBytesOnHost_gesvd
                               );
    printf("gesvd device workspace: %zu bytes, host workspace: %zu bytes\n",
           workspaceInBytesOnDevice_gesvd, workspaceInBytesOnHost_gesvd);

    thrust::device_vector<char> bufferOnDevice_gesvd(workspaceInBytesOnDevice_gesvd);
    std::vector<char> bufferOnHost_gesvd;
    if (workspaceInBytesOnHost_gesvd > 0) {
        bufferOnHost_gesvd.resize(workspaceInBytesOnHost_gesvd);
    }

    thrust::device_vector<int> d_info_gesvd(1);
    
    startTimer();
    cusolverStatus_t gesvd_status = cusolverDnXgesvd(cusolverHandle, params_gesvd,
                                                     jobu_gesvd, jobvt_gesvd,
                                                     (int64_t)m, (int64_t)n,
                                                     dtype_A, d_A_input_ptr, (int64_t)lda_gesvd,
                                                     CUDA_R_32F, (void*)thrust::raw_pointer_cast(d_S_gesvd.data()),
                                                     CUDA_R_32F, (void*)thrust::raw_pointer_cast(d_U_gesvd.data()), (int64_t)ldu_gesvd,
                                                     CUDA_R_32F, (void*)thrust::raw_pointer_cast(d_VT_gesvd.data()), (int64_t)ldvt_gesvd,
                                                     CUDA_R_32F,
                                                     (void*)thrust::raw_pointer_cast(bufferOnDevice_gesvd.data()), workspaceInBytesOnDevice_gesvd,
                                                     (workspaceInBytesOnHost_gesvd > 0) ? (void*)bufferOnHost_gesvd.data() : nullptr, workspaceInBytesOnHost_gesvd,
                                                     thrust::raw_pointer_cast(d_info_gesvd.data())
                                                    );
    cudaDeviceSynchronize();
    float elapsed_time_gesvd = stopTimer();

    if (g_precision == PrecisionMode::Half16) {
        thrust::device_vector<__half> d_U_half(ldu_gesvd * m);
        thrust::device_vector<__half> d_VT_half(ldvt_gesvd * n);
        thrust::device_vector<__half> d_S_half(min_mn);
        int threads = 256;
        int blocks_u = static_cast<int>(((ldu_gesvd * (size_t)m) + threads - 1) / threads);
        int blocks_v = static_cast<int>(((ldvt_gesvd * (size_t)n) + threads - 1) / threads);
        int blocks_s = static_cast<int>(((size_t)min_mn + threads - 1) / threads);
        startTimer();
        convert_float_to_half_kernel<<<blocks_u, threads>>>(
            thrust::raw_pointer_cast(d_U_gesvd.data()),
            thrust::raw_pointer_cast(d_U_half.data()),
            ldu_gesvd * (size_t)m);
        convert_float_to_half_kernel<<<blocks_v, threads>>>(
            thrust::raw_pointer_cast(d_VT_gesvd.data()),
            thrust::raw_pointer_cast(d_VT_half.data()),
            ldvt_gesvd * (size_t)n);
        convert_float_to_half_kernel<<<blocks_s, threads>>>(
            thrust::raw_pointer_cast(d_S_gesvd.data()),
            thrust::raw_pointer_cast(d_S_half.data()),
            min_mn);
        cudaDeviceSynchronize();
        cudaError_t out_status = cudaGetLastError();
        if (out_status != cudaSuccess) {
            printf("[Error] float->half output conversion failed: %s\n", cudaGetErrorString(out_status));
            return -1;
        }
        quant_out_ms = stopTimer();
    } else if (g_precision == PrecisionMode::BFloat16) {
        thrust::device_vector<__nv_bfloat16> d_U_bf16(ldu_gesvd * m);
        thrust::device_vector<__nv_bfloat16> d_VT_bf16(ldvt_gesvd * n);
        thrust::device_vector<__nv_bfloat16> d_S_bf16(min_mn);
        int threads = 256;
        int blocks_u = static_cast<int>(((ldu_gesvd * (size_t)m) + threads - 1) / threads);
        int blocks_v = static_cast<int>(((ldvt_gesvd * (size_t)n) + threads - 1) / threads);
        int blocks_s = static_cast<int>(((size_t)min_mn + threads - 1) / threads);
        startTimer();
        convert_float_to_bfloat16_kernel<<<blocks_u, threads>>>(
            thrust::raw_pointer_cast(d_U_gesvd.data()),
            thrust::raw_pointer_cast(d_U_bf16.data()),
            ldu_gesvd * (size_t)m);
        convert_float_to_bfloat16_kernel<<<blocks_v, threads>>>(
            thrust::raw_pointer_cast(d_VT_gesvd.data()),
            thrust::raw_pointer_cast(d_VT_bf16.data()),
            ldvt_gesvd * (size_t)n);
        convert_float_to_bfloat16_kernel<<<blocks_s, threads>>>(
            thrust::raw_pointer_cast(d_S_gesvd.data()),
            thrust::raw_pointer_cast(d_S_bf16.data()),
            min_mn);
        cudaDeviceSynchronize();
        cudaError_t out_status = cudaGetLastError();
        if (out_status != cudaSuccess) {
            printf("[Error] float->bfloat16 output conversion failed: %s\n", cudaGetErrorString(out_status));
            return -1;
        }
        quant_out_ms = stopTimer();
    }

    float total_time_ms = quant_in_ms + elapsed_time_gesvd + quant_out_ms;

    printf("cusolverDnXgesvd finished in %f ms (SVD compute).\n", elapsed_time_gesvd);
    if (g_precision != PrecisionMode::Float32) {
        printf("  Quantization (input) time : %f ms\n", quant_in_ms);
        printf("  Re-quantization (output) time : %f ms\n", quant_out_ms);
        printf("  Total (quant + SVD + re-quant) : %f ms\n", total_time_ms);
    }
    
    int info_host_gesvd = 0;
    cudaMemcpy(&info_host_gesvd, thrust::raw_pointer_cast(d_info_gesvd.data()), sizeof(int), cudaMemcpyDeviceToHost);
    if(gesvd_status != CUSOLVER_STATUS_SUCCESS || info_host_gesvd != 0) {
        printf("!!! cusolverDnXgesvd failed! Status: %d, Info: %d !!!\n", gesvd_status, info_host_gesvd);
        // 清理资源
        cusolverDnDestroyParams(params_gesvd);
        curandDestroyGenerator(main_gen);
        cublasDestroy(cublasHandle);
        cusolverDnDestroy(cusolverHandle);
        return -1;
    } else {
    
        printf("\n--- Verifying results from cusolverDnXgesvd ---\n");
        
        // 简单的后向误差验证
        // 1. 验证 U 的正交性 (完整的U矩阵: m x m)
        thrust::device_vector<float> d_UtU_gesvd(m * m);
        cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                    m, m, m,
                    &one, thrust::raw_pointer_cast(d_U_gesvd.data()), m,
                    thrust::raw_pointer_cast(d_U_gesvd.data()), m,
                    &zero, thrust::raw_pointer_cast(d_UtU_gesvd.data()), m);
        
        // 对角线减1
        std::vector<float> h_UtU_gesvd(m * m);
        thrust::copy(d_UtU_gesvd.begin(), d_UtU_gesvd.end(), h_UtU_gesvd.begin());
        for(size_t i = 0; i < m; ++i) {
            h_UtU_gesvd[i * m + i] -= 1.0f;
        }
        thrust::copy(h_UtU_gesvd.begin(), h_UtU_gesvd.end(), d_UtU_gesvd.begin());
        
        float u_ortho_error_gesvd;
        cublasSnrm2(cublasHandle, m * m,
                    thrust::raw_pointer_cast(d_UtU_gesvd.data()), 1, &u_ortho_error_gesvd);
        printf("Orthogonality of U ||U'*U - I||_F = %e\n", u_ortho_error_gesvd);
        
        // 2. 验证 V^T 的正交性 (完整的V^T矩阵: n x n)
        thrust::device_vector<float> d_VVt_gesvd(n * n);
        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    n, n, n,
                    &one, thrust::raw_pointer_cast(d_VT_gesvd.data()), n,
                    thrust::raw_pointer_cast(d_VT_gesvd.data()), n,
                    &zero, thrust::raw_pointer_cast(d_VVt_gesvd.data()), n);
        
        // 对角线减1
        std::vector<float> h_VVt_gesvd(n * n);
        thrust::copy(d_VVt_gesvd.begin(), d_VVt_gesvd.end(), h_VVt_gesvd.begin());
        for(size_t i = 0; i < n; ++i) {
            h_VVt_gesvd[i * n + i] -= 1.0f;
        }
        thrust::copy(h_VVt_gesvd.begin(), h_VVt_gesvd.end(), d_VVt_gesvd.begin());
        
        float v_ortho_error_gesvd;
        cublasSnrm2(cublasHandle, n * n,
                    thrust::raw_pointer_cast(d_VVt_gesvd.data()), 1, &v_ortho_error_gesvd);
        printf("Orthogonality of V ||V*V' - I||_F = %e\n", v_ortho_error_gesvd);
        
        // 3. 验证重构误差
        // A_approx = U * S * V^T (使用前min_mn列的U和前min_mn行的V^T)
        thrust::device_vector<float> d_U_truncated(m * min_mn);
        thrust::device_vector<float> d_VT_truncated(min_mn * n);
        
        // 提取U的前min_mn列
        cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, min_mn,
                    &one, thrust::raw_pointer_cast(d_U_gesvd.data()), m,
                    &zero, thrust::raw_pointer_cast(d_U_gesvd.data()), m,
                    thrust::raw_pointer_cast(d_U_truncated.data()), m);
        
        // 提取V^T的前min_mn行
        cublasSgeam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    min_mn, n,
                    &one, thrust::raw_pointer_cast(d_VT_gesvd.data()), n,
                    &zero, thrust::raw_pointer_cast(d_VT_gesvd.data()), n,
                    thrust::raw_pointer_cast(d_VT_truncated.data()), min_mn);
        
        thrust::device_vector<float> d_US_gesvd(m * min_mn);
        dim3 threads_scale_gesvd(16, 16);
        dim3 blocks_scale_gesvd((min_mn + 15) / 16, (m + 15) / 16);
        scale_columns_kernel<<<blocks_scale_gesvd, threads_scale_gesvd>>>(
            thrust::raw_pointer_cast(d_US_gesvd.data()),
            thrust::raw_pointer_cast(d_U_truncated.data()),
            thrust::raw_pointer_cast(d_S_gesvd.data()),
            m, min_mn, m, m);
        
        thrust::device_vector<float> d_A_approx_gesvd(m * n);
        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, min_mn,
                    &one, thrust::raw_pointer_cast(d_US_gesvd.data()), m,
                    thrust::raw_pointer_cast(d_VT_truncated.data()), min_mn,
                    &zero, thrust::raw_pointer_cast(d_A_approx_gesvd.data()), m);
        
        // 计算重构误差 ||A - USV^T||_F
        // 使用备份的原始矩阵A进行计算
        thrust::device_vector<float> d_error_matrix_gesvd(m * n);
        thrust::copy(d_A_backup.begin(), d_A_backup.end(), d_error_matrix_gesvd.begin());
        
        // 计算 d_error_matrix_gesvd = A_backup - A_approx_gesvd = A - USV^T
        cublasSaxpy(cublasHandle, m * n, &minus_one,
                    thrust::raw_pointer_cast(d_A_approx_gesvd.data()), 1,
                    thrust::raw_pointer_cast(d_error_matrix_gesvd.data()), 1);
        
        // 计算重构误差的范数
        float recon_error_gesvd;
        cublasSnrm2(cublasHandle, m * n, thrust::raw_pointer_cast(d_error_matrix_gesvd.data()), 1, &recon_error_gesvd);
        
        // 计算原始矩阵A的范数
        float norm_A_gesvd;
        cublasSnrm2(cublasHandle, m * n, thrust::raw_pointer_cast(d_A_backup.data()), 1, &norm_A_gesvd);
        
        printf("Reconstruction error ||A - USV^T||_F = %e\n", recon_error_gesvd);
        printf("Relative reconstruction error ||A - USV^T||_F / ||A||_F = %e\n", recon_error_gesvd / norm_A_gesvd);
        
        // 打印前几个奇异值
        std::vector<float> h_S_gesvd(min_mn);
        thrust::copy(d_S_gesvd.begin(), d_S_gesvd.end(), h_S_gesvd.begin());
        printf("\nComputed singular values (first 10):\n");
        size_t print_count_computed_gesvd = (min_mn < 10) ? min_mn : 10;
        for(size_t i = 0; i < print_count_computed_gesvd; ++i) {
            printf("  s[%zu] = %e\n", i, h_S_gesvd[i]);
        }
        
        // 与真实奇异值比较
        printf("\nComparison with expected singular values:\n");
        size_t compare_count_gesvd = min_mn;
        if (compare_count_gesvd > h_S_expected.size()) compare_count_gesvd = h_S_expected.size();
        if (compare_count_gesvd > 10) compare_count_gesvd = 10;
        
        for(size_t i = 0; i < compare_count_gesvd; ++i) {
            printf("  Expected s[%zu] = %e, Computed s[%zu] = %e, Error = %e\n",
                   i, h_S_expected[i], i, h_S_gesvd[i],
                   fabsf(h_S_expected[i] - h_S_gesvd[i]));
        }
    }

    // 清理资源
    cusolverDnDestroyParams(params_gesvd);
    curandDestroyGenerator(main_gen);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    printf("\n=== cusolverDnXgesvd Test completed ===\n");
    return 0;
}
