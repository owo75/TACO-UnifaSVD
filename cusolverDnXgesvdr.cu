// Randomized-SVD benchmark. Half/BF16 modes quantize inputs and time output
// casts; the cuSOLVER factorization itself uses FP32.
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
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstring>
#include "src/matrix_io.h"
#include "src/cli_utils.h"

long int m, n, rank;
enum class PrecisionMode { Float32, Half16, BFloat16 };
PrecisionMode g_precision = PrecisionMode::Float32;
static bool g_use_custom_matrix = false;
static std::string g_matrix_file_path;
static MatrixFileFormat g_matrix_file_format = MatrixFileFormat::Text;
static MatrixFileLayout g_matrix_file_layout = MatrixFileLayout::RowMajor;

__global__ void convert_float_to_half_kernel(const float* __restrict__ src, __half* __restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void convert_half_to_float_kernel(const __half* __restrict__ src, float* __restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

__global__ void convert_float_to_bfloat16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2bfloat16(src[idx]);
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
// Power iterations for cusolverDnXgesvdr. Default 10 matches the original code.
int g_niters = 10;
int parseArguments(int argc,char *argv[])
{
    const bool show_help = cli::help_requested(argc, argv);
    if(argc < 2 || show_help)
    {
        printf("Usage (generated matrix): %s <m> <n> <rank|auto> <spectrum_type> <s_max> <s_min> [--precision=<float|half|bfloat16>] [--niters=<N>]\n", argv[0]);
        printf("Usage (matrix file)    : %s <m|auto> <n|auto> <rank|auto> --matrix-file=<path> [--matrix-format=text|binary|libsvm] [--matrix-layout=row|col] [--precision=<float|half|bfloat16>] [--niters=<N>]\n", argv[0]);
        printf("  <spectrum_type>: 'geometric', 'uniform', 'cluster0', 'cluster1', 'arithmetic', 'normal'\n");
        printf("  <s_max>: Maximum singular value (e.g., 1.0)\n");
        printf("  <s_min>: Minimum singular value (e.g., 1e-10)\n");
        printf("  --niters=<N>: Power iterations for cusolverDnXgesvdr (default 10)\n");
        return show_help ? 1 : -1;
    }
    std::vector<std::string> pos_args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--precision=", 0) == 0) {
            std::string val = arg.substr(strlen("--precision="));
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "float" || val == "fp32") g_precision = PrecisionMode::Float32;
            else if (val == "half" || val == "fp16" || val == "16f") g_precision = PrecisionMode::Half16;
            else if (val == "bfloat16" || val == "bf16") g_precision = PrecisionMode::BFloat16;
            else {
                printf("Error: Unknown precision '%s'. Use float, half, or bfloat16.\n", val.c_str());
                return -1;
            }
        } else if (arg == "--matrix-file" || arg == "-matrix-file") {
            if (i + 1 < argc) {
                g_use_custom_matrix = true;
                g_matrix_file_path = argv[++i];
            } else {
                printf("[Error] --matrix-file requires a path argument.\n");
                return -1;
            }
        } else if (arg.rfind("--matrix-file=", 0) == 0 || arg.rfind("-matrix-file=", 0) == 0) {
            size_t eq_pos = arg.find('=');
            if (eq_pos == std::string::npos || eq_pos + 1 >= arg.size()) {
                printf("[Error] --matrix-file requires a non-empty path.\n");
                return -1;
            }
            g_use_custom_matrix = true;
            g_matrix_file_path = arg.substr(eq_pos + 1);
        } else if (arg.rfind("--matrix-format=", 0) == 0 || arg.rfind("-matrix-format=", 0) == 0) {
            std::string val = arg.substr(arg.find('=') + 1);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "text" || val == "txt") {
                g_matrix_file_format = MatrixFileFormat::Text;
            } else if (val == "binary" || val == "bin") {
                g_matrix_file_format = MatrixFileFormat::Binary;
            } else if (val == "libsvm" || val == "svm") {
                g_matrix_file_format = MatrixFileFormat::LibSVM;
            } else {
                printf("Error: Unknown matrix format '%s'. Use text, binary, or libsvm.\n", val.c_str());
                return -1;
            }
        } else if (arg.rfind("--matrix-layout=", 0) == 0 || arg.rfind("-matrix-layout=", 0) == 0) {
            std::string val = arg.substr(arg.find('=') + 1);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "row" || val == "row-major" || val == "r") {
                g_matrix_file_layout = MatrixFileLayout::RowMajor;
            } else if (val == "col" || val == "column" || val == "column-major" || val == "c") {
                g_matrix_file_layout = MatrixFileLayout::ColumnMajor;
            } else {
                printf("Error: Unknown matrix layout '%s'. Use row or col.\n", val.c_str());
                return -1;
            }
        } else if (arg.rfind("--niters=", 0) == 0) {
            std::string val = arg.substr(strlen("--niters="));
            long parsed = 0;
            if (!cli::positive_integer(val, "--niters", parsed)) return -1;
            g_niters = (int)parsed;
        } else if (!arg.empty() && arg[0] == '-') {
            std::fprintf(stderr, "Error: Unrecognized option '%s'. Use --help for usage.\n", arg.c_str());
            return -1;
        } else {
            pos_args.push_back(arg);
        }
    }

    auto is_auto_token = [](const std::string& s) {
        std::string v = s;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return v == "auto" || v == "unknown";
    };

    if (!g_use_custom_matrix) {
        if (pos_args.size() != 6) {
            printf("Error: Expected 6 generated-matrix arguments, got %zu. Use --help for usage.\n", pos_args.size());
            return -1;
        }
        if (!cli::positive_integer(pos_args[0], "m", m) ||
            !cli::positive_integer(pos_args[1], "n", n)) return -1;
        if (is_auto_token(pos_args[2])) {
            rank = std::min(m, n);
        } else {
            if (!cli::positive_integer(pos_args[2], "rank", rank)) return -1;
        }
        spectrum_type = pos_args[3];
        if (!cli::positive_real(pos_args[4], "s_max", s_max) ||
            !cli::positive_real(pos_args[5], "s_min", s_min)) return -1;
        if (s_max < s_min) {
            std::fprintf(stderr, "Error: s_max must be greater than or equal to s_min.\n");
            return -1;
        }
    } else {
        if (pos_args.size() != 3) {
            printf("Error: Missing required matrix-file arguments. Expected <m|auto> <n|auto> <rank|auto>.\n");
            return -1;
        }
        bool m_auto = is_auto_token(pos_args[0]);
        bool n_auto = is_auto_token(pos_args[1]);
        if ((!m_auto && !cli::positive_integer(pos_args[0], "m", m)) ||
            (!n_auto && !cli::positive_integer(pos_args[1], "n", n))) return -1;
        if (m_auto || n_auto) {
            if (g_matrix_file_format == MatrixFileFormat::Binary) {
                printf("Error: Binary matrix files require explicit m and n.\n");
                return -1;
            }
            long int inferred_rows = 0, inferred_cols = 0;
            bool ok = (g_matrix_file_format == MatrixFileFormat::LibSVM)
                          ? infer_libsvm_shape(g_matrix_file_path, inferred_rows, inferred_cols)
                          : infer_text_matrix_shape(g_matrix_file_path, g_matrix_file_layout, inferred_rows, inferred_cols);
            if (!ok) return -1;
            if ((!m_auto && m != inferred_rows) || (!n_auto && n != inferred_cols)) {
                std::fprintf(stderr, "Error: Explicit dimensions do not match the matrix file (%ld x %ld).\n",
                             inferred_rows, inferred_cols);
                return -1;
            }
            m = inferred_rows;
            n = inferred_cols;
        }
        if (is_auto_token(pos_args[2])) {
            rank = std::min(m, n);
        } else {
            if (!cli::positive_integer(pos_args[2], "rank", rank)) return -1;
        }
        spectrum_type = "custom";
        s_max = 1.0f;
        s_min = 1.0f;
    }

    printf("Parsed arguments:\n");
    printf("  m = %ld, n = %ld, rank = %ld\n", m, n, rank);
    const char* precision_str = (g_precision == PrecisionMode::Float32) ? "float32" : (g_precision == PrecisionMode::Half16 ? "half" : "bfloat16");
    printf("  Precision: %s\n", precision_str);
    if (g_use_custom_matrix) {
        const char* fmt = (g_matrix_file_format == MatrixFileFormat::Text) ? "text" :
                          (g_matrix_file_format == MatrixFileFormat::Binary ? "binary" : "libsvm");
        const char* layout = (g_matrix_file_layout == MatrixFileLayout::RowMajor) ? "row-major" : "column-major";
        printf("  Matrix Source: user file '%s' (format=%s, layout=%s)\n\n",
               g_matrix_file_path.c_str(), fmt, layout);
    } else {
        printf("  Spectrum Type: %s\n", spectrum_type.c_str());
        printf("  Singular Value Range: [%e, %e]\n\n", s_max, s_min);
    }

    if (!g_use_custom_matrix) {
        if (spectrum_type != "geometric" && spectrum_type != "uniform" &&
            spectrum_type != "cluster0" && spectrum_type != "cluster1" &&
            spectrum_type != "arithmetic" && spectrum_type != "normal") {
            printf("Error: Invalid spectrum_type '%s'.\n", spectrum_type.c_str());
            return -1;
        }
    }

    if (m <= 0 || n <= 0 || m > INT_MAX || n > INT_MAX ||
        rank <= 0 || rank > std::min(m, n)) {
        printf("Error: Require dimensions in [1, INT_MAX] and 0<rank<=min(m,n).\n");
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
    if (!g_use_custom_matrix) {
        printf("Creating a large low-rank matrix of size %ld x %ld with rank %ld.\n", m, n, real_rank);
    } else {
        printf("Loading input matrix of size %ld x %ld from '%s'.\n", m, n, g_matrix_file_path.c_str());
    }

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
    if (!g_use_custom_matrix) {
        thrust::device_vector<float> factor1(m * real_rank);
        thrust::device_vector<float> factor2(n * real_rank);
        
        curandSetPseudoRandomGeneratorSeed(main_gen, 1234ULL);
        generateNormalMatrix(main_gen, factor1, m, real_rank); 
        curandSetPseudoRandomGeneratorSeed(main_gen, 9999ULL);
        generateNormalMatrix(main_gen, factor2, n, real_rank); 
        int lwork_geqrf1 = 0;
        int lwork_orgqr1 = 0;
        cusolverDnSgeqrf_bufferSize(cusolverHandle, m, real_rank, thrust::raw_pointer_cast(factor1.data()), m, &lwork_geqrf1);
        cusolverDnSorgqr_bufferSize(cusolverHandle, m, real_rank, real_rank, thrust::raw_pointer_cast(factor1.data()), m, thrust::raw_pointer_cast(factor1.data()), &lwork_orgqr1);
        int lwork1 = std::max(lwork_geqrf1, lwork_orgqr1);
        printf("Workspace for factor1 (m=%ld): geqrf=%d, orgqr=%d. Using max size: %d\n", m, lwork_geqrf1, lwork_orgqr1, lwork1);
        
        thrust::device_vector<float> work1(lwork1);
        thrust::device_vector<float> tau1(real_rank);
        thrust::device_vector<int> devInfo(1);
        cusolverDnSgeqrf(cusolverHandle, m, real_rank, thrust::raw_pointer_cast(factor1.data()), m,
                         thrust::raw_pointer_cast(tau1.data()), thrust::raw_pointer_cast(work1.data()), lwork1,
                         thrust::raw_pointer_cast(devInfo.data()));
        cusolverDnSorgqr(cusolverHandle, m, real_rank, real_rank, thrust::raw_pointer_cast(factor1.data()), m,
                         thrust::raw_pointer_cast(tau1.data()), thrust::raw_pointer_cast(work1.data()), lwork1,
                         thrust::raw_pointer_cast(devInfo.data()));
        const thrust::device_vector<float>& Q1 = factor1;
        
        int lwork_geqrf2 = 0;
        int lwork_orgqr2 = 0;
        cusolverDnSgeqrf_bufferSize(cusolverHandle, n, real_rank, thrust::raw_pointer_cast(factor2.data()), n, &lwork_geqrf2);
        cusolverDnSorgqr_bufferSize(cusolverHandle, n, real_rank, real_rank, thrust::raw_pointer_cast(factor2.data()), n, thrust::raw_pointer_cast(factor2.data()), &lwork_orgqr2);
        int lwork2 = std::max(lwork_geqrf2, lwork_orgqr2);
        printf("Workspace for factor2 (m=%ld): geqrf=%d, orgqr=%d. Using max size: %d\n", n, lwork_geqrf2, lwork_orgqr2, lwork2);
        thrust::device_vector<float> work2(lwork2);
        thrust::device_vector<float> tau2(real_rank);
        cusolverDnSgeqrf(cusolverHandle, n, real_rank, thrust::raw_pointer_cast(factor2.data()), n,
                         thrust::raw_pointer_cast(tau2.data()), thrust::raw_pointer_cast(work2.data()), lwork2,
                         thrust::raw_pointer_cast(devInfo.data()));
        cusolverDnSorgqr(cusolverHandle, n, real_rank, real_rank, thrust::raw_pointer_cast(factor2.data()), n,
                         thrust::raw_pointer_cast(tau2.data()), thrust::raw_pointer_cast(work2.data()), lwork2,
                         thrust::raw_pointer_cast(devInfo.data()));
        const thrust::device_vector<float>& Q2 = factor2;
        printf("Step 3: Defining a spectrum of singular values...\n");
        generate_singular_values(spectrum_type, s_max, s_min, h_S_expected, real_rank);
        thrust::device_vector<float> d_S = h_S_expected;
        printf("Constructed singular values (top 64):\n");
        size_t print_count = (real_rank < 64) ? real_rank : 64;
        for(size_t i=0; i<print_count; ++i) printf("  S_expected[%zu] = %f\n", i, h_S_expected[i]);

        printf("Step 4: Assembling A = factor1 * diag(S) * factor2^T...\n");
        thrust::device_vector<float> d_Temp(m * real_rank);
        dim3 threads(16, 16);
        dim3 blocks((real_rank + 15) / 16, (m + 15) / 16);
        scale_columns_kernel<<<blocks, threads>>>(thrust::raw_pointer_cast(d_Temp.data()),thrust::raw_pointer_cast(Q1.data()),
                                                  thrust::raw_pointer_cast(d_S.data()),m, real_rank, m, m);

        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, real_rank,
                    &one, thrust::raw_pointer_cast(d_Temp.data()), m,
                    thrust::raw_pointer_cast(Q2.data()), n,
                    &zero, thrust::raw_pointer_cast(d_A.data()), m);      
        cudaDeviceSynchronize();
        printf("Matrix A constructed successfully using cuSolver.\n\n");
    } else {
        if (!load_matrix_from_file<float>(nullptr,
                                          m,
                                          m,
                                          n,
                                          g_matrix_file_path,
                                          g_matrix_file_format,
                                          g_matrix_file_layout,
                                          d_A)) {
            printf("[Error] Aborting cusolverDnXgesvdr due to matrix loading failure.\n");
            return -1;
        }
        printf("Matrix A loaded successfully from file.\n\n");
    }
    
    // 创建A的副本用于后续验证，因为cusolverDnXgesvdr会修改输入矩阵
    thrust::device_vector<float> d_A_backup(d_A.size());
    thrust::copy(d_A.begin(), d_A.end(), d_A_backup.begin());
    
    // =================================================================
    // ==== 对比实验 1: 使用 cusolverDnXgesvdr (float版本) ====
    // =================================================================
    printf("\n\n===========================================================\n");
    printf("--- Comparison: Calling built-in cusolverDnXgesvdr ---\n");
    printf("===========================================================\n");

    int64_t target_rank_gesvdr = rank;  // 目标秩：要计算的奇异值数量
    int64_t p_oversampling = 2*rank;       // 过采样：额外计算随机向量提高精度
    const int64_t max_valid_oversampling = std::max<int64_t>(0, static_cast<int64_t>(mn) - target_rank_gesvdr);
    if (p_oversampling > max_valid_oversampling) {
        p_oversampling = max_valid_oversampling;
    }
    int64_t niters = (int64_t)g_niters;        // 幂迭代次数（可由 --niters 指定）
    printf("gesvdr config: k=%lld, p=%lld, niters=%lld (mn=%zu)\n",
           (long long)target_rank_gesvdr,
           (long long)p_oversampling,
           (long long)niters,
           mn);

    signed char jobu_gesvdr = 'S';
    signed char jobv_gesvdr = 'S';

    cusolverDnParams_t params_gesvdr;
    cusolverDnCreateParams(&params_gesvdr);

    thrust::device_vector<float> d_S_gesvdr(target_rank_gesvdr);
    thrust::device_vector<float> d_U_gesvdr(m * target_rank_gesvdr);
    thrust::device_vector<float> d_V_gesvdr(n * target_rank_gesvdr);
    thrust::device_vector<__half> d_U_half;
    thrust::device_vector<__half> d_V_half;
    thrust::device_vector<__half> d_S_half;
    thrust::device_vector<__nv_bfloat16> d_U_bf16;
    thrust::device_vector<__nv_bfloat16> d_V_bf16;
    thrust::device_vector<__nv_bfloat16> d_S_bf16;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;
    
    float quant_in_ms = 0.0f;
    float quant_out_ms = 0.0f;
    const size_t total_elems_A = static_cast<size_t>(m) * static_cast<size_t>(n);
    cudaDataType dtype_A = CUDA_R_32F;
    void* d_A_input_ptr = (void*)thrust::raw_pointer_cast(d_A.data());
    if (g_precision == PrecisionMode::Half16) {
        d_A_half.resize(total_elems_A);
        int threads_conv = 256;
        int blocks_conv = static_cast<int>((total_elems_A + threads_conv - 1) / threads_conv);
        startTimer();
        convert_float_to_half_kernel<<<blocks_conv, threads_conv>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_A_half.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        cudaError_t conv_stat = cudaGetLastError();
        if (conv_stat != cudaSuccess) {
            printf("[Error] float->half conversion failed: %s\n", cudaGetErrorString(conv_stat));
            return -1;
        }
        convert_half_to_float_kernel<<<blocks_conv, threads_conv>>>(
            thrust::raw_pointer_cast(d_A_half.data()),
            thrust::raw_pointer_cast(d_A.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        conv_stat = cudaGetLastError();
        if (conv_stat != cudaSuccess) {
            printf("[Error] half->float conversion failed: %s\n", cudaGetErrorString(conv_stat));
            return -1;
        }
        quant_in_ms = stopTimer();
        d_A_input_ptr = (void*)thrust::raw_pointer_cast(d_A.data());
    } else if (g_precision == PrecisionMode::BFloat16) {
        d_A_bf16.resize(total_elems_A);
        int threads_conv = 256;
        int blocks_conv = static_cast<int>((total_elems_A + threads_conv - 1) / threads_conv);
        startTimer();
        convert_float_to_bfloat16_kernel<<<blocks_conv, threads_conv>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_A_bf16.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        cudaError_t conv_stat = cudaGetLastError();
        if (conv_stat != cudaSuccess) {
            printf("[Error] float->bfloat16 conversion failed: %s\n", cudaGetErrorString(conv_stat));
            return -1;
        }
        convert_bfloat16_to_float_kernel<<<blocks_conv, threads_conv>>>(
            thrust::raw_pointer_cast(d_A_bf16.data()),
            thrust::raw_pointer_cast(d_A.data()),
            total_elems_A);
        cudaDeviceSynchronize();
        conv_stat = cudaGetLastError();
        if (conv_stat != cudaSuccess) {
            printf("[Error] bfloat16->float conversion failed: %s\n", cudaGetErrorString(conv_stat));
            return -1;
        }
        quant_in_ms = stopTimer();
        d_A_input_ptr = (void*)thrust::raw_pointer_cast(d_A.data());
    }
    dtype_A = CUDA_R_32F;

    cusolverDnXgesvdr_bufferSize(cusolverHandle, params_gesvdr,
                                 jobu_gesvdr, jobv_gesvdr,
                                 m, n, target_rank_gesvdr, p_oversampling, niters,
                                 dtype_A, (const void*)d_A_input_ptr, m,
                                 CUDA_R_32F, (const void*)thrust::raw_pointer_cast(d_S_gesvdr.data()),
                                 CUDA_R_32F, (const void*)thrust::raw_pointer_cast(d_U_gesvdr.data()), m,
                                 CUDA_R_32F, (const void*)thrust::raw_pointer_cast(d_V_gesvdr.data()), n,
                                 CUDA_R_32F,
                                 &workspaceInBytesOnDevice,
                                 &workspaceInBytesOnHost
                                );
    printf("gesvdr device workspace: %zu bytes, host workspace: %zu bytes\n",
           workspaceInBytesOnDevice, workspaceInBytesOnHost);

    thrust::device_vector<char> bufferOnDevice(workspaceInBytesOnDevice);
    std::vector<char> bufferOnHost;
    if (workspaceInBytesOnHost > 0) {
        bufferOnHost.resize(workspaceInBytesOnHost);
    }

    thrust::device_vector<int> d_info_gesvdr(1);
    
    startTimer();
    cusolverStatus_t gesvdr_status = cusolverDnXgesvdr(cusolverHandle, params_gesvdr,
                                                       jobu_gesvdr, jobv_gesvdr,
                                                       (int64_t)m, (int64_t)n, target_rank_gesvdr, p_oversampling, niters,
                                                       dtype_A, (void*)d_A_input_ptr, (int64_t)m,
                                                       CUDA_R_32F, (void*)thrust::raw_pointer_cast(d_S_gesvdr.data()),
                                                       CUDA_R_32F, (void*)thrust::raw_pointer_cast(d_U_gesvdr.data()), (int64_t)m,
                                                       CUDA_R_32F, (void*)thrust::raw_pointer_cast(d_V_gesvdr.data()), (int64_t)n,
                                                       CUDA_R_32F,
                                                        (void*)thrust::raw_pointer_cast(bufferOnDevice.data()), workspaceInBytesOnDevice,
                                                        (workspaceInBytesOnHost > 0) ? (void*)bufferOnHost.data() : nullptr, workspaceInBytesOnHost,
                                                        thrust::raw_pointer_cast(d_info_gesvdr.data())
                                                      );
    cudaDeviceSynchronize();
    float elapsed_time_gesvdr = stopTimer();

    float total_time_ms = quant_in_ms + elapsed_time_gesvdr;

    printf("cusolverDnXgesvdr finished in %f ms.\n", elapsed_time_gesvdr);
    
    int info_host = 0;
    cudaMemcpy(&info_host, thrust::raw_pointer_cast(d_info_gesvdr.data()), sizeof(int), cudaMemcpyDeviceToHost);
    if(gesvdr_status != CUSOLVER_STATUS_SUCCESS || info_host != 0) {
        printf("!!! cusolverDnXgesvdr failed! Status: %d, Info: %d !!!\n", gesvdr_status, info_host);
    } else {
        if (g_precision == PrecisionMode::Half16) {
            size_t elems_u = static_cast<size_t>(m) * static_cast<size_t>(target_rank_gesvdr);
            size_t elems_v = static_cast<size_t>(n) * static_cast<size_t>(target_rank_gesvdr);
            size_t elems_s = static_cast<size_t>(target_rank_gesvdr);
            d_U_half.resize(elems_u);
            d_V_half.resize(elems_v);
            d_S_half.resize(elems_s);
            int threads_conv = 256;
            int blocks_u = static_cast<int>((elems_u + threads_conv - 1) / threads_conv);
            int blocks_v = static_cast<int>((elems_v + threads_conv - 1) / threads_conv);
            int blocks_s = static_cast<int>((elems_s + threads_conv - 1) / threads_conv);
            startTimer();
            convert_float_to_half_kernel<<<blocks_u, threads_conv>>>(
                thrust::raw_pointer_cast(d_U_gesvdr.data()),
                thrust::raw_pointer_cast(d_U_half.data()),
                elems_u);
            convert_float_to_half_kernel<<<blocks_v, threads_conv>>>(
                thrust::raw_pointer_cast(d_V_gesvdr.data()),
                thrust::raw_pointer_cast(d_V_half.data()),
                elems_v);
            convert_float_to_half_kernel<<<blocks_s, threads_conv>>>(
                thrust::raw_pointer_cast(d_S_gesvdr.data()),
                thrust::raw_pointer_cast(d_S_half.data()),
                elems_s);
            cudaDeviceSynchronize();
            cudaError_t conv_stat = cudaGetLastError();
            if (conv_stat != cudaSuccess) {
                printf("[Error] float->half output conversion failed: %s\n", cudaGetErrorString(conv_stat));
                return -1;
            }
            quant_out_ms = stopTimer();
        } else if (g_precision == PrecisionMode::BFloat16) {
            size_t elems_u = static_cast<size_t>(m) * static_cast<size_t>(target_rank_gesvdr);
            size_t elems_v = static_cast<size_t>(n) * static_cast<size_t>(target_rank_gesvdr);
            size_t elems_s = static_cast<size_t>(target_rank_gesvdr);
            d_U_bf16.resize(elems_u);
            d_V_bf16.resize(elems_v);
            d_S_bf16.resize(elems_s);
            int threads_conv = 256;
            int blocks_u = static_cast<int>((elems_u + threads_conv - 1) / threads_conv);
            int blocks_v = static_cast<int>((elems_v + threads_conv - 1) / threads_conv);
            int blocks_s = static_cast<int>((elems_s + threads_conv - 1) / threads_conv);
            startTimer();
            convert_float_to_bfloat16_kernel<<<blocks_u, threads_conv>>>(
                thrust::raw_pointer_cast(d_U_gesvdr.data()),
                thrust::raw_pointer_cast(d_U_bf16.data()),
                elems_u);
            convert_float_to_bfloat16_kernel<<<blocks_v, threads_conv>>>(
                thrust::raw_pointer_cast(d_V_gesvdr.data()),
                thrust::raw_pointer_cast(d_V_bf16.data()),
                elems_v);
            convert_float_to_bfloat16_kernel<<<blocks_s, threads_conv>>>(
                thrust::raw_pointer_cast(d_S_gesvdr.data()),
                thrust::raw_pointer_cast(d_S_bf16.data()),
                elems_s);
            cudaDeviceSynchronize();
            cudaError_t conv_stat = cudaGetLastError();
            if (conv_stat != cudaSuccess) {
                printf("[Error] float->bfloat16 output conversion failed: %s\n", cudaGetErrorString(conv_stat));
                return -1;
            }
            quant_out_ms = stopTimer();
        }

        total_time_ms = quant_in_ms + elapsed_time_gesvdr + quant_out_ms;

        if (g_precision != PrecisionMode::Float32) {
            printf("  Quantization (input) time : %f ms\n", quant_in_ms);
            printf("  Re-quantization (output) time : %f ms\n", quant_out_ms);
            printf("  Total (quant + compute + re-quant) : %f ms\n", total_time_ms);
        }

        printf("\n--- Verifying results from cusolverDnXgesvdr ---\n");
        
        // 简单的后向误差验证
        // 1. 验证 U 的正交性
        thrust::device_vector<float> d_UtU(target_rank_gesvdr * target_rank_gesvdr);
        cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                    target_rank_gesvdr, target_rank_gesvdr, m,
                    &one, thrust::raw_pointer_cast(d_U_gesvdr.data()), m,
                    thrust::raw_pointer_cast(d_U_gesvdr.data()), m,
                    &zero, thrust::raw_pointer_cast(d_UtU.data()), target_rank_gesvdr);
        
        // 对角线减1
        std::vector<float> h_UtU(target_rank_gesvdr * target_rank_gesvdr);
        thrust::copy(d_UtU.begin(), d_UtU.end(), h_UtU.begin());
        for(size_t i = 0; i < target_rank_gesvdr; ++i) {
            h_UtU[i * target_rank_gesvdr + i] -= 1.0f;
        }
        thrust::copy(h_UtU.begin(), h_UtU.end(), d_UtU.begin());
        
        float u_ortho_error;
        cublasSnrm2(cublasHandle, target_rank_gesvdr * target_rank_gesvdr,
                    thrust::raw_pointer_cast(d_UtU.data()), 1, &u_ortho_error);
        printf("Orthogonality of U ||U'*U - I||_F = %e\n", u_ortho_error);
        
        // 2. 验证 V 的正交性
        thrust::device_vector<float> d_VtV(target_rank_gesvdr * target_rank_gesvdr);
        cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                    target_rank_gesvdr, target_rank_gesvdr, n,
                    &one, thrust::raw_pointer_cast(d_V_gesvdr.data()), n,
                    thrust::raw_pointer_cast(d_V_gesvdr.data()), n,
                    &zero, thrust::raw_pointer_cast(d_VtV.data()), target_rank_gesvdr);
        
        // 对角线减1
        std::vector<float> h_VtV(target_rank_gesvdr * target_rank_gesvdr);
        thrust::copy(d_VtV.begin(), d_VtV.end(), h_VtV.begin());
        for(size_t i = 0; i < target_rank_gesvdr; ++i) {
            h_VtV[i * target_rank_gesvdr + i] -= 1.0f;
        }
        thrust::copy(h_VtV.begin(), h_VtV.end(), d_VtV.begin());
        
        float v_ortho_error;
        cublasSnrm2(cublasHandle, target_rank_gesvdr * target_rank_gesvdr,
                    thrust::raw_pointer_cast(d_VtV.data()), 1, &v_ortho_error);
        printf("Orthogonality of V ||V'*V - I||_F = %e\n", v_ortho_error);
        
        // 3. 验证重构误差
        // A_approx = U * S * V^T
        thrust::device_vector<float> d_US_gesvdr(m * target_rank_gesvdr);
        dim3 threads_scale_gesvdr(16, 16);
        dim3 blocks_scale_gesvdr((target_rank_gesvdr + 15) / 16, (m + 15) / 16);
        scale_columns_kernel<<<blocks_scale_gesvdr, threads_scale_gesvdr>>>(
            thrust::raw_pointer_cast(d_US_gesvdr.data()),
            thrust::raw_pointer_cast(d_U_gesvdr.data()),
            thrust::raw_pointer_cast(d_S_gesvdr.data()),
            m, target_rank_gesvdr, m, m);
        
        thrust::device_vector<float> d_A_approx(m * n);
        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    m, n, target_rank_gesvdr,
                    &one, thrust::raw_pointer_cast(d_US_gesvdr.data()), m,
                    thrust::raw_pointer_cast(d_V_gesvdr.data()), n,
                    &zero, thrust::raw_pointer_cast(d_A_approx.data()), m);
        
        // 计算重构误差 ||A - USV'||_F
        // 使用备份的原始矩阵A进行计算
        thrust::device_vector<float> d_error_matrix(m * n);
        thrust::copy(d_A_backup.begin(), d_A_backup.end(), d_error_matrix.begin());
        
        // 计算 d_error_matrix = A_backup - A_approx = A - USV'
        cublasSaxpy(cublasHandle, m * n, &minus_one,
                    thrust::raw_pointer_cast(d_A_approx.data()), 1,
                    thrust::raw_pointer_cast(d_error_matrix.data()), 1);
        
        // 计算重构误差的范数
        float recon_error;
        cublasSnrm2(cublasHandle, m * n, thrust::raw_pointer_cast(d_error_matrix.data()), 1, &recon_error);
        
        // 计算原始矩阵A的范数
        float norm_A;
        cublasSnrm2(cublasHandle, m * n, thrust::raw_pointer_cast(d_A_backup.data()), 1, &norm_A);
        
        printf("Reconstruction error ||A - USV'||_F = %e\n", recon_error);
        printf("Relative reconstruction error ||A - USV'||_F / ||A||_F = %e\n", recon_error / norm_A);
        
        // 打印前几个奇异值
        std::vector<float> h_S_gesvdr(target_rank_gesvdr);
        thrust::copy(d_S_gesvdr.begin(), d_S_gesvdr.end(), h_S_gesvdr.begin());
        printf("\nComputed singular values (first 10):\n");
        size_t print_count_computed = (target_rank_gesvdr < 10) ? target_rank_gesvdr : 10;
        for(size_t i = 0; i < print_count_computed; ++i) {
            printf("  s[%zu] = %e\n", i, h_S_gesvdr[i]);
        }
        
        // 与真实奇异值比较
        if (!g_use_custom_matrix) {
            printf("\nComparison with expected singular values:\n");
            size_t compare_count = target_rank_gesvdr;
            if (compare_count > h_S_expected.size()) compare_count = h_S_expected.size();
            if (compare_count > 10) compare_count = 10;
            
            for(size_t i = 0; i < compare_count; ++i) {
                printf("  Expected s[%zu] = %e, Computed s[%zu] = %e, Error = %e\n",
                       i, h_S_expected[i], i, h_S_gesvdr[i],
                       fabsf(h_S_expected[i] - h_S_gesvdr[i]));
            }
        }
    }

    // 清理资源
    cusolverDnDestroyParams(params_gesvdr);
    curandDestroyGenerator(main_gen);
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);

    printf("\n=== cusolverDnXgesvdr Test completed ===\n");
    return (gesvdr_status == CUSOLVER_STATUS_SUCCESS && info_host == 0)
               ? EXIT_SUCCESS : EXIT_FAILURE;
}
