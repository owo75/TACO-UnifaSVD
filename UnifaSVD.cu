#include <iostream> 
#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cusolverDn.h>
#include <type_traits>
#include <cmath>
#include <vector>
#include <string>
#include <iomanip>
#include <algorithm>
#include <cctype>
#include <random>
#include <fstream>
#include <limits>
#include <utility>
#include <cuda_fp16.h>     
#include <cuda_bf16.h>
#include "src/TallShinnyQR.h"
#include "src/HouseholdermixedMGSTSQR.h"
#include "src/my_cuda_kernels.h"
#include "src/cuda_checks.h"
#include "src/QB.h"
#include "src/my_cuda_kernels_extras.cu"
#include "src/matrix_io.h"



long int m, n, k, rank;
float eps;
std::string spectrum_type; // "geometric", "uniform", "cluster0", "cluster1"，"arithmetic", "normal"
float s_max;
float s_min;
std::string precision_str; 
bool debug_mode = false;
bool fast_mode = false; // --fast：速度优先，float 路径尽量保持纯 float 内部计算
static constexpr double kRFinalQrRatioThreshold = 1.0; // n/final_rank 超过该阈值时跳过 Step4 QR


// 基于 EVD 的 SVD 精度等级控制（影响是否触发二次 EVD 等）
enum class AccuracyLevel { Low = 0, Medium = 1, High = 2 };
static AccuracyLevel g_accuracy = AccuracyLevel::Medium; // 默认中等精度

// 秩类型控制（用于路径选择/调度）
enum class RankType { Unknown = 0, Full = 1, Low = 2 };
static RankType g_rank_type = RankType::Unknown;

// 验证相关控制
// Padding 控制
static bool g_padding_enabled = true;
// 面板秩截断控制（调试/分析场景下可强制保留完整面板）
static bool g_force_full_rank = true;
// Unknown-rank / adaptive-stop 控制
static bool g_unknown_rank_mode = false;
static bool g_unknown_rank_stage2 = false;
static bool g_energy_criterion_enabled = false;
static double g_energy_tau = -1.0;
static bool g_tail_criterion_enabled = false;
static double g_tail_norm_eps = -1.0;

// 奇异值/奇异向量计算模式控制，支持 U_Only 与 Vt_Only
enum class ComputeVectors { S_Only = 0, U_Only = 1, Vt_Only = 2, US = 3, SVt = 4, All = 5 };
static ComputeVectors g_compute_vectors = ComputeVectors::All; // 默认：计算 U 和 V

// 是否需要计算任一奇异向量（U 或 V）
static inline bool need_vectors() { return g_compute_vectors != ComputeVectors::S_Only; }
static inline bool want_U()   { return g_compute_vectors == ComputeVectors::U_Only || g_compute_vectors == ComputeVectors::US || g_compute_vectors == ComputeVectors::All; }
static inline bool want_Vt()  { return g_compute_vectors == ComputeVectors::Vt_Only || g_compute_vectors == ComputeVectors::SVt || g_compute_vectors == ComputeVectors::All; }

template <typename T>
static inline const char* precision_name() {
    if constexpr (std::is_same_v<T, __half>) return "half";
    if constexpr (std::is_same_v<T, __nv_bfloat16>) return "bfloat16";
    if constexpr (std::is_same_v<T, float>) return "float";
    if constexpr (std::is_same_v<T, double>) return "double";
    return "unknown";
}

struct ToFloat {
    __host__ __device__ float operator()(const double& v) const { return static_cast<float>(v); }
};

struct EigenToSigma {
    __host__ __device__ double operator()(const double& l) const {
        double v = (l > 0.0) ? l : 0.0;
        return sqrt(v);
    }
};
struct EigenToSigmaF {
    __host__ __device__ float operator()(const float& l) const {
        float v = (l > 0.0f) ? l : 0.0f;
        return sqrtf(v);
    }
};

// 从列主序矩阵中按给定行索引抽取行，输出仍为列主序。
template <typename T>
__global__ void select_rows_kernel(
    T* __restrict__ dst, int ld_dst,
    const T* __restrict__ src, int ld_src,
    const int* __restrict__ row_indices,
    int num_rows, int num_cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < num_rows && col < num_cols) {
        int src_row = row_indices[row];
        dst[(size_t)row + (size_t)col * (size_t)ld_dst] =
            src[(size_t)src_row + (size_t)col * (size_t)ld_src];
    }
}

template <typename T>
__global__ void matrix_norm_sq_strided_kernel(
    const T* __restrict__ src, int ld_src,
    int rows, int cols,
    double* __restrict__ out)
{
    size_t idx = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t total = (size_t)rows * (size_t)cols;
    double local = 0.0;
    for (size_t linear = idx; linear < total; linear += (size_t)blockDim.x * (size_t)gridDim.x) {
        const int col = (int)(linear / (size_t)rows);
        const int row = (int)(linear - (size_t)col * (size_t)rows);
        const double v = static_cast<double>(src[(size_t)row + (size_t)col * (size_t)ld_src]);
        local += v * v;
    }
    if (local != 0.0) atomicAdd(out, local);
}

template <typename T>
__global__ void upper_row_tail_norm_sq_kernel(
    double* __restrict__ dst,
    const T* __restrict__ src, int ld_src,
    int width)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= width) return;
    double sum = 0.0;
    for (int col = row; col < width; ++col) {
        const double v = static_cast<double>(src[(size_t)row + (size_t)col * (size_t)ld_src]);
        sum += v * v;
    }
    dst[row] = sum;
}

template <typename T>
static double device_matrix_norm_sq(cudaStream_t stream,
                                    const T* d_src,
                                    int ld_src,
                                    int rows,
                                    int cols)
{
    if (d_src == nullptr || rows <= 0 || cols <= 0) return 0.0;
    thrust::device_vector<double> d_sum(1);
    CUDA_CHECK(cudaMemsetAsync(thrust::raw_pointer_cast(d_sum.data()), 0, sizeof(double), stream));
    const size_t total = (size_t)rows * (size_t)cols;
    const int threads = 256;
    const int blocks = (int)std::min<size_t>((total + (size_t)threads - 1) / (size_t)threads, (size_t)4096);
    matrix_norm_sq_strided_kernel<T><<<blocks, threads, 0, stream>>>(
        d_src, ld_src, rows, cols, thrust::raw_pointer_cast(d_sum.data()));
    CUDA_CHECK(cudaGetLastError());
    double h_sum = 0.0;
    CUDA_CHECK(cudaMemcpyAsync(&h_sum, thrust::raw_pointer_cast(d_sum.data()), sizeof(double), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return h_sum;
}

template <typename T_compute>
static size_t search_unknown_rank_only(cublasHandle_t cublasHandle,
                                       cusolverDnHandle_t cusolverHandle,
                                       cudaStream_t stream,
                                       thrust::device_vector<T_compute>& d_A_work,
                                       long int m_rows,
                                       long int n_cols,
                                       size_t rank_upper,
                                       long int block_k,
                                       double eps_rel,
                                       bool tail_enabled,
                                       double tail_eps,
                                       bool energy_enabled,
                                       double energy_tau,
                                       double initial_matrix_norm_sq,
                                       bool debug_logs)
{
    const T_compute one = T_compute(1.0);
    const T_compute zero = T_compute(0.0);
    const T_compute minus_one = T_compute(-1.0);
    T_compute* d_A_ptr = thrust::raw_pointer_cast(d_A_work.data());
    const size_t num_blocks = ((size_t)n_cols + (size_t)block_k - 1) / (size_t)block_k;
    const int k_max = (int)std::min<long int>(block_k, n_cols);

    int lwork_geqrf_max = 0;
    SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, (int)m_rows, k_max, d_A_ptr, (int)m_rows, &lwork_geqrf_max);
    int lwork_orgqr_max = 0;
    SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, (int)m_rows, k_max, k_max, d_A_ptr, (int)m_rows, nullptr, &lwork_orgqr_max);
    const int lwork_max_qr = std::max(lwork_geqrf_max, lwork_orgqr_max);

    thrust::device_vector<T_compute> d_work_qr((size_t)lwork_max_qr);
    thrust::device_vector<int> d_devInfo(1);
    thrust::device_vector<T_compute> d_tau((size_t)k_max);
    thrust::device_vector<T_compute> d_R_tsqr((size_t)k_max * (size_t)k_max);
    thrust::device_vector<T_compute> diag_gpu((size_t)k_max);
    thrust::device_vector<double> row_tail_sq_gpu((size_t)k_max);
    thrust::device_vector<T_compute> d_Temp1((size_t)k_max * (size_t)n_cols);

    size_t accumulated_rank = 0;
    size_t final_rank = 0;

    startTimer();
    for (size_t j = 0; j < num_blocks; ++j) {
        long int col_start = (long int)j * block_k;
        long int row_start = (long int)j * block_k;
        long int panel_height = m_rows - row_start;
        long int panel_width = std::min<long int>(block_k, n_cols - col_start);
        panel_width = std::min<long int>(panel_width, (long int)(rank_upper - accumulated_rank));
        if (panel_height <= 0 || panel_width <= 0) break;

        T_compute* d_A_panel_ptr = d_A_ptr + (size_t)col_start * (size_t)m_rows + (size_t)row_start;
        T_compute* d_R_tsqr_ptr = thrust::raw_pointer_cast(d_R_tsqr.data());

        SolverCaller<T_compute>::geqrf(cusolverHandle,
                                       (int)panel_height, (int)panel_width,
                                       d_A_panel_ptr, (int)m_rows,
                                       thrust::raw_pointer_cast(d_tau.data()),
                                       thrust::raw_pointer_cast(d_work_qr.data()), lwork_max_qr,
                                       thrust::raw_pointer_cast(d_devInfo.data()));

        dim3 threads_R_extract(16, 16);
        dim3 blocks_R_extract((int)((panel_width + 15) / 16), (int)((panel_width + 15) / 16));
        extract_upper_triangle_kernel<T_compute><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
            d_R_tsqr_ptr, (int)panel_width, d_A_panel_ptr, (int)m_rows, (int)panel_width, (int)panel_width);
        CUDA_CHECK(cudaGetLastError());

        SolverCaller<T_compute>::orgqr(cusolverHandle,
                                       (int)panel_height, (int)panel_width, (int)panel_width,
                                       d_A_panel_ptr, (int)m_rows,
                                       thrust::raw_pointer_cast(d_tau.data()),
                                       thrust::raw_pointer_cast(d_work_qr.data()), lwork_max_qr,
                                       thrust::raw_pointer_cast(d_devInfo.data()));

        extractDiagonal<<<(int)((panel_width + 255) / 256), 256, 0, stream>>>(
            thrust::raw_pointer_cast(diag_gpu.data()), d_R_tsqr_ptr, (int)panel_width, 0, (int)panel_width);
        CUDA_CHECK(cudaGetLastError());

        std::vector<T_compute> h_diag((size_t)panel_width);
        thrust::copy(diag_gpu.begin(), diag_gpu.begin() + panel_width, h_diag.begin());
        double max_diag = 0.0;
        for (long int i = 0; i < panel_width; ++i) {
            max_diag = std::max(max_diag, std::abs((double)h_diag[(size_t)i]));
        }
        const double relative_eps = max_diag * eps_rel;
        long int diag_kept = panel_width;
        for (long int i = 0; i < panel_width; ++i) {
            if (std::abs((double)h_diag[(size_t)i]) < relative_eps) {
                diag_kept = i;
                break;
            }
        }

        long int tail_kept = panel_width;
        if (tail_enabled && panel_width > 0) {
            upper_row_tail_norm_sq_kernel<T_compute><<<((int)panel_width + 255) / 256, 256, 0, stream>>>(
                thrust::raw_pointer_cast(row_tail_sq_gpu.data()),
                d_R_tsqr_ptr, (int)panel_width, (int)panel_width);
            CUDA_CHECK(cudaGetLastError());
            std::vector<double> h_row_tail_sq((size_t)panel_width);
            thrust::copy(row_tail_sq_gpu.begin(), row_tail_sq_gpu.begin() + panel_width, h_row_tail_sq.begin());
            const double base_tail_sq = h_row_tail_sq.empty() ? 0.0 : h_row_tail_sq[0];
            const double tail_rel_eps = (tail_eps > 0.0) ? tail_eps : std::max(1e-12, eps_rel);
            const double tail_threshold_sq = base_tail_sq * tail_rel_eps * tail_rel_eps;
            for (long int i = 1; i < panel_width; ++i) {
                if (h_row_tail_sq[(size_t)i] <= tail_threshold_sq) {
                    tail_kept = i;
                    break;
                }
            }
        }

        const long int new_panel_width = std::max<long int>(0, std::min<long int>(diag_kept, tail_kept));
        if (debug_logs) {
            printf("[AdaptiveRank][Search] panel %zu kept=%ld (diag=%ld, tail=%ld, panel=%ld).\n",
                   j, new_panel_width, diag_kept, tail_kept, panel_width);
        }
        if (new_panel_width <= 0) {
            final_rank = accumulated_rank;
            printf("[AdaptiveRank][Search] Stop: panel %zu contributes no new columns (accumulated_rank=%zu).\n",
                   j, accumulated_rank);
            break;
        }

        const long int trailing_matrix_cols = n_cols - (col_start + panel_width);
        if (trailing_matrix_cols > 0) {
            T_compute* d_A_trailing_ptr = d_A_ptr + (size_t)(col_start + panel_width) * (size_t)m_rows + (size_t)row_start;
            T_compute* d_Temp1_ptr = thrust::raw_pointer_cast(d_Temp1.data());
            BlasCaller<T_compute>::gemm(cublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)new_panel_width, (int)trailing_matrix_cols, (int)panel_height,
                                        &one,
                                        d_A_panel_ptr, (int)m_rows,
                                        d_A_trailing_ptr, (int)m_rows,
                                        &zero,
                                        d_Temp1_ptr, (int)new_panel_width);
            BlasCaller<T_compute>::gemm(cublasHandle,
                                        CUBLAS_OP_N, CUBLAS_OP_N,
                                        (int)panel_height, (int)trailing_matrix_cols, (int)new_panel_width,
                                        &minus_one,
                                        d_A_panel_ptr, (int)m_rows,
                                        d_Temp1_ptr, (int)new_panel_width,
                                        &one,
                                        d_A_trailing_ptr, (int)m_rows);
        }

        accumulated_rank += (size_t)new_panel_width;
        if (new_panel_width < panel_width) {
            final_rank = accumulated_rank;
            printf("[AdaptiveRank][Search] Stop: local cutoff at panel %zu reduced width from %ld to %ld.\n",
                   j, panel_width, new_panel_width);
            break;
        }

        if (energy_enabled && initial_matrix_norm_sq > 0.0) {
            const long int next_row_start = row_start + panel_width;
            const long int next_col_start = col_start + panel_width;
            const int residual_rows = (next_row_start < m_rows) ? (int)(m_rows - next_row_start) : 0;
            const int residual_cols = (next_col_start < n_cols) ? (int)(n_cols - next_col_start) : 0;
            double residual_norm_sq = 0.0;
            if (residual_rows > 0 && residual_cols > 0) {
                const T_compute* d_residual_ptr = d_A_ptr + (size_t)next_col_start * (size_t)m_rows + (size_t)next_row_start;
                residual_norm_sq = device_matrix_norm_sq<T_compute>(stream, d_residual_ptr, (int)m_rows, residual_rows, residual_cols);
            }
            const double captured_energy = std::max(0.0, 1.0 - residual_norm_sq / initial_matrix_norm_sq);
            if (debug_logs) {
                printf("[AdaptiveRank][Search] panel %zu captured_energy=%.6e residual_ratio=%.6e\n",
                       j, captured_energy, residual_norm_sq / initial_matrix_norm_sq);
            }
            if (captured_energy >= 1.0 - energy_tau) {
                final_rank = accumulated_rank;
                printf("[AdaptiveRank][Search] Stop: captured energy %.6e reached target %.6e at panel %zu.\n",
                       captured_energy, 1.0 - energy_tau, j);
                break;
            }
        }

        if (accumulated_rank >= rank_upper) {
            final_rank = accumulated_rank;
            printf("[AdaptiveRank][Search] Stop: reached rank upper bound. Final rank set to: %zu\n", final_rank);
            break;
        }
    }
    float elapsed_ms = stopTimer();
    if (final_rank == 0) final_rank = accumulated_rank;
    printf("[AdaptiveRank][Search] Search-only QB finished in %f ms. detected_rank=%zu\n", elapsed_ms, final_rank);
    return final_rank;
}

template <typename T>
struct abs_as_double_op {
    __host__ __device__ double operator()(T x) const {
        return fabs((double)x);
    }
};

template <typename T>
struct keep_if_ge_relative_eps_op {
    double relative_eps;
    __host__ __device__ explicit keep_if_ge_relative_eps_op(double eps_in) : relative_eps(eps_in) {}
    __host__ __device__ bool operator()(T x) const {
        return fabs((double)x) >= relative_eps;
    }
};

// 用户自定义矩阵输入控制
static bool g_use_custom_matrix = false;
static std::string g_matrix_file_path;
static MatrixFileFormat g_matrix_file_format = MatrixFileFormat::Text;
static MatrixFileLayout g_matrix_file_layout = MatrixFileLayout::RowMajor;


int parseArguments(int argc,char *argv[])
{
    if(argc < 2)
    {
        printf("Usage (generated matrix): %s <m> <n> <k> <rank|auto> <eps> <spectrum_type> <s_max> <s_min> <precision_str> [--debug] [--fast] [--accuracy=<low|medium|high>] [--vectors=<S|U|Vt|US|SVt|All>] [--unknown-rank] [--energy-tol=<tau>] [--tail-norm-eps=<eps>]\n", argv[0]);
        printf("Usage (matrix file)    : %s <m|auto> <n|auto> <k> <rank|auto> <eps> <precision_str> --matrix-file=<path> [--matrix-format=text|binary|libsvm] [--matrix-layout=row|col] [--debug] [--fast] [--accuracy=<low|medium|high>] [--vectors=<...>] [--unknown-rank] [--energy-tol=<tau>] [--tail-norm-eps=<eps>]\n", argv[0]);
        printf("Usage (matrix file, named): %s --matrix-file=<path> --k=<k> --rank=<rank|auto> --eps=<eps> --precision=<precision> [--m=<m|auto>] [--n=<n|auto>] [--matrix-format=text|binary|libsvm]\n", argv[0]);
        printf("  Shortcut options: --S / --U / --V / --SU / --SV / --All\n");
        printf("  <spectrum_type>: 'geometric', 'uniform', 'cluster0', 'cluster1', 'arithmetic', 'normal'\n");
        printf("  <s_max>: Maximum singular value (e.g., 1.0)\n");
        printf("  <s_min>: Minimum singular value (e.g., 1e-10)\n");
        printf("  <precision_str> : 'half', 'bfloat16', 'float', 'double'\n");
        printf("  --debug (optional): Enable expensive verification checks.\n");
        printf("  --fast             : Prefer pure-float fast path on float runs (skip mixed-precision QB/Step4 lifts when possible).\n");
        printf("  --accuracy=low|medium|high: Control SVD refine checks (default=medium).\n");
        printf("  --force-full-rank         : Keep accumulating full panels even if panel width shrinks.\n");
        printf("  --allow-rank-drop         : Allow adaptive panel rank drop (PPOPP-like truncation behavior).\n");
        printf("  --unknown-rank            : Treat input rank as an upper bound and enable adaptive stop.\n");
        printf("  --energy-tol=<tau>        : Stop when captured energy >= 1 - tau (adaptive mode).\n");
        printf("  --tail-norm-eps=<eps>     : Stop local panel expansion when row-tail norm decays below eps (adaptive mode).\n");
        printf("  --matrix-file=<path>       : Load matrix from file instead of internal generator.\n");
        printf("  --matrix-format=text|binary|libsvm (default=text).\n");
        printf("  --matrix-layout=row|col    (default=row = row-major input).\n");
        printf("  --padding=on|off           Enable/disable odd-dimension padding (default=on).\n");
        printf("  For --matrix-file, spectrum_type/s_max/s_min are omitted.\n");
        printf("  For text/libsvm matrix files, m and n default to auto and are inferred from the file.\n");
        printf("  For binary matrix files, m and n must be provided explicitly.\n");
        return -1;
    }
    std::vector<char*> pos_args; 
    bool force_full_rank_explicit = false;
    std::string named_m, named_n, named_k, named_rank, named_eps, named_precision;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--debug") {
            debug_mode = true; 
        } else if (arg == "--fast") {
            fast_mode = true;
        } else if (arg.rfind("--m=", 0) == 0) {
            named_m = arg.substr(4);
        } else if (arg.rfind("--n=", 0) == 0) {
            named_n = arg.substr(4);
        } else if (arg.rfind("--k=", 0) == 0) {
            named_k = arg.substr(4);
        } else if (arg.rfind("--rank=", 0) == 0) {
            named_rank = arg.substr(7);
        } else if (arg.rfind("--eps=", 0) == 0) {
            named_eps = arg.substr(6);
        } else if (arg.rfind("--precision=", 0) == 0) {
            named_precision = arg.substr(12);
        } else if (arg.rfind("--accuracy=", 0) == 0 || arg.rfind("--acc=", 0) == 0 ||
                   arg.rfind("-accuracy=", 0) == 0  || arg.rfind("-acc=", 0) == 0) {
            std::string val;
            if (arg.rfind("--accuracy=", 0) == 0) val = arg.substr(11);
            else if (arg.rfind("--acc=", 0) == 0) val = arg.substr(6);
            else if (arg.rfind("-accuracy=", 0) == 0) val = arg.substr(10);
            else val = arg.substr(5); // -acc=
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "low" || val == "fast") g_accuracy = AccuracyLevel::Low;
            else if (val == "high" || val == "full") g_accuracy = AccuracyLevel::High;
            else g_accuracy = AccuracyLevel::Medium;
        } else if (arg == "--force-full-rank") {
            g_force_full_rank = true;
            force_full_rank_explicit = true;
        } else if (arg == "--allow-rank-drop") {
            g_force_full_rank = false;
            force_full_rank_explicit = true;
        } else if (arg == "--unknown-rank" || arg == "-unknown-rank") {
            g_unknown_rank_mode = true;
            g_rank_type = RankType::Unknown;
        } else if (arg.rfind("--energy-tol=", 0) == 0 || arg.rfind("-energy-tol=", 0) == 0) {
            g_energy_tau = atof(arg.substr(arg.find('=') + 1).c_str());
            g_energy_criterion_enabled = (g_energy_tau > 0.0);
        } else if (arg.rfind("--tail-norm-eps=", 0) == 0 || arg.rfind("-tail-norm-eps=", 0) == 0) {
            g_tail_norm_eps = atof(arg.substr(arg.find('=') + 1).c_str());
            g_tail_criterion_enabled = (g_tail_norm_eps > 0.0);
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
                printf("[Error] Unknown matrix format '%s'. Use 'text', 'binary', or 'libsvm'.\n", val.c_str());
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
                printf("[Error] Unknown matrix layout '%s'. Use 'row' or 'col'.\n", val.c_str());
                return -1;
            }
        } else if (arg == "--no-padding") {
            g_padding_enabled = false;
        } else if (arg == "--padding" || arg == "-padding") {
            if (i + 1 < argc) {
                std::string val = argv[++i];
                std::transform(val.begin(), val.end(), val.begin(), ::tolower);
                g_padding_enabled = (val != "off" && val != "0" && val != "false");
            } else {
                printf("[Error] --padding requires a value (on/off).\n");
                return -1;
            }
        } else if (arg.rfind("--padding=", 0) == 0) {
            std::string val = arg.substr(10);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            g_padding_enabled = (val != "off" && val != "0" && val != "false");
        } else if (arg == "--S" || arg == "-S") {
            g_compute_vectors = ComputeVectors::S_Only;
        } else if (arg == "--U" || arg == "-U") {
            g_compute_vectors = ComputeVectors::U_Only;
        } else if (arg == "--V" || arg == "-V" || arg == "--Vt" || arg == "-Vt") {
            g_compute_vectors = ComputeVectors::Vt_Only;
        } else if (arg == "--SU" || arg == "-SU" || arg == "--US" || arg == "-US") {
            g_compute_vectors = ComputeVectors::US;
        } else if (arg == "--SV" || arg == "-SV" || arg == "--VS" || arg == "-VS" || arg == "--SVt" || arg == "-SVt") {
            g_compute_vectors = ComputeVectors::SVt;
        } else if (arg == "--All" || arg == "-All" || arg == "--ALL") {
            g_compute_vectors = ComputeVectors::All;
        } else if (arg.rfind("--vectors=", 0) == 0 || arg.rfind("-vectors=", 0) == 0) {
            std::string val = (arg.rfind("--vectors=", 0) == 0) ? arg.substr(10) : arg.substr(9);
            std::transform(val.begin(), val.end(), val.begin(), ::tolower);
            if (val == "s" || val == "s_only" || val == "s-only" || val == "sonly") g_compute_vectors = ComputeVectors::S_Only;
            else if (val == "u" || val == "u_only" || val == "u-only" || val == "uonly") g_compute_vectors = ComputeVectors::U_Only;
            else if (val == "vt" || val == "vt_only" || val == "vt-only" || val == "vtonly" || val == "v^t") g_compute_vectors = ComputeVectors::Vt_Only;
            else if (val == "us") g_compute_vectors = ComputeVectors::US;
            else if (val == "svt" || val == "sv^t") g_compute_vectors = ComputeVectors::SVt;
            else g_compute_vectors = ComputeVectors::All;
        } else {
            pos_args.push_back(argv[i]);
        }
    }

    const bool custom_matrix = g_use_custom_matrix;
    const bool provided_full_params = !custom_matrix;

    if (!custom_matrix && pos_args.size() < 9) {
        printf("Error: Missing required arguments. Found %zu, expected at least 9.\n", pos_args.size());
        return -1;
    }

    auto is_auto_token = [](const std::string& s) {
        std::string v = s;
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        return v == "auto" || v == "unknown";
    };
    auto require_token = [&](const std::string& positional,
                             const std::string& named,
                             const char* label) -> std::string {
        if (!positional.empty()) return positional;
        if (!named.empty()) return named;
        printf("Error: Missing required argument '%s'.\n", label);
        return std::string();
    };
    auto optional_token = [&](const std::string& positional,
                              const std::string& named,
                              const std::string& fallback) -> std::string {
        if (!positional.empty()) return positional;
        if (!named.empty()) return named;
        return fallback;
    };

    // --debug(进行误差验证)/--fast(开启性能较快模式)
    std::string m_arg, n_arg, k_arg, rank_arg, eps_arg, precision_arg;
    if (custom_matrix) {
        const std::string dim_fallback =
            (g_matrix_file_format == MatrixFileFormat::Text || g_matrix_file_format == MatrixFileFormat::LibSVM) ? "auto" : "";
        m_arg = optional_token(pos_args.size() > 0 ? pos_args[0] : "", named_m, dim_fallback);
        n_arg = optional_token(pos_args.size() > 1 ? pos_args[1] : "", named_n, dim_fallback);
        k_arg = require_token(pos_args.size() > 2 ? pos_args[2] : "", named_k, "k");
        rank_arg = require_token(pos_args.size() > 3 ? pos_args[3] : "", named_rank, "rank");
        eps_arg = require_token(pos_args.size() > 4 ? pos_args[4] : "", named_eps, "eps");
        precision_arg = require_token(pos_args.size() > 5 ? pos_args[5] : "", named_precision, "precision");
        if (m_arg.empty() || n_arg.empty() || k_arg.empty() || rank_arg.empty() || eps_arg.empty() || precision_arg.empty()) {
            return -1;
        }
    } else {
        m_arg = pos_args[0];
        n_arg = pos_args[1];
        k_arg = pos_args[2];
        rank_arg = pos_args[3];
        eps_arg = pos_args[4];
        precision_arg = pos_args[8];
    }
    const bool m_auto_arg = is_auto_token(m_arg);
    const bool n_auto_arg = is_auto_token(n_arg);
    m = m_auto_arg ? 0 : atol(m_arg.c_str());
    n = n_auto_arg ? 0 : atol(n_arg.c_str());
    k = atol(k_arg.c_str());
    const bool rank_auto_arg = is_auto_token(rank_arg);
    if (rank_auto_arg) {
        g_unknown_rank_mode = true;
        g_rank_type = RankType::Unknown;
        rank = 0;
    } else {
        rank = atol(rank_arg.c_str());
    }
    eps = atof(eps_arg.c_str());

    if (provided_full_params) {
        spectrum_type = pos_args[5];
        s_max = atof(pos_args[6]);
        s_min = atof(pos_args[7]);
        precision_str = precision_arg;
    } else {
        spectrum_type = "custom";
        s_max = 1.0f;
        s_min = 1.0f;
        precision_str = precision_arg;
    }

    if (g_use_custom_matrix && g_matrix_file_path.empty()) {
        printf("[Error] --matrix-file was specified but no valid path was provided.\n");
        return -1;
    }

    if (g_use_custom_matrix && (m_auto_arg || n_auto_arg)) {
        if (g_matrix_file_format == MatrixFileFormat::Binary) {
            printf("[Error] Automatic dimension inference is only supported for text/libsvm matrix files.\n");
            return -1;
        }
        long int inferred_rows = 0, inferred_cols = 0;
        const bool inferred_ok =
            (g_matrix_file_format == MatrixFileFormat::LibSVM)
                ? infer_libsvm_shape(g_matrix_file_path, inferred_rows, inferred_cols)
                : infer_text_matrix_shape(g_matrix_file_path, g_matrix_file_layout, inferred_rows, inferred_cols);
        if (!inferred_ok) {
            return -1;
        }
        if (!m_auto_arg && m != inferred_rows) {
            printf("[Error] Provided m=%ld does not match inferred matrix rows=%ld from '%s'.\n",
                   m, inferred_rows, g_matrix_file_path.c_str());
            return -1;
        }
        if (!n_auto_arg && n != inferred_cols) {
            printf("[Error] Provided n=%ld does not match inferred matrix cols=%ld from '%s'.\n",
                   n, inferred_cols, g_matrix_file_path.c_str());
            return -1;
        }
        m = inferred_rows;
        n = inferred_cols;
    }

    if (!g_use_custom_matrix && (m_auto_arg || n_auto_arg)) {
        printf("[Error] Automatic dimension inference requires --matrix-file.\n");
        return -1;
    }

    if (rank_auto_arg) {
        rank = std::min(m, n);
    }

    if (g_unknown_rank_mode) {
        if (!force_full_rank_explicit) g_force_full_rank = false;
        if (!g_energy_criterion_enabled) {
            g_energy_tau = std::max(1e-12, (double)eps);
            g_energy_criterion_enabled = true;
        }
        if (!g_tail_criterion_enabled) {
            g_tail_norm_eps = std::max(1e-12, (double)eps);
            g_tail_criterion_enabled = true;
        }
    }

    if (g_rank_type == RankType::Unknown && rank == std::min(m, n)) {
        if (!g_unknown_rank_mode) g_rank_type = RankType::Full;
    }
    printf("Parsed arguments:\n");
    printf("  m = %ld, n = %ld, k = %ld, rank = %ld. eps = %e\n", m, n, k, rank, eps);
    if (provided_full_params) {
        printf("  Spectrum Type: %s\n", spectrum_type.c_str());
        printf("  Singular Value Range: [%e, %e]\n\n", s_max, s_min);
    } else {
        printf("  Spectrum Type: custom (ignored for user-provided matrix)\n");
        printf("  Singular Value Range: [N/A]\n\n");
    }
    const char* acc_str = (g_accuracy == AccuracyLevel::Low ? "low" : (g_accuracy == AccuracyLevel::High ? "high" : "medium"));
    printf("  Precision: %s\n", precision_str.c_str());
    printf("  AccuracyLevel: %s\n", acc_str);
    printf("  FastMode: %s\n", fast_mode ? "on" : "off");
    if (g_use_custom_matrix) {
        const char* fmt =
            (g_matrix_file_format == MatrixFileFormat::Text) ? "text" :
            (g_matrix_file_format == MatrixFileFormat::Binary) ? "binary" : "libsvm";
        const char* layout = (g_matrix_file_layout == MatrixFileLayout::RowMajor) ? "row-major" : "column-major";
        printf("  Matrix Source: user file '%s' (format=%s, layout=%s)\n", g_matrix_file_path.c_str(), fmt, layout);
    } else {
        printf("  Matrix Source: internal low-rank generator\n");
    }
    printf("  Padding: %s\n", g_padding_enabled ? "on" : "off");
    printf("  ForceFullRank: %s\n", g_force_full_rank ? "on" : "off");
    printf("  UnknownRankMode: %s\n", g_unknown_rank_mode ? "on" : "off");
    printf("  EnergyCriterion: %s", g_energy_criterion_enabled ? "on" : "off");
    if (g_energy_criterion_enabled) printf(" (tau=%e)", g_energy_tau);
    printf("\n");
    printf("  TailNormCriterion: %s", g_tail_criterion_enabled ? "on" : "off");
    if (g_tail_criterion_enabled) printf(" (eps=%e)", g_tail_norm_eps);
    printf("\n");
    printf("  eps role: QB panel threshold on |diag(R_panel)|, not a final SVD error target.\n");
    if (g_force_full_rank) {
        printf("  eps truncation: inactive while ForceFullRank=on; panels stay full until target/theoretical rank.\n");
    }
    const char* rstr = (g_rank_type == RankType::Full ? "full" : (g_rank_type == RankType::Low ? "low" : "unknown"));
    const char* vstr =
        (g_compute_vectors == ComputeVectors::S_Only ? "S_Only" :
        (g_compute_vectors == ComputeVectors::U_Only ? "U_Only" :
        (g_compute_vectors == ComputeVectors::Vt_Only ? "Vt_Only" :
        (g_compute_vectors == ComputeVectors::US ? "US" :
        (g_compute_vectors == ComputeVectors::SVt ? "SVt" : "All")))));
    printf("  RankType: %s\n\n", rstr);
    printf("  Vectors: %s\n\n", vstr);
    
    bool has_error = false;
    if (m <= 0 || n <= 0 || k <= 0 || rank <= 0) {
        printf("Error: Matrix dimensions (m, n, k, rank) must be positive.\n");
        has_error = true;
    }
    if (rank > std::min(m, n)) {
        printf("Error: Rank (%ld) cannot be greater than min(m, n) (%ld).\n", rank, std::min(m, n));
        has_error = true;
    }
    if (provided_full_params && !g_use_custom_matrix) {
        if (s_max < s_min) {
            printf("Error: s_max (%e) must be greater than or equal to s_min (%e).\n", s_max, s_min);
            has_error = true;
        }
        if (s_min <= 0) {
            printf("Error: s_min (%e) must be positive.\n", s_min);
            has_error = true;
        }
        if (spectrum_type != "geometric" && spectrum_type != "uniform" &&
            spectrum_type != "cluster0" && spectrum_type != "cluster1" &&
            spectrum_type != "arithmetic" && spectrum_type != "normal") {
            printf("Error: Invalid spectrum_type '%s'.\n", spectrum_type.c_str());
            has_error = true;
        }
    }
    if (precision_str != "half" && precision_str != "bfloat16" &&
        precision_str != "float" && precision_str != "double") {
        printf("Error: Invalid precision '%s'. Supported options are: 'half', 'bfloat16', 'float', 'double'.\n", precision_str.c_str());
        has_error = true;
    }
    if (has_error) {
        printf("\nInput validation failed. Exiting.\n");
        return -1;
    }

    return 0;
}

template<typename T_storage, typename T_compute>
void HQB_SVD(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle, cudaStream_t stream)
{
    // 若维度未对齐到 Tensor Core 友好的 16 倍数，则对行数执行 padding，后续验证再裁剪回原尺寸
    long int m_orig = m;
    constexpr long int kPadAlignment = 16;
    const bool misaligned_m = (m_orig % kPadAlignment) != 0;
    const bool misaligned_n = (n % kPadAlignment) != 0;
    const bool misaligned_rank = (rank % kPadAlignment) != 0;
    const bool want_pad = misaligned_m || misaligned_n || misaligned_rank;
    long int pad_rows = 0;
    if (g_padding_enabled && misaligned_m) {
        pad_rows = (kPadAlignment - (m_orig % kPadAlignment)) % kPadAlignment;
    }
    const bool need_pad = (pad_rows > 0);
    if (need_pad) {
        m += pad_rows;
        printf("[Info] Detected misaligned dimension(s) (m=%ld mod16=%ld, n=%ld mod16=%ld, rank=%ld mod16=%ld). Padding m by %ld -> %ld.\n",
               m_orig, m_orig % kPadAlignment, n, n % kPadAlignment, rank, rank % kPadAlignment, pad_rows, m);
    } else if (want_pad && !g_padding_enabled) {
        printf("[Info] Padding disabled despite misaligned dimension(s) (m=%ld mod16=%ld, n=%ld mod16=%ld, rank=%ld mod16=%ld).\n",
               m, m % kPadAlignment, n, n % kPadAlignment, rank, rank % kPadAlignment);
    } else if (want_pad && g_padding_enabled && !misaligned_m) {
        printf("[Info] Dimensions misaligned (n mod16=%ld, rank mod16=%ld) but m already 16-aligned; skipping row padding.\n",
               n % kPadAlignment, rank % kPadAlignment);
    }
    const long int m_effective = need_pad ? m_orig : m;
    const long int padded_rows = need_pad ? pad_rows : 0;
    const size_t real_rank = rank;
    bool stop_condition_met = false;
    bool panel_rank_drop_warned = false;
    size_t accumulated_rank = 0;
    const size_t mn = std::min(m, n);
    std::cout << "Constant mn: " << mn << std::endl;

    double stage_time_qb_ms = 0.0;
    double stage_time_q_build_ms = 0.0;
    double stage_time_rproj_ms = 0.0;
    double stage_time_step4_ms = 0.0;
    double stage_time_step4_qbuild_ms = 0.0;
    double stage_time_svd_ms = 0.0;
    double stage_time_step5_verify_ms = 0.0;

    // 计算精度常量
    const T_compute one = T_compute(1.0);
    const T_compute zero = T_compute(0.0);
    const T_compute minus_one = T_compute(-1.0);
    const double one_d = 1.0;
    const double zero_d = 0.0;
    const double minus_one_d = -1.0;
    // 将 cuBLAS 绑定到外部传入的 stream，确保与内核/拷贝的同步关系可控
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    // 浮点精度默认开启 TF32 以获得更好的 GEMM 吞吐；可通过 HQB_TF32=0 显式关闭
    if constexpr (std::is_same_v<T_compute, float>) {
        bool use_tf32 = true;
        if (const char* e = std::getenv("HQB_TF32")) { use_tf32 = (e[0] != '0'); }
        cublasMath_t mode = use_tf32
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
            ? CUBLAS_TF32_TENSOR_OP_MATH
#else
            ? CUBLAS_TENSOR_OP_MATH
#endif
            : CUBLAS_DEFAULT_MATH;
        cublasSetMathMode(cublasHandle, mode);
        if (debug_mode) printf("[Info] TF32 %s for float GEMM (debug=%d, env=%s).\n",
                               use_tf32?"enabled":"disabled", (int)debug_mode,
                               (std::getenv("HQB_TF32")? std::getenv("HQB_TF32") : "unset"));
    }

    thrust::device_vector<T_compute> d_A;
    std::vector<T_compute> h_S_expected;

    if (!g_use_custom_matrix) {
        // 1. 创建低秩矩阵
        thrust::device_vector<T_storage> d_A_storage;
        long int ldA_storage = 0;
        create_low_rank_matrix<T_storage, T_compute>(d_A_storage, h_S_expected, m, n, real_rank, spectrum_type,
                                                     static_cast<T_compute>(s_max), static_cast<T_compute>(s_min),
                                                     cublasHandle, cusolverHandle, stream, ldA_storage);

        printf("=== Step 2: Convert Storage Matrix To Compute Layout (ld=m) ===\n");
        d_A.resize((size_t)m * (size_t)n);
        {
            dim3 t_(16,16);
            dim3 b_((n + t_.x - 1) / t_.x, (m + t_.y - 1) / t_.y);
            convert_kernel_strided<T_storage, T_compute><<<b_, t_, 0, stream>>>(
                thrust::raw_pointer_cast(d_A_storage.data()), (size_t)ldA_storage,
                thrust::raw_pointer_cast(d_A.data()), (size_t)m,
                (size_t)m, (size_t)n);
        }
        d_A_storage.clear();
        d_A_storage.shrink_to_fit();
    } else {
        printf("=== Step 1: 从用户文件 '%s' 加载矩阵 ===\n", g_matrix_file_path.c_str());
        if (!load_matrix_from_file<T_compute>(stream,
                                              m_orig,
                                              m,
                                              n,
                                              g_matrix_file_path,
                                              g_matrix_file_format,
                                              g_matrix_file_layout,
                                              d_A)) {
            printf("[Error] Aborting HQB_SVD due to matrix loading failure.\n");
            return;
        }
    }

        // 若做了 padding，将新增的行清零，等效于对 A 进行零填充，不改变原问题
        if (need_pad) {
            dim3 t_zero(32, 8);
            dim3 b_zero((int)((n + t_zero.x - 1) / t_zero.x),
                        (int)((padded_rows + t_zero.y - 1) / t_zero.y));
            zero_rows_kernel<T_compute><<<b_zero, t_zero, 0, stream>>>(
                thrust::raw_pointer_cast(d_A.data()), (int)m, (int)m_effective, (int)m, (int)n);
            CUDA_CHECK(cudaGetLastError());
        }

        // 复制一个新的 d_A_backup，并视情况裁剪出原始尺寸
        thrust::device_vector<T_compute> d_A_backup(d_A);
        thrust::device_vector<T_compute> d_A_backup_trim;
        thrust::device_vector<T_compute>* d_A_effective_ptr = &d_A_backup;
        int ld_A_effective = (int)m;
        if (need_pad) {
            d_A_backup_trim.resize((size_t)m_effective * (size_t)n);
            dim3 t_trim(16, 16);
            dim3 b_trim((int)((n + t_trim.x - 1) / t_trim.x),
                        (int)(((int)m_effective + t_trim.y - 1) / t_trim.y));
            convert_kernel_strided<T_compute, T_compute><<<b_trim, t_trim, 0, stream>>>(
                thrust::raw_pointer_cast(d_A_backup.data()), (size_t)m,
                thrust::raw_pointer_cast(d_A_backup_trim.data()), (size_t)m_effective,
                (size_t)m_effective, (size_t)n);
            CUDA_CHECK(cudaGetLastError());
            d_A_effective_ptr = &d_A_backup_trim;
            ld_A_effective = (int)m_effective;
        }
        const T_compute* d_A_effective_raw = thrust::raw_pointer_cast(d_A_effective_ptr->data());
        const double initial_matrix_norm_sq =
            (g_unknown_rank_mode && g_energy_criterion_enabled)
                ? device_matrix_norm_sq<T_compute>(stream, d_A_effective_raw, ld_A_effective, (int)m_effective, (int)n)
                : 0.0;
        if (g_unknown_rank_mode && g_energy_criterion_enabled) {
            printf("[AdaptiveRank] Initial ||A||_F^2 = %.6e, target captured energy >= %.6e.\n",
                   initial_matrix_norm_sq, 1.0 - g_energy_tau);
        }
        if (g_unknown_rank_mode && !g_unknown_rank_stage2) {
            const long int search_block_k = std::max<long int>(1, std::min<long int>(k, n));
            thrust::device_vector<T_compute> d_A_search(d_A);
            const size_t detected_rank = search_unknown_rank_only<T_compute>(
                cublasHandle,
                cusolverHandle,
                stream,
                d_A_search,
                m,
                n,
                real_rank,
                search_block_k,
                (double)eps,
                g_tail_criterion_enabled,
                g_tail_norm_eps,
                g_energy_criterion_enabled,
                g_energy_tau,
                initial_matrix_norm_sq,
                debug_mode);
            const long int saved_m = m;
            const long int saved_rank = rank;
            const bool saved_unknown = g_unknown_rank_mode;
            const bool saved_force_full_rank = g_force_full_rank;
            const long int rerun_rank = std::max<long int>(1, (long int)detected_rank);
            printf("[AdaptiveRank] Stage1 detected rank=%ld. Re-running full HQB-SVD with fixed rank.\n", rerun_rank);
            m = m_orig;
            rank = rerun_rank;
            g_unknown_rank_mode = false;
            g_force_full_rank = true;
            g_unknown_rank_stage2 = true;
            HQB_SVD<T_storage, T_compute>(cublasHandle, cusolverHandle, stream);
            g_unknown_rank_stage2 = false;
            m = saved_m;
            rank = saved_rank;
            g_unknown_rank_mode = saved_unknown;
            g_force_full_rank = saved_force_full_rank;
            return;
        }
        // 获取 d_A 的原始指针
        T_compute* d_A_ptr = thrust::raw_pointer_cast(d_A.data());
        size_t final_rank = mn;

        // PPOPP-aligned float-compute branch: default for float/half/bfloat16 storage
        // when compute type is float, unless the user explicitly requests --fast.
        constexpr bool ppopp_eligible_storage =
            std::is_same_v<T_storage, float> ||
            std::is_same_v<T_storage, __half> ||
            std::is_same_v<T_storage, __nv_bfloat16>;
        if constexpr (ppopp_eligible_storage && std::is_same_v<T_compute, float>) {
            const bool enable_ppopp_like_fast_path = !fast_mode;
            if (enable_ppopp_like_fast_path) {
                const char* ppopp_path_label =
                    std::is_same_v<T_storage, float> ? "float default" :
                    (std::is_same_v<T_storage, __half> ? "half default" : "bfloat16 default");
                printf("=== Step 3: Run QB Algorithm (%s -> PPOPP-aligned path) ===\n", ppopp_path_label);
                cublasMath_t prev_math_mode_float_branch = CUBLAS_DEFAULT_MATH;
                cublasGetMathMode(cublasHandle, &prev_math_mode_float_branch);
                if constexpr (std::is_same_v<T_storage, float> && std::is_same_v<T_compute, float>) {
                    // Match Right.cu float path: force DEFAULT math mode (no TF32).
                    cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
                    if (debug_mode) {
                        printf("[Info] PPOPP float path math mode: DEFAULT (Right.cu alignment).\n");
                    }
                } else {
                    bool ppopp_tf32 = true;
                    if (const char* e = std::getenv("HQB_PPOPP_TF32")) { ppopp_tf32 = (e[0] != '0'); }
                    cublasMath_t ppopp_mode = CUBLAS_DEFAULT_MATH;
                    if (ppopp_tf32) {
#ifdef CUBLAS_TF32_TENSOR_OP_MATH
                        ppopp_mode = CUBLAS_TF32_TENSOR_OP_MATH;
#else
                        ppopp_mode = CUBLAS_TENSOR_OP_MATH;
#endif
                    }
                    cublasSetMathMode(cublasHandle, ppopp_mode);
                    if (debug_mode) {
                        printf("[Info] PPOPP path math mode: %s (env HQB_PPOPP_TF32=%s)\n",
                               ppopp_tf32 ? "TF32/TensorOp" : "DEFAULT",
                               std::getenv("HQB_PPOPP_TF32") ? std::getenv("HQB_PPOPP_TF32") : "unset");
                    }
                }
                // Warm up double GEMM once to avoid lazy-load penalties inside timed SVD tail.
                static bool s_ppopp_dgemm_warm = false;
                if (!s_ppopp_dgemm_warm) {
                    thrust::device_vector<double> d_warm_a(32 * 32);
                    thrust::device_vector<double> d_warm_b(32 * 32);
                    thrust::device_vector<double> d_warm_c(32 * 32);
                    double* d_warm_a_ptr = thrust::raw_pointer_cast(d_warm_a.data());
                    double* d_warm_b_ptr = thrust::raw_pointer_cast(d_warm_b.data());
                    double* d_warm_c_ptr = thrust::raw_pointer_cast(d_warm_c.data());
                    cudaMemsetAsync(d_warm_a_ptr, 0, d_warm_a.size() * sizeof(double), stream);
                    cudaMemsetAsync(d_warm_b_ptr, 0, d_warm_b.size() * sizeof(double), stream);
                    cudaMemsetAsync(d_warm_c_ptr, 0, d_warm_c.size() * sizeof(double), stream);
                    double one_d = 1.0;
                    double zero_d = 0.0;
                    cublasDgemm(cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                32, 32, 32,
                                &one_d,
                                d_warm_a_ptr, 32,
                                d_warm_b_ptr, 32,
                                &zero_d,
                                d_warm_c_ptr, 32);
                    cudaStreamSynchronize(stream);
                    s_ppopp_dgemm_warm = true;
                    if (debug_mode) {
                        printf("[Info] PPOPP path: warmed up double GEMM.\n");
                    }
                }
                thrust::device_vector<double> d_A_backup_fast((size_t)m_effective * (size_t)n);
                double* d_A_backup_fast_ptr = thrust::raw_pointer_cast(d_A_backup_fast.data());
                convert_kernel<float, double><<<(int)((((size_t)m_effective * (size_t)n) + 255) / 256), 256, 0, stream>>>(
                    d_A_effective_raw, d_A_backup_fast_ptr, (size_t)m_effective * (size_t)n);

                const long int fast_block_k = k;
                const size_t num_blocks_fast = (n + fast_block_k - 1) / fast_block_k;
                size_t accumulated_rank_fast = 0;
                bool stop_condition_met_fast = false;
                final_rank = mn;

                thrust::device_vector<T_compute> d_R_final((size_t)n * (size_t)n);
                T_compute* d_R_final_ptr = thrust::raw_pointer_cast(d_R_final.data());
                cudaMemset(d_R_final_ptr, 0, (size_t)n * (size_t)n * sizeof(T_compute));

                thrust::device_vector<T_compute> d_Q_final((size_t)m * (size_t)real_rank);
                T_compute* d_Q_final_ptr = thrust::raw_pointer_cast(d_Q_final.data());
                cudaMemset(d_Q_final_ptr, 0, (size_t)m * (size_t)real_rank * sizeof(T_compute));

                thrust::device_vector<T_compute> d_W_final((size_t)m * (size_t)real_rank);
                T_compute* d_W_final_ptr = thrust::raw_pointer_cast(d_W_final.data());
                cudaMemset(d_W_final_ptr, 0, (size_t)m * (size_t)real_rank * sizeof(T_compute));

                thrust::device_vector<T_compute> d_Y_final((size_t)m * (size_t)real_rank);
                T_compute* d_Y_final_ptr = thrust::raw_pointer_cast(d_Y_final.data());
                cudaMemset(d_Y_final_ptr, 0, (size_t)m * (size_t)real_rank * sizeof(T_compute));

                startTimer();
                for (size_t j = 0; j < num_blocks_fast; ++j) {
                    long int col_start = (long int)j * fast_block_k;
                    long int row_start = (long int)j * fast_block_k;
                    long int panel_height = m - row_start;
                    long int panel_width = std::min(fast_block_k, n - col_start);
                    panel_width = std::min(panel_width, (long int)(real_rank - accumulated_rank_fast));
                    if (panel_height <= 0 || panel_width <= 0) break;

                    thrust::device_vector<T_compute> d_R_tsqr((size_t)panel_width * (size_t)panel_width);
                    T_compute* d_R_tsqr_ptr = thrust::raw_pointer_cast(d_R_tsqr.data());
                    T_compute* d_panel_ptr = d_A_ptr + (size_t)col_start * (size_t)m + (size_t)row_start;
                    int lwork_geqrf = 0;
                    int lwork_orgqr = 0;
                    thrust::device_vector<T_compute> d_tau((size_t)panel_width);
                    thrust::device_vector<int> d_devInfo_qr(1);

                    SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle,
                                                              (int)panel_height, (int)panel_width,
                                                              d_panel_ptr, (int)m, &lwork_geqrf);
                    thrust::device_vector<T_compute> d_work_geqrf((size_t)lwork_geqrf);
                    SolverCaller<T_compute>::geqrf(cusolverHandle,
                                                   (int)panel_height, (int)panel_width,
                                                   d_panel_ptr, (int)m,
                                                   thrust::raw_pointer_cast(d_tau.data()),
                                                   thrust::raw_pointer_cast(d_work_geqrf.data()),
                                                   lwork_geqrf,
                                                   thrust::raw_pointer_cast(d_devInfo_qr.data()));

                    dim3 threads_R_extract(16, 16);
                    dim3 blocks_R_extract((int)((panel_width + 15) / 16), (int)((panel_width + 15) / 16));
                    extract_upper_triangle_kernel<T_compute><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
                        d_R_tsqr_ptr, (int)panel_width, d_panel_ptr, (int)m, (int)panel_width, (int)panel_width);

                    SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle,
                                                              (int)panel_height, (int)panel_width, (int)panel_width,
                                                              d_panel_ptr, (int)m,
                                                              thrust::raw_pointer_cast(d_tau.data()),
                                                              &lwork_orgqr);
                    thrust::device_vector<T_compute> d_work_orgqr((size_t)lwork_orgqr);
                    SolverCaller<T_compute>::orgqr(cusolverHandle,
                                                   (int)panel_height, (int)panel_width, (int)panel_width,
                                                   d_panel_ptr, (int)m,
                                                   thrust::raw_pointer_cast(d_tau.data()),
                                                   thrust::raw_pointer_cast(d_work_orgqr.data()),
                                                   lwork_orgqr,
                                                   thrust::raw_pointer_cast(d_devInfo_qr.data()));

                    thrust::device_vector<T_compute> diag_elements_gpu((size_t)panel_width);
                    extractDiagonal<<<(int)((panel_width + 255) / 256), 256, 0, stream>>>(
                        thrust::raw_pointer_cast(diag_elements_gpu.data()), d_R_tsqr_ptr, (int)panel_width, 0, (int)panel_width);
                    CUDA_CHECK(cudaGetLastError());

                    std::vector<T_compute> diag_elements_cpu((size_t)panel_width);
                    thrust::copy(diag_elements_gpu.begin(), diag_elements_gpu.end(), diag_elements_cpu.begin());

                    std::vector<int> h_keep_indices;
                    h_keep_indices.reserve((size_t)panel_width);
                    double max_diag = 0.0;
                    for (int i = 0; i < panel_width; ++i) {
                        double abs_val = std::abs((double)diag_elements_cpu[(size_t)i]);
                        if (abs_val > max_diag) max_diag = abs_val;
                    }
                    double relative_eps = max_diag * (double)eps;
                    for (int i = 0; i < panel_width; ++i) {
                        if (std::abs((double)diag_elements_cpu[(size_t)i]) >= relative_eps) {
                            h_keep_indices.push_back(i);
                        }
                    }

                    long int desired_panel_width = std::min<long int>(panel_width, (long int)(real_rank - accumulated_rank_fast));
                    long int diag_kept = (long int)h_keep_indices.size();
                    long int tail_kept = panel_width;
                    if (g_unknown_rank_mode) {
                        diag_kept = panel_width;
                        h_keep_indices.clear();
                        for (int i = 0; i < panel_width; ++i) {
                            if (std::abs((double)diag_elements_cpu[(size_t)i]) < relative_eps) {
                                diag_kept = i;
                                break;
                            }
                        }
                        if (g_tail_criterion_enabled && panel_width > 0) {
                            thrust::device_vector<double> d_row_tail_sq((size_t)panel_width);
                            upper_row_tail_norm_sq_kernel<T_compute><<<((int)panel_width + 255) / 256, 256, 0, stream>>>(
                                thrust::raw_pointer_cast(d_row_tail_sq.data()),
                                d_R_tsqr_ptr, (int)panel_width, (int)panel_width);
                            CUDA_CHECK(cudaGetLastError());
                            std::vector<double> h_row_tail_sq((size_t)panel_width);
                            thrust::copy(d_row_tail_sq.begin(), d_row_tail_sq.end(), h_row_tail_sq.begin());
                            const double base_tail_sq = h_row_tail_sq.empty() ? 0.0 : h_row_tail_sq[0];
                            const double tail_rel_eps = (g_tail_norm_eps > 0.0) ? g_tail_norm_eps : std::max(1e-12, (double)eps);
                            const double tail_threshold_sq = base_tail_sq * tail_rel_eps * tail_rel_eps;
                            for (int i = 1; i < panel_width; ++i) {
                                if (h_row_tail_sq[(size_t)i] <= tail_threshold_sq) {
                                    tail_kept = i;
                                    break;
                                }
                            }
                        }
                        const long int prefix_kept = std::max<long int>(0, std::min<long int>(diag_kept, tail_kept));
                        h_keep_indices.resize((size_t)prefix_kept);
                        for (long int i = 0; i < prefix_kept; ++i) h_keep_indices[(size_t)i] = (int)i;
                    }
                    long int new_panel_width = (long int)h_keep_indices.size();
                    if (g_force_full_rank && desired_panel_width > 0) {
                        h_keep_indices.resize((size_t)desired_panel_width);
                        for (int i = 0; i < desired_panel_width; ++i) h_keep_indices[(size_t)i] = i;
                        new_panel_width = desired_panel_width;
                    } else if (g_unknown_rank_mode && debug_mode) {
                        printf("[AdaptiveRank] panel %zu kept=%ld (diag=%ld, tail=%ld, panel=%ld).\n",
                               j, new_panel_width, diag_kept, tail_kept, panel_width);
                    }
                    if (new_panel_width <= 0) {
                        if (g_unknown_rank_mode) {
                            stop_condition_met_fast = true;
                            final_rank = accumulated_rank_fast;
                            printf("[AdaptiveRank] Stop: panel %zu contributes no new columns (accumulated_rank=%zu).\n",
                                   j, accumulated_rank_fast);
                            break;
                        }
                        continue;
                    }

                    thrust::device_vector<int> d_keep_indices = h_keep_indices;
                    thrust::device_vector<T_compute> d_Q_filtered((size_t)panel_height * (size_t)new_panel_width);
                    T_compute* d_Q_filtered_ptr = thrust::raw_pointer_cast(d_Q_filtered.data());
                    dim3 threads_gather(16, 16);
                    dim3 blocks_gather((int)((new_panel_width + 15) / 16), (int)((panel_height + 15) / 16));
                    select_columns_kernel<T_compute><<<blocks_gather, threads_gather, 0, stream>>>(
                        d_Q_filtered_ptr, (int)panel_height,
                        d_A_ptr + (size_t)col_start * (size_t)m + (size_t)row_start, (int)m,
                        thrust::raw_pointer_cast(d_keep_indices.data()),
                        (int)panel_height, (int)new_panel_width);

                    T_compute* d_Q_final_dest_ptr = d_Q_final_ptr + (size_t)accumulated_rank_fast * (size_t)m + (size_t)row_start;
                    cudaMemcpy2DAsync(d_Q_final_dest_ptr,
                                      (size_t)m * sizeof(T_compute),
                                      d_Q_filtered_ptr,
                                      (size_t)panel_height * sizeof(T_compute),
                                      (size_t)panel_height * sizeof(T_compute),
                                      (size_t)new_panel_width,
                                      cudaMemcpyDeviceToDevice,
                                      stream);

                    thrust::device_vector<T_compute> d_R_temp((size_t)panel_width * (size_t)new_panel_width);
                    T_compute* d_R_temp_ptr = thrust::raw_pointer_cast(d_R_temp.data());
                    dim3 threads_gather_R(16, 16);
                    dim3 blocks_gather_R((int)((new_panel_width + 15) / 16), (int)((panel_width + 15) / 16));
                    select_columns_kernel<T_compute><<<blocks_gather_R, threads_gather_R, 0, stream>>>(
                        d_R_temp_ptr, (int)panel_width,
                        d_R_tsqr_ptr, (int)panel_width,
                        thrust::raw_pointer_cast(d_keep_indices.data()),
                        (int)panel_width, (int)new_panel_width);

                    thrust::device_vector<T_compute> d_R_filtered((size_t)new_panel_width * (size_t)new_panel_width);
                    T_compute* d_R_filtered_ptr = thrust::raw_pointer_cast(d_R_filtered.data());
                    dim3 threads_gather_R_rows(16, 16);
                    dim3 blocks_gather_R_rows((int)((new_panel_width + 15) / 16), (int)((new_panel_width + 15) / 16));
                    select_rows_kernel<T_compute><<<blocks_gather_R_rows, threads_gather_R_rows, 0, stream>>>(
                        d_R_filtered_ptr, (int)new_panel_width,
                        d_R_temp_ptr, (int)panel_width,
                        thrust::raw_pointer_cast(d_keep_indices.data()),
                        (int)new_panel_width, (int)new_panel_width);

                    T_compute* d_R_dest_ptr = d_R_final_ptr + (size_t)accumulated_rank_fast * (size_t)n + (size_t)accumulated_rank_fast;
                    cudaMemcpy2DAsync(d_R_dest_ptr,
                                      (size_t)n * sizeof(T_compute),
                                      d_R_filtered_ptr,
                                      (size_t)new_panel_width * sizeof(T_compute),
                                      (size_t)new_panel_width * sizeof(T_compute),
                                      (size_t)new_panel_width,
                                      cudaMemcpyDeviceToDevice,
                                      stream);

                    thrust::device_vector<T_compute> d_B((size_t)panel_height * (size_t)new_panel_width);
                    T_compute* d_B_ptr = thrust::raw_pointer_cast(d_B.data());
                    BlasCaller<T_compute>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                (int)panel_height, (int)new_panel_width,
                                                &minus_one, d_Q_filtered_ptr, (int)panel_height,
                                                &zero, d_Q_filtered_ptr, (int)panel_height,
                                                d_B_ptr, (int)panel_height);
                    dim3 threads_id(256);
                    dim3 blocks_id((int)((std::min(panel_height, new_panel_width) + threads_id.x - 1) / threads_id.x));
                    add_identity_diagonal_kernel<T_compute><<<blocks_id, threads_id, 0, stream>>>(
                        d_B_ptr, (int)panel_height, (int)new_panel_width, (int)panel_height);

                    thrust::device_vector<T_compute> d_W(d_B.size());
                    T_compute* d_W_ptr = thrust::raw_pointer_cast(d_W.data());
                    cudaMemcpyAsync(d_W_ptr, d_B_ptr, d_B.size() * sizeof(T_compute), cudaMemcpyDeviceToDevice, stream);

                    thrust::device_vector<T_compute> d_Y((size_t)panel_height * (size_t)new_panel_width);
                    T_compute* d_Y_ptr = thrust::raw_pointer_cast(d_Y.data());
                    int lwork_getrf = 0;
                    SolverCaller<T_compute>::getrf_bufferSize(cusolverHandle, (int)panel_height, (int)new_panel_width,
                                                              d_B_ptr, (int)panel_height, &lwork_getrf);
                    thrust::device_vector<T_compute> d_getrf_work((size_t)lwork_getrf);
                    thrust::device_vector<int> devInfo_getrf(1);
                    SolverCaller<T_compute>::getrf(cusolverHandle,
                                                   (int)panel_height, (int)new_panel_width,
                                                   d_B_ptr, (int)panel_height,
                                                   thrust::raw_pointer_cast(d_getrf_work.data()),
                                                   NULL,
                                                   thrust::raw_pointer_cast(devInfo_getrf.data()));

                    cudaMemsetAsync(d_Y_ptr, 0, (size_t)panel_height * (size_t)new_panel_width * sizeof(T_compute), stream);
                    dim3 threads_for_L(16, 16);
                    dim3 blocks_for_L((int)((new_panel_width + threads_for_L.x - 1) / threads_for_L.x),
                                      (int)((panel_height + threads_for_L.y - 1) / threads_for_L.y));
                    extract_L_factor_kernel<T_compute><<<blocks_for_L, threads_for_L, 0, stream>>>(
                        d_Y_ptr, (int)panel_height, d_B_ptr, (int)panel_height, (int)panel_height, (int)new_panel_width);

                    BlasCaller<T_compute>::trsm(cublasHandle,
                                                CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                                CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                                (int)panel_height, (int)new_panel_width,
                                                &one, d_Y_ptr, (int)panel_height, d_W_ptr, (int)panel_height);

                    T_compute* d_W_final_dest_ptr = d_W_final_ptr + (size_t)accumulated_rank_fast * (size_t)m;
                    cudaMemcpy2DAsync(d_W_final_dest_ptr + row_start,
                                      (size_t)m * sizeof(T_compute),
                                      d_W_ptr,
                                      (size_t)panel_height * sizeof(T_compute),
                                      (size_t)panel_height * sizeof(T_compute),
                                      (size_t)new_panel_width,
                                      cudaMemcpyDeviceToDevice,
                                      stream);

                    T_compute* d_Y_final_dest_ptr = d_Y_final_ptr + (size_t)accumulated_rank_fast * (size_t)m;
                    cudaMemcpy2DAsync(d_Y_final_dest_ptr + row_start,
                                      (size_t)m * sizeof(T_compute),
                                      d_Y_ptr,
                                      (size_t)panel_height * sizeof(T_compute),
                                      (size_t)panel_height * sizeof(T_compute),
                                      (size_t)new_panel_width,
                                      cudaMemcpyDeviceToDevice,
                                      stream);

                    long int trailing_matrix_cols = n - (col_start + panel_width);
                    if (trailing_matrix_cols > 0) {
                        T_compute* d_A_trailing_ptr = d_A_ptr + (size_t)(col_start + panel_width) * (size_t)m + (size_t)row_start;
                        thrust::device_vector<T_compute> d_Temp1((size_t)new_panel_width * (size_t)trailing_matrix_cols);
                        T_compute* d_Temp1_ptr = thrust::raw_pointer_cast(d_Temp1.data());

                        BlasCaller<T_compute>::gemm(cublasHandle,
                                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                                    (int)new_panel_width, (int)trailing_matrix_cols, (int)panel_height,
                                                    &one,
                                                    d_W_ptr, (int)panel_height,
                                                    d_A_trailing_ptr, (int)m,
                                                    &zero,
                                                    d_Temp1_ptr, (int)new_panel_width);

                        BlasCaller<T_compute>::gemm(cublasHandle,
                                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)panel_height, (int)trailing_matrix_cols, (int)new_panel_width,
                                                    &minus_one,
                                                    d_Y_ptr, (int)panel_height,
                                                    d_Temp1_ptr, (int)new_panel_width,
                                                    &one,
                                                    d_A_trailing_ptr, (int)m);

                        T_compute* d_R_offdiag_dest_ptr = d_R_final_ptr + (size_t)(col_start + panel_width) * (size_t)n + (size_t)accumulated_rank_fast;
                        cudaMemcpy2DAsync(d_R_offdiag_dest_ptr,
                                          (size_t)n * sizeof(T_compute),
                                          d_A_trailing_ptr,
                                          (size_t)m * sizeof(T_compute),
                                          (size_t)new_panel_width * sizeof(T_compute),
                                          (size_t)trailing_matrix_cols,
                                          cudaMemcpyDeviceToDevice,
                                          stream);
                    }

                    accumulated_rank_fast += (size_t)new_panel_width;
                    if (!g_force_full_rank && new_panel_width < panel_width) {
                        stop_condition_met_fast = true;
                        final_rank = accumulated_rank_fast;
                        if (g_unknown_rank_mode) {
                            printf("[AdaptiveRank] Stop: local cutoff at panel %zu reduced width from %ld to %ld.\n",
                                   j, panel_width, new_panel_width);
                        }
                    }
                    if (g_unknown_rank_mode && g_energy_criterion_enabled && initial_matrix_norm_sq > 0.0) {
                        const long int next_row_start = row_start + panel_width;
                        const long int next_col_start = col_start + panel_width;
                        const int residual_rows = (next_row_start < m) ? (int)(m - next_row_start) : 0;
                        const int residual_cols = (next_col_start < n) ? (int)(n - next_col_start) : 0;
                        double residual_norm_sq = 0.0;
                        if (residual_rows > 0 && residual_cols > 0) {
                            const T_compute* d_residual_ptr = d_A_ptr + (size_t)next_col_start * (size_t)m + (size_t)next_row_start;
                            residual_norm_sq = device_matrix_norm_sq<T_compute>(stream, d_residual_ptr, (int)m, residual_rows, residual_cols);
                        }
                        const double captured_energy = std::max(0.0, 1.0 - residual_norm_sq / initial_matrix_norm_sq);
                        if (debug_mode) {
                            printf("[AdaptiveRank] panel %zu captured_energy=%.6e residual_ratio=%.6e\n",
                                   j, captured_energy, residual_norm_sq / initial_matrix_norm_sq);
                        }
                        if (captured_energy >= 1.0 - g_energy_tau) {
                            stop_condition_met_fast = true;
                            final_rank = accumulated_rank_fast;
                            printf("[AdaptiveRank] Stop: captured energy %.6e reached target %.6e at panel %zu.\n",
                                   captured_energy, 1.0 - g_energy_tau, j);
                        }
                    }
                    if (accumulated_rank_fast >= (size_t)real_rank) {
                        stop_condition_met_fast = true;
                        final_rank = accumulated_rank_fast;
                        printf("\n         !!! Stopping: Reached or exceeded theoretical rank. Final rank set to: %zu\n", final_rank);
                    }
                    if (stop_condition_met_fast) break;
                }

                float elapsed_time_QB = stopTimer();
                printf("QB finished successfully in %f ms.\n", elapsed_time_QB);

                startTimer();
                thrust::device_vector<double> d_Q_basis((size_t)m * (size_t)final_rank);
                double* d_Q_basis_ptr = thrust::raw_pointer_cast(d_Q_basis.data());
                const bool fast_single_panel_q = (final_rank <= (size_t)fast_block_k);
                if (fast_single_panel_q) {
                    convert_kernel<float, double><<<(int)((((size_t)m * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                        d_Q_final_ptr, d_Q_basis_ptr, (size_t)m * (size_t)final_rank);
                } else {
                    thrust::device_vector<double> d_W_final_double((size_t)m * (size_t)final_rank);
                    thrust::device_vector<double> d_Y_final_double((size_t)m * (size_t)final_rank);
                    double* d_W_final_double_ptr = thrust::raw_pointer_cast(d_W_final_double.data());
                    double* d_Y_final_double_ptr = thrust::raw_pointer_cast(d_Y_final_double.data());
                    convert_kernel<float, double><<<(int)((((size_t)m * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                        d_W_final_ptr, d_W_final_double_ptr, (size_t)m * (size_t)final_rank);
                    convert_kernel<float, double><<<(int)((((size_t)m * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                        d_Y_final_ptr, d_Y_final_double_ptr, (size_t)m * (size_t)final_rank);

                    thrust::device_vector<double> d_recursive_work_W((size_t)final_rank * (size_t)fast_block_k);
                    double* d_recursive_work_W_ptr = thrust::raw_pointer_cast(d_recursive_work_W.data());
                    recursive_compute_w<double>(cublasHandle,
                                                (size_t)m,
                                                (size_t)final_rank,
                                                (size_t)fast_block_k,
                                                d_W_final_double_ptr, (size_t)m,
                                                d_Y_final_double_ptr, (size_t)m,
                                                d_recursive_work_W_ptr);

                    cudaMemsetAsync(d_Q_basis_ptr, 0, sizeof(double) * (size_t)m * (size_t)final_rank, stream);
                    BlasCaller<double>::gemm(cublasHandle,
                                             CUBLAS_OP_N, CUBLAS_OP_T,
                                             (int)m, (int)final_rank, (int)final_rank,
                                             &minus_one_d,
                                             d_W_final_double_ptr, (int)m,
                                             d_Y_final_double_ptr, (int)m,
                                             &zero_d,
                                             d_Q_basis_ptr, (int)m);
                    dim3 threads_diag(256);
                    dim3 blocks_diag((int)((final_rank + 255) / 256));
                    add_identity_diagonal_kernel<double><<<blocks_diag, threads_diag, 0, stream>>>(
                        d_Q_basis_ptr, (int)m, (int)final_rank, (int)m, 1.0);
                }
                float elapsed_time_Q = stopTimer();
                printf("Q generation finished successfully in %f ms.\n", elapsed_time_Q);

                startTimer();
                thrust::device_vector<double> d_Q_basis_effective;
                const double* d_Q_basis_effective_ptr = d_Q_basis_ptr;
                int ld_Q_basis_effective = (int)m;
                if (need_pad) {
                    d_Q_basis_effective.resize((size_t)m_effective * (size_t)final_rank);
                    dim3 tQtrim(16, 16);
                    dim3 bQtrim((int)((final_rank + tQtrim.x - 1) / tQtrim.x),
                                (int)(((int)m_effective + tQtrim.y - 1) / tQtrim.y));
                    convert_kernel_strided<double, double><<<bQtrim, tQtrim, 0, stream>>>(
                        d_Q_basis_ptr, (size_t)m,
                        thrust::raw_pointer_cast(d_Q_basis_effective.data()), (size_t)m_effective,
                        (size_t)m_effective, (size_t)final_rank);
                    d_Q_basis_effective_ptr = thrust::raw_pointer_cast(d_Q_basis_effective.data());
                    ld_Q_basis_effective = (int)m_effective;
                }

                const double ratio_n_over_rank = (final_rank > 0)
                    ? static_cast<double>(n) / static_cast<double>(final_rank)
                    : std::numeric_limits<double>::infinity();
                constexpr bool ppopp_full_match =
                    std::is_same_v<T_storage, float> && std::is_same_v<T_compute, float>;
                bool ppopp_float_evd = false;
                if (const char* e = std::getenv("HQB_PPOPP_FLOAT_EVD")) {
                    ppopp_float_evd = (e[0] != '0');
                }
                bool ppopp_evd_breakdown = false;
                if (const char* e = std::getenv("HQB_PPOPP_EVD_BREAKDOWN")) {
                    ppopp_evd_breakdown = (e[0] != '0');
                }
                bool use_double_rproj = (g_accuracy == AccuracyLevel::High);
                if (const char* e = std::getenv("HQB_RPROJ_DOUBLE")) {
                    use_double_rproj = (e[0] != '0');
                }
                bool skip_final_qr =
                    (ratio_n_over_rank > kRFinalQrRatioThreshold);
                if constexpr (ppopp_full_match) {
                    use_double_rproj = false;
                    skip_final_qr = false;
                }
                printf("[Route] solver=HQB path=PPOPP-aligned storage=%s compute=%s final_rank=%zu step4=%s rproj=%s ratio_n_over_rank=%.3f threshold=%.3f\n",
                       precision_name<T_storage>(),
                       precision_name<T_compute>(),
                       final_rank,
                       skip_final_qr ? "bypass" : "qr",
                       ppopp_full_match ? "R_final" : (use_double_rproj ? "double" : "float_tf32"),
                       ratio_n_over_rank,
                       kRFinalQrRatioThreshold);

                float elapsed_time_rproj = 0.0f;
                float elapsed_time_r_factor = 0.0f;
                float elapsed_time_qbuild_step4 = 0.0f;
                thrust::device_vector<double> d_R_final_T;
                thrust::device_vector<double> d_R_final_T_verify;
                double* d_R_final_T_ptr = nullptr;

                thrust::device_vector<double> d_S_final((size_t)final_rank);
                double* d_S_final_ptr = thrust::raw_pointer_cast(d_S_final.data());
                thrust::device_vector<double> d_U_R((size_t)final_rank * (size_t)final_rank);
                double* d_U_R_ptr = thrust::raw_pointer_cast(d_U_R.data());
                thrust::device_vector<double> d_V_R((size_t)final_rank * (size_t)final_rank);
                double* d_V_R_ptr = thrust::raw_pointer_cast(d_V_R.data());
                std::vector<double> h_S((size_t)final_rank), h_Sinv((size_t)final_rank);

                if constexpr (ppopp_full_match) {
                    // Match PPOPP: use R_final from QB, float QR, then double EVD on R^T R.
                    printf("[Info] PPOPP-aligned float path: skipping R_proj; using R_final directly.\n");
                    if (ppopp_float_evd) {
                        printf("[Info] PPOPP float EVD enabled (HQB_PPOPP_FLOAT_EVD=1).\n");
                    }
                    thrust::device_vector<float> d_R_final_T_f((size_t)n * (size_t)final_rank);
                    float* d_R_final_T_f_ptr = thrust::raw_pointer_cast(d_R_final_T_f.data());
                    cublasSgeam(cublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                (int)n, (int)final_rank,
                                &one,
                                d_R_final_ptr, (int)n,
                                &zero,
                                d_R_final_T_f_ptr, (int)n,
                                d_R_final_T_f_ptr, (int)n);
                    (void)stopTimer();
                    printf("R_proj / basis prep finished in %.6f ms.\n", elapsed_time_rproj);

                    printf("[Step4] Mode: Execute final QR (PPOPP match: float QR on R_final_T).\n");
                    startTimer();
                    int lwork_geqrf_final = 0;
                    cusolverDnSgeqrf_bufferSize(cusolverHandle, (int)n, (int)final_rank,
                                                d_R_final_T_f_ptr, (int)n, &lwork_geqrf_final);
                    thrust::device_vector<float> d_work_geqrf((size_t)lwork_geqrf_final);
                    thrust::device_vector<float> d_tau_final((size_t)final_rank);
                    thrust::device_vector<int> d_devInfo_final(1);
                    cusolverDnSgeqrf(cusolverHandle, (int)n, (int)final_rank,
                                     d_R_final_T_f_ptr, (int)n,
                                     thrust::raw_pointer_cast(d_tau_final.data()),
                                     thrust::raw_pointer_cast(d_work_geqrf.data()),
                                     lwork_geqrf_final,
                                     thrust::raw_pointer_cast(d_devInfo_final.data()));
                    thrust::device_vector<float> d_R_true_final_f((size_t)final_rank * (size_t)final_rank);
                    float* d_R_true_final_f_ptr = thrust::raw_pointer_cast(d_R_true_final_f.data());
                    dim3 threads_R_extract2(16, 16);
                    dim3 blocks_R_extract2((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                    extract_upper_triangle_kernel<float><<<blocks_R_extract2, threads_R_extract2, 0, stream>>>(
                        d_R_true_final_f_ptr, (int)final_rank,
                        d_R_final_T_f_ptr, (int)n,
                        (int)final_rank, (int)final_rank);
                    elapsed_time_r_factor = stopTimer();
                    printf("R factorization finished in %f ms.\n", elapsed_time_r_factor);

                    startTimer();
                    int lwork_orgqr_final = 0;
                    cusolverDnSorgqr_bufferSize(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                d_R_final_T_f_ptr, (int)n,
                                                thrust::raw_pointer_cast(d_tau_final.data()),
                                                &lwork_orgqr_final);
                    thrust::device_vector<float> d_work_orgqr((size_t)lwork_orgqr_final);
                    cusolverDnSorgqr(cusolverHandle,
                                     (int)n, (int)final_rank, (int)final_rank,
                                     d_R_final_T_f_ptr, (int)n,
                                     thrust::raw_pointer_cast(d_tau_final.data()),
                                     thrust::raw_pointer_cast(d_work_orgqr.data()),
                                     lwork_orgqr_final,
                                     thrust::raw_pointer_cast(d_devInfo_final.data()));
                    elapsed_time_qbuild_step4 = stopTimer();
                    printf("Step4 Q-build (verification/vector assembly) finished in %f ms.\n", elapsed_time_qbuild_step4);
                    if (debug_mode) {
                        d_R_final_T_verify.resize((size_t)n * (size_t)final_rank);
                        double* d_R_final_T_verify_ptr = thrust::raw_pointer_cast(d_R_final_T_verify.data());
                        convert_kernel<float, double><<<(int)((((size_t)n * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_R_final_T_f_ptr, d_R_final_T_verify_ptr, (size_t)n * (size_t)final_rank);
                        d_R_final_T_ptr = d_R_final_T_verify_ptr;
                    }

                    thrust::device_vector<float> d_R_true_final_T_f((size_t)final_rank * (size_t)final_rank);
                    float* d_R_true_final_T_f_ptr = thrust::raw_pointer_cast(d_R_true_final_T_f.data());
                    cublasSgeam(cublasHandle,
                                CUBLAS_OP_T, CUBLAS_OP_N,
                                (int)final_rank, (int)final_rank,
                                &one,
                                d_R_true_final_f_ptr, (int)final_rank,
                                &zero,
                                d_R_true_final_f_ptr, (int)final_rank,
                                d_R_true_final_T_f_ptr, (int)final_rank);

                    auto time_block = [&](const char* label, auto&& fn) -> float {
                        cudaEvent_t b, e;
                        cudaEventCreate(&b);
                        cudaEventCreate(&e);
                        cudaEventRecord(b, stream);
                        fn();
                        cudaEventRecord(e, stream);
                        cudaEventSynchronize(e);
                        float ms = 0.0f;
                        cudaEventElapsedTime(&ms, b, e);
                        cudaEventDestroy(b);
                        cudaEventDestroy(e);
                        if (ppopp_evd_breakdown) {
                            printf("[PPOPP-EVD] %s: %.6f ms\n", label, ms);
                        }
                        return ms;
                    };

                    startTimer();
                    if (ppopp_float_evd) {
                        // Fast PPOPP variant: float EVD on R*R^T, then lift to double for downstream verification.
                        const float one_f = 1.0f;
                        const float zero_f = 0.0f;
                        thrust::device_vector<float> d_RRT_f((size_t)final_rank * (size_t)final_rank);
                        float* d_RRT_f_ptr = thrust::raw_pointer_cast(d_RRT_f.data());
                        time_block("RRT (float) syrk", [&](){
                            cublasSsyrk(cublasHandle,
                                        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank,
                                        &one_f,
                                        d_R_true_final_T_f_ptr, (int)final_rank,
                                        &zero_f,
                                        d_RRT_f_ptr, (int)final_rank);
                        });

                        thrust::device_vector<float> d_eigvals_f((size_t)final_rank);
                        float* d_eigvals_f_ptr = thrust::raw_pointer_cast(d_eigvals_f.data());
                        size_t lwork_evd_dev = 0, lwork_evd_host = 0;
                        cusolverDnXsyevd_bufferSize(cusolverHandle,
                                                    nullptr,
                                                    CUSOLVER_EIG_MODE_VECTOR,
                                                    CUBLAS_FILL_MODE_LOWER,
                                                    (int)final_rank,
                                                    CUDA_R_32F,
                                                    d_RRT_f_ptr,
                                                    (int)final_rank,
                                                    CUDA_R_32F,
                                                    d_eigvals_f_ptr,
                                                    CUDA_R_32F,
                                                    &lwork_evd_dev,
                                                    &lwork_evd_host);
                        thrust::device_vector<float> d_work_evd_f(lwork_evd_dev);
                        void* h_work_evd = nullptr;
                        if (lwork_evd_host > 0) h_work_evd = malloc(lwork_evd_host);
                        thrust::device_vector<int> d_devInfo_evd(1);
                        time_block("syevd (float)", [&](){
                            cusolverDnXsyevd(cusolverHandle,
                                             nullptr,
                                             CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER,
                                             (int)final_rank,
                                             CUDA_R_32F,
                                             d_RRT_f_ptr,
                                             (int)final_rank,
                                             CUDA_R_32F,
                                             d_eigvals_f_ptr,
                                             CUDA_R_32F,
                                             thrust::raw_pointer_cast(d_work_evd_f.data()),
                                             lwork_evd_dev,
                                             h_work_evd,
                                             lwork_evd_host,
                                             thrust::raw_pointer_cast(d_devInfo_evd.data()));
                        });

                        thrust::device_vector<float> d_S_final_f((size_t)final_rank);
                        time_block("sigma (float)", [&](){
                            thrust::transform(d_eigvals_f.begin(), d_eigvals_f.end(),
                                              d_S_final_f.begin(),
                                              EigenToSigmaF());
                        });

                        thrust::device_vector<float> d_temp_UTR_f((size_t)final_rank * (size_t)final_rank);
                        float* d_temp_UTR_f_ptr = thrust::raw_pointer_cast(d_temp_UTR_f.data());
                        time_block("U^T * R (float)", [&](){
                            cublasSgemm(cublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank, (int)final_rank,
                                        &one_f,
                                        d_RRT_f_ptr, (int)final_rank,
                                        d_R_true_final_T_f_ptr, (int)final_rank,
                                        &zero_f,
                                        d_temp_UTR_f_ptr, (int)final_rank);
                        });

                        std::vector<float> h_S_f((size_t)final_rank), h_Sinv_f((size_t)final_rank);
                        thrust::copy(d_S_final_f.begin(), d_S_final_f.end(), h_S_f.begin());
                        const float tol_sigma_f = 1e-7f;
                        for (int i = 0; i < final_rank; ++i) {
                            if (h_S_f[(size_t)i] > tol_sigma_f) {
                                h_Sinv_f[(size_t)i] = 1.0f / h_S_f[(size_t)i];
                            } else {
                                h_Sinv_f[(size_t)i] = 0.0f;
                            }
                            h_S[(size_t)i] = static_cast<double>(h_S_f[(size_t)i]);
                            h_Sinv[(size_t)i] = static_cast<double>(h_Sinv_f[(size_t)i]);
                        }
                        time_block("scale V^T (float)", [&](){
                            for (int i = 0; i < final_rank; ++i) {
                                float alpha = h_Sinv_f[(size_t)i];
                                cublasSscal(cublasHandle,
                                            (int)final_rank,
                                            &alpha,
                                            d_temp_UTR_f_ptr + (size_t)i, (int)final_rank);
                            }
                        });

                        thrust::device_vector<float> d_V_R_f((size_t)final_rank * (size_t)final_rank);
                        float* d_V_R_f_ptr = thrust::raw_pointer_cast(d_V_R_f.data());
                        time_block("V^T -> V (float)", [&](){
                            cublasSgeam(cublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank,
                                        &one_f,
                                        d_temp_UTR_f_ptr, (int)final_rank,
                                        &zero_f,
                                        d_temp_UTR_f_ptr, (int)final_rank,
                                        d_V_R_f_ptr, (int)final_rank);
                        });

                        time_block("lift float->double", [&](){
                            convert_kernel<float, double><<<(int)((((size_t)final_rank * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                d_RRT_f_ptr, d_U_R_ptr, (size_t)final_rank * (size_t)final_rank);
                            convert_kernel<float, double><<<(int)((((size_t)final_rank * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                d_V_R_f_ptr, d_V_R_ptr, (size_t)final_rank * (size_t)final_rank);
                            convert_kernel<float, double><<<(int)(((size_t)final_rank + 255) / 256), 256, 0, stream>>>(
                                thrust::raw_pointer_cast(d_S_final_f.data()), d_S_final_ptr, (size_t)final_rank);
                        });

                        if (h_work_evd) free(h_work_evd);
                    } else {
                        thrust::device_vector<double> d_R_double((size_t)final_rank * (size_t)final_rank);
                        double* d_R_double_ptr = thrust::raw_pointer_cast(d_R_double.data());
                        time_block("R -> double", [&](){
                            convert_kernel<float, double><<<(int)((((size_t)final_rank * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                d_R_true_final_T_f_ptr, d_R_double_ptr, (size_t)final_rank * (size_t)final_rank);
                        });

                        thrust::device_vector<double> d_R_orig((size_t)final_rank * (size_t)final_rank);
                        double* d_R_orig_ptr = thrust::raw_pointer_cast(d_R_orig.data());
                        cudaMemcpyAsync(d_R_orig_ptr, d_R_double_ptr,
                                        sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                        cudaMemcpyDeviceToDevice, stream);

                        thrust::device_vector<double> d_RRT((size_t)final_rank * (size_t)final_rank);
                        double* d_RRT_ptr = thrust::raw_pointer_cast(d_RRT.data());
                        time_block("RRT (double) syrk", [&](){
                            cublasDsyrk(cublasHandle,
                                        CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank,
                                        &one_d,
                                        d_R_double_ptr, (int)final_rank,
                                        &zero_d,
                                        d_RRT_ptr, (int)final_rank);
                        });

                        thrust::device_vector<double> d_eigvals((size_t)final_rank);
                        double* d_eigvals_ptr = thrust::raw_pointer_cast(d_eigvals.data());
                        size_t lwork_evd_dev = 0, lwork_evd_host = 0;
                        cusolverDnXsyevd_bufferSize(cusolverHandle,
                                                    nullptr,
                                                    CUSOLVER_EIG_MODE_VECTOR,
                                                    CUBLAS_FILL_MODE_LOWER,
                                                    (int)final_rank,
                                                    CUDA_R_64F,
                                                    d_RRT_ptr,
                                                    (int)final_rank,
                                                    CUDA_R_64F,
                                                    d_eigvals_ptr,
                                                    CUDA_R_64F,
                                                    &lwork_evd_dev,
                                                    &lwork_evd_host);
                        thrust::device_vector<double> d_work_evd(lwork_evd_dev);
                        void* h_work_evd = nullptr;
                        if (lwork_evd_host > 0) h_work_evd = malloc(lwork_evd_host);
                        thrust::device_vector<int> d_devInfo_evd(1);
                        time_block("syevd (double)", [&](){
                            cusolverDnXsyevd(cusolverHandle,
                                             nullptr,
                                             CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER,
                                             (int)final_rank,
                                             CUDA_R_64F,
                                             d_RRT_ptr,
                                             (int)final_rank,
                                             CUDA_R_64F,
                                             d_eigvals_ptr,
                                             CUDA_R_64F,
                                             thrust::raw_pointer_cast(d_work_evd.data()),
                                             lwork_evd_dev,
                                             h_work_evd,
                                             lwork_evd_host,
                                             thrust::raw_pointer_cast(d_devInfo_evd.data()));
                        });

                        time_block("sigma (double)", [&](){
                            thrust::transform(d_eigvals.begin(), d_eigvals.end(),
                                              d_S_final.begin(),
                                              EigenToSigma());
                        });

                        cudaMemcpyAsync(d_U_R_ptr,
                                        d_RRT_ptr,
                                        sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                        cudaMemcpyDeviceToDevice,
                                        stream);

                        thrust::device_vector<double> d_temp_UTR((size_t)final_rank * (size_t)final_rank);
                        double* d_temp_UTR_ptr = thrust::raw_pointer_cast(d_temp_UTR.data());
                        time_block("U^T * R (double)", [&](){
                            cublasDgemm(cublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank, (int)final_rank,
                                        &one_d,
                                        d_U_R_ptr, (int)final_rank,
                                        d_R_orig_ptr, (int)final_rank,
                                        &zero_d,
                                        d_temp_UTR_ptr, (int)final_rank);
                        });

                        thrust::copy(d_S_final.begin(), d_S_final.end(), h_S.begin());
                        const double tol_sigma = 1e-14;
                        for (int i = 0; i < final_rank; ++i) {
                            if (h_S[(size_t)i] > tol_sigma) {
                                h_Sinv[(size_t)i] = 1.0 / h_S[(size_t)i];
                            } else {
                                h_Sinv[(size_t)i] = 0.0;
                            }
                        }
                        time_block("scale V^T (double)", [&](){
                            for (int i = 0; i < final_rank; ++i) {
                                double alpha = h_Sinv[(size_t)i];
                                // Scale row i of (U^T R) to form V^T = S^{-1} U^T R
                                cublasDscal(cublasHandle,
                                            (int)final_rank,
                                            &alpha,
                                            d_temp_UTR_ptr + (size_t)i, (int)final_rank);
                            }
                        });
                        time_block("V^T -> V (double)", [&](){
                            cublasDgeam(cublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank,
                                        &one_d,
                                        d_temp_UTR_ptr, (int)final_rank,
                                        &zero_d,
                                        d_temp_UTR_ptr, (int)final_rank,
                                        d_V_R_ptr, (int)final_rank);
                        });

                        if (h_work_evd) free(h_work_evd);
                    }
                } else {
                    d_R_final_T.resize((size_t)n * (size_t)final_rank);
                    d_R_final_T_ptr = thrust::raw_pointer_cast(d_R_final_T.data());
                    thrust::device_vector<double> d_R_true_final((size_t)final_rank * (size_t)final_rank);
                    double* d_R_true_final_ptr = thrust::raw_pointer_cast(d_R_true_final.data());
                    thrust::device_vector<double> d_R_true_final_T((size_t)final_rank * (size_t)final_rank);
                    double* d_R_true_final_T_ptr = thrust::raw_pointer_cast(d_R_true_final_T.data());
                    if (use_double_rproj) {
                        thrust::device_vector<double> d_R_proj((size_t)final_rank * (size_t)n);
                        double* d_R_proj_ptr = thrust::raw_pointer_cast(d_R_proj.data());
                        cublasDgemm(cublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    (int)final_rank, (int)n, (int)m_effective,
                                    &one_d,
                                    d_Q_basis_effective_ptr, ld_Q_basis_effective,
                                    d_A_backup_fast_ptr, (int)m_effective,
                                    &zero_d,
                                    d_R_proj_ptr, (int)final_rank);

                        cublasDgeam(cublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    (int)n, (int)final_rank,
                                    &one_d,
                                    d_R_proj_ptr, (int)final_rank,
                                    &zero_d,
                                    d_R_final_T_ptr, (int)n,
                                    d_R_final_T_ptr, (int)n);
                        if (skip_final_qr) {
                            dim3 threads_R_extract2(16, 16);
                            dim3 blocks_R_extract2((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                            extract_upper_triangle_kernel<double><<<blocks_R_extract2, threads_R_extract2, 0, stream>>>(
                                d_R_true_final_ptr, (int)final_rank,
                                d_R_proj_ptr, (int)final_rank,
                                (int)final_rank, (int)final_rank);
                            cublasDgeam(cublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank,
                                        &one_d,
                                        d_R_true_final_ptr, (int)final_rank,
                                        &zero_d,
                                        d_R_true_final_ptr, (int)final_rank,
                                        d_R_true_final_T_ptr, (int)final_rank);
                        }
                    } else {
                        if (debug_mode) {
                            printf("[Info] R_proj path: using float/TF32 GEMM for medium/low accuracy.\n");
                        }
                        thrust::device_vector<float> d_Q_basis_effective_float((size_t)m_effective * (size_t)final_rank);
                        float* d_Q_basis_effective_float_ptr = thrust::raw_pointer_cast(d_Q_basis_effective_float.data());
                        convert_kernel<double, float><<<(int)((((size_t)m_effective * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_Q_basis_effective_ptr, d_Q_basis_effective_float_ptr, (size_t)m_effective * (size_t)final_rank);

                        thrust::device_vector<float> d_R_proj_float((size_t)final_rank * (size_t)n);
                        float* d_R_proj_float_ptr = thrust::raw_pointer_cast(d_R_proj_float.data());
                        BlasCaller<float>::gemm(cublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                (int)final_rank, (int)n, (int)m_effective,
                                                &one,
                                                d_Q_basis_effective_float_ptr, (int)m_effective,
                                                d_A_effective_raw, (int)m_effective,
                                                &zero,
                                                d_R_proj_float_ptr, (int)final_rank);

                        thrust::device_vector<float> d_R_final_T_float((size_t)n * (size_t)final_rank);
                        float* d_R_final_T_float_ptr = thrust::raw_pointer_cast(d_R_final_T_float.data());
                        BlasCaller<float>::geam(cublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                (int)n, (int)final_rank,
                                                &one,
                                                d_R_proj_float_ptr, (int)final_rank,
                                                &zero,
                                                d_R_final_T_float_ptr, (int)n,
                                                d_R_final_T_float_ptr, (int)n);

                        convert_kernel<float, double><<<(int)((((size_t)n * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_R_final_T_float_ptr, d_R_final_T_ptr, (size_t)n * (size_t)final_rank);
                        if (skip_final_qr) {
                            thrust::device_vector<float> d_R_true_final_float((size_t)final_rank * (size_t)final_rank);
                            float* d_R_true_final_float_ptr = thrust::raw_pointer_cast(d_R_true_final_float.data());
                            dim3 threads_R_extract2(16, 16);
                            dim3 blocks_R_extract2((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                            extract_upper_triangle_kernel<float><<<blocks_R_extract2, threads_R_extract2, 0, stream>>>(
                                d_R_true_final_float_ptr, (int)final_rank,
                                d_R_proj_float_ptr, (int)final_rank,
                                (int)final_rank, (int)final_rank);
                            convert_kernel<float, double><<<(int)((((size_t)final_rank * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                d_R_true_final_float_ptr, d_R_true_final_ptr, (size_t)final_rank * (size_t)final_rank);
                            cublasDgeam(cublasHandle,
                                        CUBLAS_OP_T, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank,
                                        &one_d,
                                        d_R_true_final_ptr, (int)final_rank,
                                        &zero_d,
                                        d_R_true_final_ptr, (int)final_rank,
                                        d_R_true_final_T_ptr, (int)final_rank);
                        }
                    }

                    elapsed_time_rproj = stopTimer();
                    printf("R_proj / basis prep finished in %f ms.\n", elapsed_time_rproj);

                    thrust::device_vector<double> d_tau_final;
                    thrust::device_vector<int> d_devInfo_final;
                    if (skip_final_qr) {
                        printf("[Step4] Mode: Bypass final QR (n/final_rank=%.3f > threshold %.3f).\n",
                               ratio_n_over_rank, kRFinalQrRatioThreshold);
                        printf("[Info] Final QR bypassed: using existing upper-triangular R block.\n");
                    } else {
                        printf("[Step4] Mode: Execute final QR (n/final_rank=%.3f <= threshold %.3f).\n",
                               ratio_n_over_rank, kRFinalQrRatioThreshold);
                        startTimer();
                        int lwork_geqrf_final = 0;
                        cusolverDnDgeqrf_bufferSize(cusolverHandle, (int)n, (int)final_rank,
                                                    d_R_final_T_ptr, (int)n, &lwork_geqrf_final);
                        thrust::device_vector<double> d_work_geqrf((size_t)lwork_geqrf_final);
                        d_tau_final.resize((size_t)final_rank);
                        d_devInfo_final.resize(1);
                        cusolverDnDgeqrf(cusolverHandle, (int)n, (int)final_rank,
                                         d_R_final_T_ptr, (int)n,
                                         thrust::raw_pointer_cast(d_tau_final.data()),
                                         thrust::raw_pointer_cast(d_work_geqrf.data()),
                                         lwork_geqrf_final,
                                         thrust::raw_pointer_cast(d_devInfo_final.data()));

                        dim3 threads_R_extract2(16, 16);
                        dim3 blocks_R_extract2((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                        extract_upper_triangle_kernel<double><<<blocks_R_extract2, threads_R_extract2, 0, stream>>>(
                            d_R_true_final_ptr, (int)final_rank,
                            d_R_final_T_ptr, (int)n,
                            (int)final_rank, (int)final_rank);

                        cublasDgeam(cublasHandle,
                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                    (int)final_rank, (int)final_rank,
                                    &one_d,
                                    d_R_true_final_ptr, (int)final_rank,
                                    &zero_d,
                                    d_R_true_final_ptr, (int)final_rank,
                                    d_R_true_final_T_ptr, (int)final_rank);
                        elapsed_time_r_factor = stopTimer();
                        printf("R factorization finished in %f ms.\n", elapsed_time_r_factor);

                        startTimer();
                        int lwork_orgqr_final = 0;
                        cusolverDnDorgqr_bufferSize(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                    d_R_final_T_ptr, (int)n,
                                                    thrust::raw_pointer_cast(d_tau_final.data()),
                                                    &lwork_orgqr_final);
                        thrust::device_vector<double> d_work_orgqr((size_t)lwork_orgqr_final);
                        cusolverDnDorgqr(cusolverHandle,
                                         (int)n, (int)final_rank, (int)final_rank,
                                         d_R_final_T_ptr, (int)n,
                                         thrust::raw_pointer_cast(d_tau_final.data()),
                                         thrust::raw_pointer_cast(d_work_orgqr.data()),
                                         lwork_orgqr_final,
                                         thrust::raw_pointer_cast(d_devInfo_final.data()));
                        elapsed_time_qbuild_step4 = stopTimer();
                        printf("Step4 Q-build (verification/vector assembly) finished in %f ms.\n", elapsed_time_qbuild_step4);
                    }
                    startTimer();
                    thrust::device_vector<double> d_Rd((size_t)final_rank * (size_t)final_rank);
                    double* d_Rd_ptr = thrust::raw_pointer_cast(d_Rd.data());
                    cudaMemcpyAsync(d_Rd_ptr,
                                    d_R_true_final_T_ptr,
                                    sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                    cudaMemcpyDeviceToDevice,
                                    stream);

                    thrust::device_vector<double> d_C((size_t)final_rank * (size_t)final_rank);
                    double* d_C_ptr = thrust::raw_pointer_cast(d_C.data());
                    // PPOPP-aligned path always applies QR(B^T), so Step5 follows the
                    // left-side route with G = \hat{R}\hat{R}^T.
                    cublasDsyrk(cublasHandle,
                                CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                (int)final_rank, (int)final_rank,
                                &one_d,
                                d_Rd_ptr, (int)final_rank,
                                &zero_d,
                                d_C_ptr, (int)final_rank);

                    thrust::device_vector<double> d_eigvals((size_t)final_rank);
                    double* d_eigvals_ptr = thrust::raw_pointer_cast(d_eigvals.data());
                    size_t lwork_evd_dev = 0, lwork_evd_host = 0;
                    cusolverDnXsyevd_bufferSize(cusolverHandle,
                                                nullptr,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUBLAS_FILL_MODE_LOWER,
                                                (int)final_rank,
                                                CUDA_R_64F,
                                                d_C_ptr,
                                                (int)final_rank,
                                                CUDA_R_64F,
                                                d_eigvals_ptr,
                                                CUDA_R_64F,
                                                &lwork_evd_dev,
                                                &lwork_evd_host);
                    thrust::device_vector<double> d_work_evd(lwork_evd_dev);
                    void* h_work_evd = nullptr;
                    if (lwork_evd_host > 0) h_work_evd = malloc(lwork_evd_host);
                    thrust::device_vector<int> d_devInfo_evd(1);
                    cusolverDnXsyevd(cusolverHandle,
                                     nullptr,
                                     CUSOLVER_EIG_MODE_VECTOR,
                                     CUBLAS_FILL_MODE_LOWER,
                                     (int)final_rank,
                                     CUDA_R_64F,
                                     d_C_ptr,
                                     (int)final_rank,
                                     CUDA_R_64F,
                                     d_eigvals_ptr,
                                     CUDA_R_64F,
                                     thrust::raw_pointer_cast(d_work_evd.data()),
                                     lwork_evd_dev,
                                     h_work_evd,
                                     lwork_evd_host,
                                     thrust::raw_pointer_cast(d_devInfo_evd.data()));

                    thrust::transform(d_eigvals.begin(), d_eigvals.end(),
                                      d_S_final.begin(),
                                      EigenToSigma());

                    if (need_vectors()) {
                        cudaMemcpyAsync(d_U_R_ptr,
                                        d_C_ptr,
                                        sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                        cudaMemcpyDeviceToDevice,
                                        stream);
                    }

                    thrust::copy(d_S_final.begin(), d_S_final.end(), h_S.begin());
                    const double tol_sigma = 1e-14;
                    for (int i = 0; i < final_rank; ++i) {
                        if (h_S[(size_t)i] > tol_sigma) {
                            h_Sinv[(size_t)i] = 1.0 / h_S[(size_t)i];
                        } else {
                            h_Sinv[(size_t)i] = 0.0;
                        }
                    }
                    if (need_vectors()) {
                        thrust::device_vector<double> d_temp((size_t)final_rank * (size_t)final_rank);
                        double* d_temp_ptr = thrust::raw_pointer_cast(d_temp.data());
                        cublasDgemm(cublasHandle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    (int)final_rank, (int)final_rank, (int)final_rank,
                                    &one_d,
                                    d_Rd_ptr, (int)final_rank,
                                    d_U_R_ptr, (int)final_rank,
                                    &zero_d,
                                    d_temp_ptr, (int)final_rank);
                        for (int i = 0; i < final_rank; ++i) {
                            double alpha = h_Sinv[(size_t)i];
                            cublasDscal(cublasHandle,
                                        (int)final_rank,
                                        &alpha,
                                        d_temp_ptr + (size_t)i * (size_t)final_rank, 1);
                        }
                        cudaMemcpyAsync(d_V_R_ptr,
                                        d_temp_ptr,
                                        sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                        cudaMemcpyDeviceToDevice,
                                        stream);
                    }

                    if (h_work_evd) free(h_work_evd);
                }

                float elapsed_time_svd = stopTimer();
                printf("EVD-based SVD finished successfully in %f ms.\n", elapsed_time_svd);
                const float compression_time = elapsed_time_QB + elapsed_time_Q;
                const float svd_tail_time = elapsed_time_rproj + elapsed_time_r_factor + elapsed_time_svd;
                float total_time = compression_time + svd_tail_time;
                printf("Compression time (QB + Q): %f ms.\n", compression_time);
                printf("SVD tail time (R_proj + R_factor + EVD-SVD): %f ms.\n", svd_tail_time);
                printf("Step4 Q-build time (verification/vector assembly): %f ms.\n", elapsed_time_qbuild_step4);
                printf("Total time (QB + Q + R_proj + R_factor + EVD-SVD): %f ms.\n", total_time);
                if (debug_mode) {
                    ToFloat to_float;
                    thrust::device_vector<float> d_Q_basis_f((size_t)m_effective * (size_t)final_rank);
                    thrust::device_vector<float> d_U_R_f((size_t)final_rank * (size_t)final_rank);
                    thrust::device_vector<float> d_V_final_for_verify_f((size_t)n * (size_t)final_rank);
                    thrust::device_vector<float> d_S_final_f((size_t)final_rank);
                    thrust::device_vector<float> d_A_backup_f((size_t)m_effective * (size_t)n);

                    if (need_pad) {
                        thrust::transform(d_Q_basis_effective.begin(), d_Q_basis_effective.end(), d_Q_basis_f.begin(), to_float);
                    } else {
                        thrust::transform(d_Q_basis.begin(), d_Q_basis.end(), d_Q_basis_f.begin(), to_float);
                    }
                    thrust::transform(d_U_R.begin(), d_U_R.end(), d_U_R_f.begin(), to_float);
                    thrust::transform(d_S_final.begin(), d_S_final.end(), d_S_final_f.begin(), to_float);
                    thrust::transform(d_A_backup_fast.begin(), d_A_backup_fast.end(), d_A_backup_f.begin(), to_float);
                    thrust::device_vector<double> d_V_final_for_verify((size_t)n * (size_t)final_rank);
                    double* d_V_final_for_verify_ptr = thrust::raw_pointer_cast(d_V_final_for_verify.data());
                    if (skip_final_qr) {
                        thrust::device_vector<double> d_U_scaled_inv((size_t)final_rank * (size_t)final_rank);
                        double* d_U_scaled_inv_ptr = thrust::raw_pointer_cast(d_U_scaled_inv.data());
                        cudaMemcpyAsync(d_U_scaled_inv_ptr,
                                        d_U_R_ptr,
                                        sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                        cudaMemcpyDeviceToDevice,
                                        stream);
                        for (int i = 0; i < final_rank; ++i) {
                            double alpha = h_Sinv[(size_t)i];
                            cublasDscal(cublasHandle,
                                        (int)final_rank,
                                        &alpha,
                                        d_U_scaled_inv_ptr + (size_t)i * (size_t)final_rank, 1);
                        }
                        cublasDgemm(cublasHandle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    (int)n, (int)final_rank, (int)final_rank,
                                    &one_d,
                                    d_R_final_T_ptr, (int)n,
                                    d_U_scaled_inv_ptr, (int)final_rank,
                                    &zero_d,
                                    d_V_final_for_verify_ptr, (int)n);
                    } else {
                        cublasDgemm(cublasHandle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    (int)n, (int)final_rank, (int)final_rank,
                                    &one_d,
                                    d_R_final_T_ptr, (int)n,
                                    d_V_R_ptr, (int)final_rank,
                                    &zero_d,
                                    d_V_final_for_verify_ptr, (int)n);
                    }
                    thrust::transform(d_V_final_for_verify.begin(), d_V_final_for_verify.end(), d_V_final_for_verify_f.begin(), to_float);

                    printf("\n--- Verifying the rSVD results [Q_basis, U_R, S, V] (Precision: float) ---\n");
                    float q_ortho_error = verify_orthogonality<float>(cublasHandle, (int)m_effective, (int)final_rank,
                                                                      thrust::raw_pointer_cast(d_Q_basis_f.data()), (int)m_effective);
                    printf("Orthogonality of Q_basis ||Q_basis'*Q_basis - I||_F = %e\n", q_ortho_error);

                    float u_r_ortho_error = verify_orthogonality<float>(cublasHandle, (int)final_rank, (int)final_rank,
                                                                        thrust::raw_pointer_cast(d_U_R_f.data()), (int)final_rank);
                    printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %e\n", u_r_ortho_error);

                    float v_ortho_error = verify_orthogonality<float>(cublasHandle, (int)n, (int)final_rank,
                                                                      thrust::raw_pointer_cast(d_V_final_for_verify_f.data()), (int)n);
                    printf("Orthogonality of V ||V'*V - I||_F = %e\n", v_ortho_error);

                    printf("\nVerifying Reconstruction Error ||A - Q_basis*U_R*S*V^T||_F...\n");
                    thrust::device_vector<float> d_US_temp((size_t)final_rank * (size_t)final_rank);
                    float* d_US_temp_ptr = thrust::raw_pointer_cast(d_US_temp.data());
                    dim3 threads_scale(16, 16);
                    dim3 blocks_scale((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                    scale_columns_kernel<<<blocks_scale, threads_scale, 0, stream>>>(
                        d_US_temp_ptr,
                        thrust::raw_pointer_cast(d_U_R_f.data()),
                        thrust::raw_pointer_cast(d_S_final_f.data()),
                        (int)final_rank,
                        (int)final_rank,
                        (int)final_rank,
                        (int)final_rank);

                    thrust::device_vector<float> d_QUS((size_t)m_effective * (size_t)final_rank);
                    float* d_QUS_ptr = thrust::raw_pointer_cast(d_QUS.data());
                    float f_one = 1.0f, f_zero = 0.0f, f_minus_one = -1.0f;
                    cublasSgemm(cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                (int)m_effective, (int)final_rank, (int)final_rank,
                                &f_one,
                                thrust::raw_pointer_cast(d_Q_basis_f.data()), (int)m_effective,
                                d_US_temp_ptr, (int)final_rank,
                                &f_zero,
                                d_QUS_ptr, (int)m_effective);

                    thrust::device_vector<float> d_A_reconstructed((size_t)m_effective * (size_t)n);
                    float* d_A_reconstructed_ptr = thrust::raw_pointer_cast(d_A_reconstructed.data());
                    cublasSgemm(cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_T,
                                (int)m_effective, (int)n, (int)final_rank,
                                &f_one,
                                d_QUS_ptr, (int)m_effective,
                                thrust::raw_pointer_cast(d_V_final_for_verify_f.data()), (int)n,
                                &f_zero,
                                d_A_reconstructed_ptr, (int)m_effective);
                    cublasSgeam(cublasHandle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                (int)m_effective, (int)n,
                                &f_one,
                                thrust::raw_pointer_cast(d_A_backup_f.data()), (int)m_effective,
                                &f_minus_one,
                                d_A_reconstructed_ptr, (int)m_effective,
                                d_A_reconstructed_ptr, (int)m_effective);

                    float reconstruction_error = 0.0f;
                    cublasSnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n), d_A_reconstructed_ptr, 1, &reconstruction_error);
                    printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %e\n", reconstruction_error);
                    float norm_A_original = 0.0f;
                    cublasSnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n), thrust::raw_pointer_cast(d_A_backup_f.data()), 1, &norm_A_original);
                    float relative_error = (norm_A_original > 1e-12f) ? (reconstruction_error / norm_A_original) : reconstruction_error;
                    printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n", relative_error);
                }
                cublasSetMathMode(cublasHandle, prev_math_mode_float_branch);
                return;
            }
        }

        printf("=== Step 3: Run QB Algorithm ===\n");
        int current_device = 0;
        CUDA_CHECK(cudaGetDevice(&current_device));
        int max_shmem_optin = 0;
        cudaError_t shmem_attr_status = cudaDeviceGetAttribute(
            &max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, current_device);
        if (shmem_attr_status != cudaSuccess || max_shmem_optin == 0) {
            CUDA_CHECK(cudaDeviceGetAttribute(
                &max_shmem_optin, cudaDevAttrMaxSharedMemoryPerBlock, current_device));
        }
        auto calc_tsqr_caps = [&](size_t elem_bytes) {
            size_t hh = (max_shmem_optin > 0)
                            ? (size_t)max_shmem_optin / (TSQR_BLOCK_SIZE * elem_bytes)
                            : (size_t)k;
            if (hh == 0) hh = 1;
            size_t mgs = hh;
            if (max_shmem_optin > 0) {
                const double cap_elems = (double)max_shmem_optin / (double)elem_bytes;
                const double disc = (double)TSQR_BLOCK_SIZE * (double)TSQR_BLOCK_SIZE + 4.0 * cap_elems;
                const double n_root = (-(double)TSQR_BLOCK_SIZE + sqrt(disc)) / 2.0;
                mgs = (n_root > 1.0) ? (size_t)floor(n_root) : (size_t)1;
            }
            return std::make_pair(hh, mgs);
        };
        auto [tsqr_max_cols_hh, tsqr_max_cols_mgs] = calc_tsqr_caps(sizeof(T_compute));
        constexpr bool float_compute_path = std::is_same_v<T_compute, float>;
        constexpr bool lowp_storage_path =
            std::is_same_v<T_storage, __half> || std::is_same_v<T_storage, __nv_bfloat16>;
        const bool force_flat_fast_float_path = float_compute_path && fast_mode;
        const bool use_ppopp_float_path = false;
        bool qb_panel_tsqr_in_double = false;
        if constexpr (float_compute_path) {
            // For half/bfloat16 storage with float compute, keep the automatic
            // TSQR precision choice driven by accuracy/debug settings only.
            bool tsqr_in_double_pref = false;
            if (!use_ppopp_float_path && !fast_mode) {
                if constexpr (lowp_storage_path) {
                    tsqr_in_double_pref = (g_accuracy == AccuracyLevel::High);
                } else {
                    tsqr_in_double_pref = (debug_mode || g_accuracy != AccuracyLevel::Low);
                }
            }
            if (tsqr_in_double_pref) {
                const auto [hh_double, mgs_double] = calc_tsqr_caps(sizeof(double));
                tsqr_max_cols_hh = hh_double;
                tsqr_max_cols_mgs = mgs_double;
                qb_panel_tsqr_in_double = true;
            }
        }
        const size_t tsqr_max_cols = std::max<size_t>(1, std::min(tsqr_max_cols_hh, tsqr_max_cols_mgs));
        const long int qb_block_k = std::min<long int>(k, (long int)tsqr_max_cols);
        if (qb_block_k <= 0) {
            printf("[Error] Invalid effective QB block size.\n");
            return;
        }
        if (qb_block_k < k) {
            printf("[Info] Effective TSQR panel width capped from %ld to %ld (hh_limit=%zu, mgs_limit=%zu, tsqr_double=%s).\n",
                   k, qb_block_k, tsqr_max_cols_hh, tsqr_max_cols_mgs, qb_panel_tsqr_in_double ? "on" : "off");
        }
        const size_t num_blocks = (n + qb_block_k - 1) / qb_block_k;
        const bool guaranteed_skip_final_qr =
            (real_rank > 0)
                ? (static_cast<double>(n) / static_cast<double>(real_rank) > kRFinalQrRatioThreshold)
                : true;
        // 分配筛选后组合的Q和R
        thrust::device_vector<T_compute> d_Q_final(m * real_rank);
        T_compute* d_Q_final_ptr = thrust::raw_pointer_cast(d_Q_final.data());
        cudaMemset(d_Q_final_ptr, 0, (size_t)m * real_rank * sizeof(T_compute));

        thrust::device_vector<T_compute> d_R_final(n * n);
        T_compute* d_R_final_ptr = thrust::raw_pointer_cast(d_R_final.data());
        cudaMemset(d_R_final_ptr, 0, (size_t)n * n * sizeof(T_compute));

        // 创建合理大小的W和Y矩阵(m*real_rank)
        thrust::device_vector<T_compute> d_W_final(m * real_rank);
        T_compute* d_W_final_ptr = thrust::raw_pointer_cast(d_W_final.data());
        cudaMemset(d_W_final_ptr, 0, m * real_rank * sizeof(T_compute));

        thrust::device_vector<T_compute> d_Y_final(m * real_rank);
        T_compute* d_Y_final_ptr = thrust::raw_pointer_cast(d_Y_final.data());
        cudaMemset(d_Y_final_ptr, 0, m * real_rank * sizeof(T_compute));
        thrust::device_vector<double> d_W_final_accum_double;
        thrust::device_vector<double> d_Y_final_accum_double;
        double* d_W_final_accum_double_ptr = nullptr;
        double* d_Y_final_accum_double_ptr = nullptr;
        if constexpr (std::is_same_v<T_storage, float> && std::is_same_v<T_compute, float>) {
            if (!use_ppopp_float_path) {
                d_W_final_accum_double.resize((size_t)m * (size_t)real_rank);
                d_Y_final_accum_double.resize((size_t)m * (size_t)real_rank);
                d_W_final_accum_double_ptr = thrust::raw_pointer_cast(d_W_final_accum_double.data());
                d_Y_final_accum_double_ptr = thrust::raw_pointer_cast(d_Y_final_accum_double.data());
                cudaMemset(d_W_final_accum_double_ptr, 0, (size_t)m * (size_t)real_rank * sizeof(double));
                cudaMemset(d_Y_final_accum_double_ptr, 0, (size_t)m * (size_t)real_rank * sizeof(double));
            }
        }
        
        
        // 为 QR 分解和 Q 的构造分配最大的工作空间
        const int k_max = (int)std::min((long)n, qb_block_k);
        int lwork_geqrf_max = 0;
        SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, m, k_max, d_A_ptr, m, &lwork_geqrf_max);
        int lwork_orgqr_max = 0;
        SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, m, k_max, k_max, d_A_ptr, m, nullptr, &lwork_orgqr_max);

        int lwork_max_qr = std::max(lwork_geqrf_max, lwork_orgqr_max);
        thrust::device_vector<T_compute> d_work_qr(lwork_max_qr);
        thrust::device_vector<int>       d_devInfo(1);
        thrust::device_vector<T_compute> d_tau(k_max);
        thrust::device_vector<T_compute> d_R_tsqr(k_max * k_max);
        // 每面板按需查询 GETRF 工作区，复用缓冲存放 R 行尾范数
        thrust::device_vector<T_compute> row_tail_norms_gpu_buf(k_max);
        thrust::device_vector<double> row_tail_norms_sq_double_buf(k_max);

        int max_block_num = (m + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;
        size_t tsqr_work_size = (size_t)max_block_num * k_max * k_max + (size_t)m * k_max *6; 
        thrust::device_vector<T_compute> d_work_tsqr(tsqr_work_size);
        constexpr bool lowp_storage_compute_path =
            std::is_same_v<T_compute,float> &&
            (std::is_same_v<T_storage, __half> || std::is_same_v<T_storage, __nv_bfloat16>);
        auto read_lowp_dim_threshold = [](const char* name, long default_value) {
            if (const char* s = std::getenv(name)) {
                long v = std::atol(s);
                if (v > 0) return v;
            }
            return default_value;
        };
        const long lowp_min_r = read_lowp_dim_threshold("HQB_MIN_R", 64);
        const long lowp_min_nt = read_lowp_dim_threshold("HQB_MIN_NT", 64);
        const long max_lowp_panel_width = std::min<long>((long)qb_block_k, (long)real_rank);
        const bool lowp_force_on = (hqbtune::tuner().force > 0);
        const bool lowp_runtime_possible = lowp_storage_compute_path &&
                                           (lowp_force_on ||
                                            (max_lowp_panel_width >= lowp_min_r &&
                                             n >= (lowp_min_r + lowp_min_nt)));
        // 校准自适应低精度选择器 (only for float compute + half/bf16 storage)
        if constexpr (lowp_storage_compute_path) {
            // 提前一次性预热 CUDA/cuBLAS，减少后续冷启动抖动
            if (lowp_runtime_possible) {
                hqbtune::early_cuda_cublas_init();
                // 使用实际的 k 进行标定，便于覆盖满秩/部分秩/低秩场景
                hqbtune::calibrate_once(cublasHandle, stream,(int)std::min<long>(m, 8192),(int)qb_block_k,(int)std::min<long>(n, 8192));
            }
        }
        // TSQR 内核选择阈值改为固定 64
        bool printed_path_info = false;
        // 复用缓冲，避免循环内频繁分配/释放
        thrust::device_vector<T_compute> buf_Q_filtered;        
        thrust::device_vector<T_compute> buf_R_temp;           
        thrust::device_vector<T_compute> buf_B;                 
        thrust::device_vector<T_compute> buf_W;                 
        thrust::device_vector<T_compute> buf_Y;                 
        thrust::device_vector<T_compute> buf_getrf_work;        
        thrust::device_vector<T_compute> buf_Temp1_fp32;        
        thrust::device_vector<double> buf_Q_filtered_double;
        thrust::device_vector<double> buf_B_double;
        thrust::device_vector<double> buf_W_double;
        thrust::device_vector<double> buf_Y_double;
        thrust::device_vector<double> buf_getrf_work_double;
        thrust::device_vector<double> buf_A_trailing_double;
        thrust::device_vector<double> buf_Temp1_double;
        // 低精路径缓冲
        thrust::device_vector<T_storage> buf_W_lowp;           
        thrust::device_vector<T_storage> buf_Y_lowp;           
        thrust::device_vector<T_storage> buf_A_trailing_lowp;   
        thrust::device_vector<T_storage> buf_Temp1_lowp;        
        // 转换与回写的辅助流与事件：仅低精 storage->float 路径才需要。
        cudaStream_t stream_conv = nullptr;
        cudaEvent_t ev_conv_done = nullptr;
        cudaEvent_t ev_back_done = nullptr;
        if (lowp_runtime_possible) {
            cudaStreamCreateWithFlags(&stream_conv, cudaStreamNonBlocking);
            cudaEventCreate(&ev_conv_done);
            cudaEventCreate(&ev_back_done);
        }
        // 小 k/rank 场景下，辅助流/事件的固定开销常常比收益更大。
        const bool small_aux_overhead_case = (qb_block_k <= 64 && k <= 64 && real_rank <= 128);
        bool aux_streams_enable = !small_aux_overhead_case;
        if (const char* e = std::getenv("HQB_AUX_STREAMS")) { aux_streams_enable = (e[0] != '0'); }
        if (debug_mode && !aux_streams_enable) {
            printf("[Info] Small-k path: disabling pack/compute auxiliary streams to reduce fixed overhead.\n");
        }
        auto pick_tile_cols = [](){
            int tile = 4096; if (const char* e = std::getenv("HQB_TILE")) { int v = std::atoi(e); if (v >= 256) tile = v; }
            return tile;
        };
        const int base_tile_cols = pick_tile_cols();
        auto read_parallel_threshold = [](const char* name, int default_value) {
            if (const char* s = std::getenv(name)) {
                int v = std::atoi(s);
                if (v > 0) return v;
            }
            return default_value;
        };
        const int par_min_chunks = read_parallel_threshold("HQB_PAR_MIN_CHUNKS", 2);
        const int par_min_trailing_cols = read_parallel_threshold("HQB_PAR_MIN_NT", std::max(2048, base_tile_cols));
        const long int default_panel_width_for_sched = std::max<long int>(1, std::min<long int>(qb_block_k, std::min<long int>(n, real_rank)));
        const long int max_sched_trailing_cols = std::max<long int>(0, n - default_panel_width_for_sched);
        const int sched_tile_cols = std::max(1, std::min<int>((int)max_sched_trailing_cols, base_tile_cols));
        const int sched_chunk_count = sched_tile_cols > 0 ? (int)((max_sched_trailing_cols + sched_tile_cols - 1) / sched_tile_cols) : 0;
        const bool pack_streams_default =
            aux_streams_enable &&
            (num_blocks > 2) &&
            (real_rank > 128) &&
            (max_sched_trailing_cols >= par_min_trailing_cols);
        // W/Y 汇入全局缓冲的拷贝流（彼此独立），以及“最后一次拷贝完成”事件
        bool use_pack_streams = pack_streams_default;
        if (const char* e = std::getenv("HQB_PACK_STREAMS")) { use_pack_streams = (e[0] != '0'); }
        cudaStream_t stream_pack_w = nullptr;
        cudaStream_t stream_pack_y = nullptr;
        cudaEvent_t ev_pack_w_last = nullptr;
        cudaEvent_t ev_pack_y_last = nullptr;
        if (use_pack_streams) {
            cudaStreamCreateWithFlags(&stream_pack_w, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&stream_pack_y, cudaStreamNonBlocking);
            cudaEventCreateWithFlags(&ev_pack_w_last, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&ev_pack_y_last, cudaEventDisableTiming);
        }
        // 安全版多流：创建计算流池与句柄(HQB_PAR=0对比基线单流，默认HQB_PAR=1)
        const bool par_default =
            aux_streams_enable &&
            (num_blocks > 2) &&
            (real_rank > 128) &&
            (max_sched_trailing_cols >= par_min_trailing_cols) &&
            (sched_chunk_count >= par_min_chunks);
        bool par_enable = par_default;
        if (const char* e = std::getenv("HQB_PAR")) { par_enable = (e[0] != '0'); }
        // 面板数据就绪事件：仅在确实存在跨流依赖时才需要。
        const bool need_panel_ready_event = use_pack_streams || par_enable || lowp_runtime_possible;
        cudaEvent_t ev_panel_ready = nullptr;
        if (need_panel_ready_event) {
            cudaEventCreateWithFlags(&ev_panel_ready, cudaEventDisableTiming);
        }
        int num_compute_streams = 3; // 建议 2~4，可通过 HQB_STREAMS 调整
        if (const char* e = std::getenv("HQB_STREAMS")) {
            int v = std::atoi(e); if (v >= 1 && v <= 8) num_compute_streams = v;
        }
        std::vector<cudaStream_t> compute_streams;
        std::vector<cublasHandle_t> compute_handles;
        std::vector<cudaEvent_t> ev_compute_done;
        if (par_enable) {
            compute_streams.resize(num_compute_streams);
            compute_handles.resize(num_compute_streams);
            ev_compute_done.resize(num_compute_streams);
            cublasMath_t base_mode = CUBLAS_DEFAULT_MATH; cublasGetMathMode(cublasHandle, &base_mode);
            for (int i = 0; i < num_compute_streams; ++i) {
                cudaStreamCreateWithFlags(&compute_streams[i], cudaStreamNonBlocking);
                cublasCreate(&compute_handles[i]);
                cublasSetStream(compute_handles[i], compute_streams[i]);
                cublasSetMathMode(compute_handles[i], base_mode);
                cudaEventCreateWithFlags(&ev_compute_done[i], cudaEventDisableTiming);
            }
        }
        const size_t tile_cols_cap = (size_t)std::max(1, std::min((int)n, std::min(pick_tile_cols(), 2048)));
        const size_t max_panel_elems = (size_t)m * (size_t)k_max;
        const size_t max_square_panel_elems = (size_t)k_max * (size_t)k_max;
        const size_t max_temp1_elems = (size_t)k_max * (size_t)n;
        const size_t max_double_tile_elems = (size_t)m * tile_cols_cap;
        const size_t max_double_temp1_elems = (size_t)k_max * tile_cols_cap;
        const size_t max_padded_panel_height = ((size_t)m + 15u) / 16u * 16u;

        // Step3 热路径缓冲循环前预分配，避免小 k 场景反复 resize 触发 allocator 抖动。
        buf_Q_filtered.resize(max_panel_elems);
        buf_R_temp.resize(max_square_panel_elems);
        buf_B.resize(max_panel_elems);
        buf_W.resize(max_panel_elems);
        buf_Y.resize(max_panel_elems);
        buf_Temp1_fp32.resize(max_temp1_elems);
        thrust::device_vector<int> d_keep_indices_buf((size_t)k_max);
        thrust::device_vector<T_compute> d_R_filtered_buf(max_square_panel_elems);
        int lwork_getrf_panel_max = 0;
        SolverCaller<T_compute>::getrf_bufferSize(cusolverHandle, (int)m, k_max, d_A_ptr, (int)m, &lwork_getrf_panel_max);
        buf_getrf_work.resize((size_t)lwork_getrf_panel_max);
        if constexpr (std::is_same_v<T_storage, float> && std::is_same_v<T_compute, float>) {
            if (!use_ppopp_float_path) {
                buf_Q_filtered_double.resize(max_panel_elems);
                buf_B_double.resize(max_panel_elems);
                buf_W_double.resize(max_panel_elems);
                buf_Y_double.resize(max_panel_elems);
                buf_A_trailing_double.resize(max_double_tile_elems);
                buf_Temp1_double.resize(max_double_temp1_elems);
                int lwork_getrf_double_max = 0;
                SolverCaller<double>::getrf_bufferSize(cusolverHandle, (int)m, k_max,
                                                       d_W_final_accum_double_ptr, (int)m, &lwork_getrf_double_max);
                buf_getrf_work_double.resize((size_t)lwork_getrf_double_max);
            }
        }
        if constexpr ((std::is_same_v<T_storage, __half> || std::is_same_v<T_storage, __nv_bfloat16>) && std::is_same_v<T_compute, float>) {
            if (lowp_runtime_possible) {
                buf_W_lowp.reserve(max_padded_panel_height * (size_t)k_max);
                buf_Y_lowp.reserve(max_padded_panel_height * (size_t)k_max);
                buf_A_trailing_lowp.reserve(max_padded_panel_height * (size_t)n);
                buf_Temp1_lowp.reserve((size_t)k_max * (size_t)n);
            }
        }
        auto use_parallel_trailing = [&](long int trailing_cols, int tile_cols) {
            if (!par_enable || compute_streams.empty()) return false;
            if (trailing_cols < (long int)par_min_trailing_cols) return false;
            const int n_chunks = (int)((trailing_cols + tile_cols - 1) / tile_cols);
            return n_chunks >= par_min_chunks;
        };
         startTimer();
            for (size_t j = 0; j < num_blocks; ++j)
            {
                long int col_start = j * qb_block_k;
                long int row_start = j * qb_block_k;
                // 计算当前面板的维度
                long int panel_height = m - row_start;
                long int panel_width = std::min(qb_block_k, n - col_start);
                // 确保不超过实际的矩阵秩
                panel_width = std::min(panel_width, (long int)(real_rank - accumulated_rank));
                if (panel_height <= 0 || panel_width <= 0) break;
                T_compute* d_A_panel_ptr = d_A_ptr + (size_t)col_start * m + row_start;
                
                // TSQR / cuSOLVER QR 进行panel QR
                T_compute* d_R_tsqr_ptr = thrust::raw_pointer_cast(d_R_tsqr.data());
                bool tsqr_shape_ok = (panel_height >= panel_width);
                // 除共享内存阈值外，再约束面板宽度不超过固定阈值
                // 默认启用 TSQR；可通过 HQB_DISABLE_TSQR=1 关闭。
                bool tsqr_env_ok = true;
                if (const char* en = std::getenv("HQB_ENABLE_TSQR")) { tsqr_env_ok = (en[0] != '0'); }
                if (const char* dis = std::getenv("HQB_DISABLE_TSQR")) { if (dis[0] != '0') tsqr_env_ok = false; }
                long int tsqr_hw_cap = 64; // 按需设置阈值上限，暂时64是安全阈值
                // 通过 HQB_TSQR_MAX_N 进一步收紧阈值；默认与硬上限一致
                long int tsqr_user_cap = tsqr_hw_cap;
                if (const char* cap = std::getenv("HQB_TSQR_MAX_N")) {
                    long v = std::atol(cap); if (v > 0) tsqr_user_cap = std::min<long>(tsqr_hw_cap, v);
                }
                bool use_tsqr = tsqr_env_ok && tsqr_shape_ok && (panel_width <= tsqr_user_cap) && (panel_width <= tsqr_hw_cap);
                if (use_tsqr) {
                    if (!printed_path_info) {
                        printf("[Info] Using HH+MGS TSQR for panels (panel_width<=%ld and panel_height>=panel_width). First panel: %ld.\n",
                               tsqr_hw_cap, panel_width);
                        printed_path_info = true;
                    }
                    // Householder + MGS TSQR
                    if constexpr (std::is_same_v<T_compute, float>) {
                        // float 路径下按开关决定 TSQR 是否提升到 double
                        bool tsqr_in_double = qb_panel_tsqr_in_double;
                        if (tsqr_in_double) {
                            // double-TSQR 对共享内存要求更高；超限时回退到 float-TSQR，避免 invalid argument
                            int smem_optin = 0, smem_max = 0;
                            cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
                            cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlock, 0);
                            int smem_cap = std::max(smem_optin, smem_max);
                            const long long hh_need  = (long long)TSQR_BLOCK_SIZE * (long long)panel_width * (long long)sizeof(double);
                            const long long mgs_need = ((long long)TSQR_BLOCK_SIZE * (long long)panel_width +
                                                        (long long)panel_width * (long long)panel_width) * (long long)sizeof(double);
                            if (hh_need > smem_cap || mgs_need > smem_cap) {
                                printf("[Info] Double-TSQR disabled by shared-memory cap (need hh=%lld, mgs=%lld, cap=%d). Fallback to float TSQR.\n",
                                       hh_need, mgs_need, smem_cap);
                                tsqr_in_double = false;
                            }
                        }
                        if (tsqr_in_double) {
                            thrust::device_vector<double> d_A_panel_d((size_t)panel_height * (size_t)panel_width);
                            double* d_A_panel_d_ptr = thrust::raw_pointer_cast(d_A_panel_d.data());
                            dim3 tcv(16,16);
                            dim3 bcv((int)((panel_width + tcv.x - 1)/tcv.x), (int)((panel_height + tcv.y - 1)/tcv.y));
                            convert_kernel_strided<T_compute, double><<<bcv, tcv, 0, stream>>>(
                                d_A_panel_ptr, (size_t)m,
                                d_A_panel_d_ptr, (size_t)panel_height,
                                (size_t)panel_height, (size_t)panel_width);

                            int tsqr_block_num = ((int)panel_height + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;
                            int tsqr_ldwork = tsqr_block_num * (int)panel_width;
                            thrust::device_vector<double> d_work_tsqr_d((size_t)tsqr_ldwork * (size_t)panel_width);
                            thrust::device_vector<double> d_R_tsqr_d((size_t)panel_width * (size_t)panel_width);
                            double* d_R_tsqr_d_ptr = thrust::raw_pointer_cast(d_R_tsqr_d.data());
                            tsqr_hh_mgs<double>(cublasHandle,
                                                (int)panel_height, (int)panel_width,
                                                d_A_panel_d_ptr, (int)panel_height,
                                                d_R_tsqr_d_ptr, (int)panel_width,
                                                thrust::raw_pointer_cast(d_work_tsqr_d.data()),
                                                tsqr_ldwork);
                            CUDA_CHECK(cudaGetLastError());
                            // 回写 Q 与 R 到 float 主路径
                            convert_kernel_strided<double, T_compute><<<bcv, tcv, 0, stream>>>(
                                d_A_panel_d_ptr, (size_t)panel_height,
                                d_A_panel_ptr, (size_t)m,
                                (size_t)panel_height, (size_t)panel_width);
                            convert_kernel<double, T_compute><<<(int)((((size_t)panel_width*(size_t)panel_width)+255)/256), 256, 0, stream>>>(
                                d_R_tsqr_d_ptr, d_R_tsqr_ptr, (size_t)panel_width * (size_t)panel_width);
                        } else {
                            int tsqr_block_num = ((int)panel_height + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;
                            int tsqr_ldwork = tsqr_block_num * (int)panel_width;
                            tsqr_hh_mgs<T_compute>(cublasHandle,
                                                   (int)panel_height, (int)panel_width,
                                                   d_A_panel_ptr, (int)m,
                                                   d_R_tsqr_ptr, (int)panel_width,
                                                   thrust::raw_pointer_cast(d_work_tsqr.data()),
                                                   tsqr_ldwork);
                            CUDA_CHECK(cudaGetLastError());
                        }
                    } else if constexpr (std::is_same_v<T_compute, double>) {
                        int tsqr_block_num = ((int)panel_height + TSQR_BLOCK_SIZE - 1) / TSQR_BLOCK_SIZE;
                        int tsqr_ldwork = tsqr_block_num * (int)panel_width;
                        tsqr_hh_mgs<T_compute>(cublasHandle,
                                               (int)panel_height, (int)panel_width,
                                               d_A_panel_ptr, (int)m,
                                               d_R_tsqr_ptr, (int)panel_width,
                                               thrust::raw_pointer_cast(d_work_tsqr.data()),
                                               tsqr_ldwork);
                        CUDA_CHECK(cudaGetLastError());
                    } else {
                        cudaMemset(thrust::raw_pointer_cast(d_work_tsqr.data()), 0, d_work_tsqr.size() * sizeof(T_compute));
                        tsqr<T_compute>(cublasHandle,
                                        (int)panel_height, (int)panel_width,
                                        d_A_panel_ptr, (int)m,
                                        d_R_tsqr_ptr, (int)panel_width,
                                        thrust::raw_pointer_cast(d_work_tsqr.data()), (int)m);
                    }
                } else {
                    if (!printed_path_info) {
                        if (!tsqr_shape_ok) {
                            printf("[Info] Using cuSOLVER QR for panels (reason: panel_height < panel_width). First panel: %ld.\n",
                                   panel_width);
                        } else if (panel_width > tsqr_user_cap) {
                            printf("[Info] Using cuSOLVER QR for panels (reason: panel_width > user cap %ld). First panel: %ld.\n",
                                   tsqr_user_cap, panel_width);
                        } else if (panel_width > tsqr_hw_cap) {
                            printf("[Info] Using cuSOLVER QR for panels (reason: panel_width > hw cap %ld). First panel: %ld.\n",
                                   tsqr_hw_cap, panel_width);
                        } else {
                            printf("[Info] Using cuSOLVER QR for panels (fallback condition). First panel: %ld.\n", panel_width);
                        }
                        printed_path_info = true;
                    }
                    if constexpr (std::is_same_v<T_compute,float>) {
                        bool panel_qr_in_double = (g_accuracy == AccuracyLevel::High);
                        if (panel_qr_in_double) {
                            thrust::device_vector<double> d_A_panel_d((size_t)panel_height * (size_t)panel_width);
                            double* d_A_panel_d_ptr = thrust::raw_pointer_cast(d_A_panel_d.data());
                            // Convert panel (ld=m) -> compact double (ld=panel_height)
                            dim3 tcv(16,16);
                            dim3 bcv((int)((panel_width + tcv.x - 1)/tcv.x), (int)((panel_height + tcv.y - 1)/tcv.y));
                            convert_kernel_strided<T_compute, double><<<bcv, tcv, 0, stream>>>(
                                d_A_panel_ptr, (size_t)m,
                                d_A_panel_d_ptr, (size_t)panel_height,
                                (size_t)panel_height, (size_t)panel_width);
                            // Double QR on compact panel
                            int lwork_geqrf_d=0; SolverCaller<double>::geqrf_bufferSize(cusolverHandle, (int)panel_height, (int)panel_width,
                                                                                         d_A_panel_d_ptr, (int)panel_height, &lwork_geqrf_d);
                            thrust::device_vector<double> d_work_geqrf_d(lwork_geqrf_d);
                            thrust::device_vector<double> d_tau_d((size_t)panel_width);
                            SolverCaller<double>::geqrf(cusolverHandle, (int)panel_height, (int)panel_width,
                                                        d_A_panel_d_ptr, (int)panel_height,
                                                        thrust::raw_pointer_cast(d_tau_d.data()),
                                                        thrust::raw_pointer_cast(d_work_geqrf_d.data()), lwork_geqrf_d,
                                                        thrust::raw_pointer_cast(d_devInfo.data()));
                            // Extract R in double then convert to float R buffer
                            thrust::device_vector<double> d_R_tsqr_d((size_t)panel_width * (size_t)panel_width);
                            double* d_R_tsqr_d_ptr = thrust::raw_pointer_cast(d_R_tsqr_d.data());
                            dim3 thRe(16,16); dim3 grRe((int)((panel_width + 15)/16), (int)((panel_width + 15)/16));
                            extract_upper_triangle_kernel<double><<<grRe, thRe, 0, stream>>>(
                                d_R_tsqr_d_ptr, (int)panel_width, d_A_panel_d_ptr, (int)panel_height, (int)panel_width, (int)panel_width);
                            // Build Q in double
                            int lwork_orgqr_d=0; SolverCaller<double>::orgqr_bufferSize(cusolverHandle, (int)panel_height, (int)panel_width, (int)panel_width,
                                                                                          d_A_panel_d_ptr, (int)panel_height,
                                                                                          thrust::raw_pointer_cast(d_tau_d.data()), &lwork_orgqr_d);
                            thrust::device_vector<double> d_work_orgqr_d(lwork_orgqr_d);
                            SolverCaller<double>::orgqr(cusolverHandle, (int)panel_height, (int)panel_width, (int)panel_width,
                                                        d_A_panel_d_ptr, (int)panel_height,
                                                        thrust::raw_pointer_cast(d_tau_d.data()),
                                                        thrust::raw_pointer_cast(d_work_orgqr_d.data()), lwork_orgqr_d,
                                                        thrust::raw_pointer_cast(d_devInfo.data()));
                            // Convert Q back to float into original panel (ld=m)
                            convert_kernel_strided<double, T_compute><<<bcv, tcv, 0, stream>>>(
                                d_A_panel_d_ptr, (size_t)panel_height,
                                d_A_panel_ptr, (size_t)m,
                                (size_t)panel_height, (size_t)panel_width);
                            // Convert R to float buffer (column-major panel_width x panel_width)
                            int nconv = (int)((((size_t)panel_width*(size_t)panel_width)+255)/256);
                            convert_kernel<double, T_compute><<<nconv, 256, 0, stream>>>(
                                d_R_tsqr_d_ptr, d_R_tsqr_ptr, (size_t)panel_width * (size_t)panel_width);
                        } else {
                            // Original float QR
                            SolverCaller<T_compute>::geqrf(cusolverHandle, (int)panel_height, (int)panel_width,
                                                            d_A_panel_ptr, (int)m,
                                                            thrust::raw_pointer_cast(d_tau.data()),
                                                            thrust::raw_pointer_cast(d_work_qr.data()), (int)lwork_max_qr,
                                                            thrust::raw_pointer_cast(d_devInfo.data()));
                            dim3 threads_R_extract(16, 16);
                            dim3 blocks_R_extract(((int)panel_width + 15) / 16, ((int)panel_width + 15) / 16);
                            extract_upper_triangle_kernel<T_compute><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
                                d_R_tsqr_ptr, (int)panel_width, d_A_panel_ptr, (int)m, (int)panel_width, (int)panel_width);
                            SolverCaller<T_compute>::orgqr(cusolverHandle, (int)panel_height, (int)panel_width, (int)panel_width,
                                                           d_A_panel_ptr, (int)m,
                                                           thrust::raw_pointer_cast(d_tau.data()),
                                                           thrust::raw_pointer_cast(d_work_qr.data()), (int)lwork_max_qr,
                                                           thrust::raw_pointer_cast(d_devInfo.data()));
                        }
                    } else {
                        // Double compute path unchanged
                        SolverCaller<T_compute>::geqrf(cusolverHandle, (int)panel_height, (int)panel_width,
                                                        d_A_panel_ptr, (int)m,
                                                        thrust::raw_pointer_cast(d_tau.data()),
                                                        thrust::raw_pointer_cast(d_work_qr.data()), (int)lwork_max_qr,
                                                        thrust::raw_pointer_cast(d_devInfo.data()));
                        dim3 threads_R_extract(16, 16);
                        dim3 blocks_R_extract(((int)panel_width + 15) / 16, ((int)panel_width + 15) / 16);
                        extract_upper_triangle_kernel<T_compute><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
                            d_R_tsqr_ptr, (int)panel_width, d_A_panel_ptr, (int)m, (int)panel_width, (int)panel_width);
                        SolverCaller<T_compute>::orgqr(cusolverHandle, (int)panel_height, (int)panel_width, (int)panel_width,
                                                       d_A_panel_ptr, (int)m,
                                                       thrust::raw_pointer_cast(d_tau.data()),
                                                       thrust::raw_pointer_cast(d_work_qr.data()), (int)lwork_max_qr,
                                                       thrust::raw_pointer_cast(d_devInfo.data()));
                    }
                }

                long int max_selectable = std::min<long int>(panel_width, (long int)(real_rank - accumulated_rank));

                T_compute* diag_ptr = thrust::raw_pointer_cast(row_tail_norms_gpu_buf.data());
                extractDiagonal<<<(panel_width + 255) / 256, 256, 0, stream>>>(
                    diag_ptr,
                    d_R_tsqr_ptr,
                    (int)panel_width,
                    0,
                    (int)panel_width);
                auto diag_begin = row_tail_norms_gpu_buf.begin();
                auto diag_end = diag_begin + panel_width;
                double max_diag = thrust::transform_reduce(
                    thrust::cuda::par.on(stream),
                    diag_begin,
                    diag_end,
                    abs_as_double_op<T_compute>{},
                    0.0,
                    thrust::maximum<double>());
                double relative_eps = max_diag * (double)eps;
                long int kept = 0;
                long int tail_kept = panel_width;
                if (g_unknown_rank_mode) {
                    std::vector<T_compute> h_diag((size_t)panel_width);
                    thrust::copy(diag_begin, diag_end, h_diag.begin());
                    kept = panel_width;
                    for (long int ii = 0; ii < panel_width; ++ii) {
                        if (std::abs((double)h_diag[(size_t)ii]) < relative_eps) {
                            kept = ii;
                            break;
                        }
                    }
                    if (g_tail_criterion_enabled && panel_width > 0) {
                        upper_row_tail_norm_sq_kernel<T_compute><<<((int)panel_width + 255) / 256, 256, 0, stream>>>(
                            thrust::raw_pointer_cast(row_tail_norms_sq_double_buf.data()),
                            d_R_tsqr_ptr, (int)panel_width, (int)panel_width);
                        CUDA_CHECK(cudaGetLastError());
                        std::vector<double> h_row_tail_sq((size_t)panel_width);
                        thrust::copy(row_tail_norms_sq_double_buf.begin(),
                                     row_tail_norms_sq_double_buf.begin() + panel_width,
                                     h_row_tail_sq.begin());
                        const double base_tail_sq = h_row_tail_sq.empty() ? 0.0 : h_row_tail_sq[0];
                        const double tail_rel_eps = (g_tail_norm_eps > 0.0) ? g_tail_norm_eps : std::max(1e-12, (double)eps);
                        const double tail_threshold_sq = base_tail_sq * tail_rel_eps * tail_rel_eps;
                        for (long int ii = 1; ii < panel_width; ++ii) {
                            if (h_row_tail_sq[(size_t)ii] <= tail_threshold_sq) {
                                tail_kept = ii;
                                break;
                            }
                        }
                        kept = std::min<long int>(kept, tail_kept);
                    }
                    kept = std::max<long int>(0, std::min<long int>(kept, panel_width));
                    if (kept > 0) {
                        thrust::sequence(thrust::cuda::par.on(stream),
                                         d_keep_indices_buf.begin(),
                                         d_keep_indices_buf.begin() + kept,
                                         0);
                    }
                } else {
                    auto keep_end = thrust::copy_if(
                        thrust::cuda::par.on(stream),
                        thrust::make_counting_iterator<int>(0),
                        thrust::make_counting_iterator<int>((int)panel_width),
                        diag_begin,
                        d_keep_indices_buf.begin(),
                        keep_if_ge_relative_eps_op<T_compute>(relative_eps));
                    kept = (long int)(keep_end - d_keep_indices_buf.begin());
                }
                long int desired_panel_width = std::min<long int>(panel_width, max_selectable);
                long int new_panel_width = std::min<long int>(kept, desired_panel_width);
                if (g_force_full_rank && desired_panel_width > 0) {
                    if (!panel_rank_drop_warned && kept < desired_panel_width) {
                        printf("[Info] QB truncation would keep %ld/%ld columns at eps=%e, but ForceFullRank is on; keeping the full panel.\n",
                               kept, desired_panel_width, (double)eps);
                        panel_rank_drop_warned = true;
                    }
                    thrust::sequence(thrust::cuda::par.on(stream),
                                     d_keep_indices_buf.begin(),
                                     d_keep_indices_buf.begin() + desired_panel_width,
                                     0);
                    new_panel_width = desired_panel_width;
                } else if (g_unknown_rank_mode && debug_mode) {
                    printf("[AdaptiveRank] panel %zu kept=%ld (diag=%ld, tail=%ld, panel=%ld).\n",
                           j, new_panel_width, kept, tail_kept, panel_width);
                }
                if (new_panel_width <= 0) {
                    if (g_unknown_rank_mode) {
                        stop_condition_met = true;
                        final_rank = accumulated_rank;
                        printf("[AdaptiveRank] Stop: panel %zu contributes no new columns (accumulated_rank=%zu).\n",
                               j, accumulated_rank);
                        break;
                    }
                    continue;
                }
                int* d_keep_indices_ptr = thrust::raw_pointer_cast(d_keep_indices_buf.data());

                T_compute* d_Q_filtered_ptr = thrust::raw_pointer_cast(buf_Q_filtered.data());
                dim3 threads_gather(16, 16);
                dim3 blocks_gather((new_panel_width + 15) / 16, (panel_height + 15) / 16);
                select_columns_kernel<T_compute><<<blocks_gather, threads_gather, 0, stream>>>(d_Q_filtered_ptr, panel_height, d_A_ptr + col_start * m + row_start, m,
                d_keep_indices_ptr, panel_height, new_panel_width);
                // 筛选列后的Q_filtered，然后存入最终的Q_final
                T_compute* d_Q_final_dest_ptr = d_Q_final_ptr + (size_t)accumulated_rank * m + row_start;
                cudaMemcpy2DAsync(d_Q_final_dest_ptr, m * sizeof(T_compute), d_Q_filtered_ptr, panel_height * sizeof(T_compute), panel_height * sizeof(T_compute), new_panel_width, cudaMemcpyDeviceToDevice, stream);
                // 筛选列后的R_temp，然后存入最终的R_final
                T_compute* d_R_temp_ptr = thrust::raw_pointer_cast(buf_R_temp.data());
                dim3 threads_gather_R(16, 16);
                dim3 blocks_gather_R((new_panel_width + 15) / 16, (panel_width + 15) / 16);
                select_columns_kernel<T_compute><<<blocks_gather_R, threads_gather_R, 0, stream>>>(d_R_temp_ptr, panel_width, d_R_tsqr_ptr, panel_width, d_keep_indices_ptr, panel_width,new_panel_width);

                // 对 R 再按相同索引筛行，得到与 PPOPP 路径一致的方阵 R11(new_panel_width x new_panel_width)
                T_compute* d_R_filtered_ptr = thrust::raw_pointer_cast(d_R_filtered_buf.data());
                dim3 threads_gather_R_rows(16, 16);
                dim3 blocks_gather_R_rows((new_panel_width + 15) / 16, (new_panel_width + 15) / 16);
                select_rows_kernel<T_compute><<<blocks_gather_R_rows, threads_gather_R_rows, 0, stream>>>(
                    d_R_filtered_ptr, (int)new_panel_width,
                    d_R_temp_ptr, (int)panel_width,
                    d_keep_indices_ptr,
                    (int)new_panel_width, (int)new_panel_width);

                if (!guaranteed_skip_final_qr) {
                    T_compute* d_R_dest_ptr = d_R_final_ptr + (size_t)accumulated_rank * n + accumulated_rank;
                    T_compute* d_R_src_ptr = d_R_filtered_ptr;
                    cudaMemcpy2DAsync(d_R_dest_ptr,
                                      n * sizeof(T_compute),
                                      d_R_src_ptr,
                                      new_panel_width * sizeof(T_compute),
                                      new_panel_width * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice,
                                      stream);
                }
                
                // 旧的 direct-projection fast path 在多 panel 下不能构造正确的全局 Q_basis，这里禁用。
                if (false && fast_mode && use_tsqr) {
                    long int trailing_matrix_cols = n - (col_start + panel_width);
                    if (trailing_matrix_cols > 0) {
                        T_compute* d_A_trailing_ptr = d_A_ptr + (size_t)(col_start + panel_width) * m + row_start;
                        thrust::device_vector<T_compute> d_R_top((size_t)new_panel_width * (size_t)trailing_matrix_cols);
                        T_compute* d_R_top_ptr = thrust::raw_pointer_cast(d_R_top.data());
                        // R_top = Q_filtered^T * A_trailing
                        BlasCaller<T_compute>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                    (int)new_panel_width, (int)trailing_matrix_cols, (int)panel_height,
                                                    &one,
                                                    d_Q_filtered_ptr, (int)panel_height,
                                                    d_A_trailing_ptr, (int)m,
                                                    &zero,
                                                    d_R_top_ptr, (int)new_panel_width);
                        // A_trailing -= Q_filtered * R_top
                        BlasCaller<T_compute>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)panel_height, (int)trailing_matrix_cols, (int)new_panel_width,
                                                    &minus_one,
                                                    d_Q_filtered_ptr, (int)panel_height,
                                                    d_R_top_ptr, (int)new_panel_width,
                                                    &one,
                                                    d_A_trailing_ptr, (int)m);
                        // 更新 R 的非对角块
                        if (!guaranteed_skip_final_qr) {
                            T_compute* d_R_offdiag_dest_ptr = d_R_final_ptr + (size_t)(col_start + panel_width) * n + accumulated_rank;
                            cudaMemcpy2DAsync(d_R_offdiag_dest_ptr,
                                              n * sizeof(T_compute),
                                              d_R_top_ptr,
                                              new_panel_width * sizeof(T_compute),
                                              new_panel_width * sizeof(T_compute),
                                              trailing_matrix_cols,
                                              cudaMemcpyDeviceToDevice, stream);
                        }
                    }
                } else {
                if constexpr (std::is_same_v<T_storage, float> && std::is_same_v<T_compute, float>) {
                if (!use_ppopp_float_path && !fast_mode) {
                // 默认 float 精度路径：保留 double W/Y 构造与累积，但 panel trailing update 回到 float，
                // 只把真正影响全局 Q_basis 稳定性的部分留在 double，避免每个 tile 做 double 往返。
                static bool s_step3_float_mixed_printed = false;
                if (debug_mode && !s_step3_float_mixed_printed) {
                    printf("[Info] Step3 float path: building panel W/Y in double, trailing update in float.\n");
                    s_step3_float_mixed_printed = true;
                }
                buf_Q_filtered_double.resize((size_t)panel_height * (size_t)new_panel_width);
                double* d_Q_filtered_double_ptr = thrust::raw_pointer_cast(buf_Q_filtered_double.data());
                convert_kernel<float, double><<<(int)((((size_t)panel_height * (size_t)new_panel_width) + 255) / 256), 256, 0, stream>>>(
                    d_Q_filtered_ptr, d_Q_filtered_double_ptr, (size_t)panel_height * (size_t)new_panel_width);

                buf_B_double.resize((size_t)panel_height * (size_t)new_panel_width);
                double* d_B_double_ptr = thrust::raw_pointer_cast(buf_B_double.data());
                BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         (int)panel_height, (int)new_panel_width,
                                         &minus_one_d, d_Q_filtered_double_ptr, (int)panel_height,
                                         &zero_d, d_Q_filtered_double_ptr, (int)panel_height,
                                         d_B_double_ptr, (int)panel_height);
                dim3 threads_add_id(256);
                dim3 blocks_add_id((std::min(panel_height, new_panel_width) + threads_add_id.x - 1) / threads_add_id.x);
                add_identity_diagonal_kernel<double><<<blocks_add_id, threads_add_id, 0, stream>>>(
                    d_B_double_ptr, panel_height, new_panel_width, panel_height, 1.0);

                buf_W_double.resize((size_t)panel_height * (size_t)new_panel_width);
                double* d_W_double_ptr = thrust::raw_pointer_cast(buf_W_double.data());
                cudaMemcpyAsync(d_W_double_ptr, d_B_double_ptr,
                                (size_t)panel_height * (size_t)new_panel_width * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);

                buf_Y_double.resize((size_t)panel_height * (size_t)new_panel_width);
                double* d_Y_double_ptr = thrust::raw_pointer_cast(buf_Y_double.data());
                int lwork_getrf_double = 0;
                SolverCaller<double>::getrf_bufferSize(cusolverHandle, (int)panel_height, (int)new_panel_width,
                                                       d_B_double_ptr, (int)panel_height, &lwork_getrf_double);
                buf_getrf_work_double.resize((size_t)lwork_getrf_double);
                double* d_getrf_work_double_ptr = thrust::raw_pointer_cast(buf_getrf_work_double.data());
                int* d_devInfo_ptr = thrust::raw_pointer_cast(d_devInfo.data());
                SolverCaller<double>::getrf(cusolverHandle, (int)panel_height, (int)new_panel_width,
                                            d_B_double_ptr, (int)panel_height,
                                            d_getrf_work_double_ptr, NULL, d_devInfo_ptr);
                cudaMemsetAsync(d_Y_double_ptr, 0,
                                (size_t)panel_height * (size_t)new_panel_width * sizeof(double), stream);
                dim3 threads_for_L(16, 16);
                dim3 blocks_for_L((new_panel_width + threads_for_L.x - 1) / threads_for_L.x,
                                  (panel_height + threads_for_L.y - 1) / threads_for_L.y);
                extract_L_factor_kernel<double><<<blocks_for_L, threads_for_L, 0, stream>>>(
                    d_Y_double_ptr, panel_height, d_B_double_ptr, panel_height, panel_height, new_panel_width);
                BlasCaller<double>::trsm(cublasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                                         CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                         (int)panel_height, (int)new_panel_width,
                                         &one_d, d_Y_double_ptr, (int)panel_height,
                                         d_W_double_ptr, (int)panel_height);

                buf_W.resize((size_t)panel_height * (size_t)new_panel_width);
                T_compute* d_W_ptr = thrust::raw_pointer_cast(buf_W.data());
                buf_Y.resize((size_t)panel_height * (size_t)new_panel_width);
                T_compute* d_Y_ptr = thrust::raw_pointer_cast(buf_Y.data());
                convert_kernel<double, float><<<(int)((((size_t)panel_height * (size_t)new_panel_width) + 255) / 256), 256, 0, stream>>>(
                    d_W_double_ptr, d_W_ptr, (size_t)panel_height * (size_t)new_panel_width);
                convert_kernel<double, float><<<(int)((((size_t)panel_height * (size_t)new_panel_width) + 255) / 256), 256, 0, stream>>>(
                    d_Y_double_ptr, d_Y_ptr, (size_t)panel_height * (size_t)new_panel_width);

                if (need_panel_ready_event) {
                    cudaEventRecord(ev_panel_ready, stream);
                }

                T_compute* d_W_final_dest_ptr = d_W_final_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaStreamWaitEvent(stream_pack_w, ev_panel_ready, 0);
                    cudaMemcpy2DAsync(d_W_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_W_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_w);
                } else {
                    cudaMemcpy2DAsync(d_W_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_W_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }

                double* d_W_final_dest_double_ptr = d_W_final_accum_double_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaMemcpy2DAsync(d_W_final_dest_double_ptr + row_start,
                                      m * sizeof(double),
                                      d_W_double_ptr,
                                      panel_height * sizeof(double),
                                      panel_height * sizeof(double),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_w);
                    cudaEventRecord(ev_pack_w_last, stream_pack_w);
                } else {
                    cudaMemcpy2DAsync(d_W_final_dest_double_ptr + row_start,
                                      m * sizeof(double),
                                      d_W_double_ptr,
                                      panel_height * sizeof(double),
                                      panel_height * sizeof(double),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }

                T_compute* d_Y_final_dest_ptr = d_Y_final_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaStreamWaitEvent(stream_pack_y, ev_panel_ready, 0);
                    cudaMemcpy2DAsync(d_Y_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_Y_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_y);
                } else {
                    cudaMemcpy2DAsync(d_Y_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_Y_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }

                double* d_Y_final_dest_double_ptr = d_Y_final_accum_double_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaMemcpy2DAsync(d_Y_final_dest_double_ptr + row_start,
                                      m * sizeof(double),
                                      d_Y_double_ptr,
                                      panel_height * sizeof(double),
                                      panel_height * sizeof(double),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_y);
                    cudaEventRecord(ev_pack_y_last, stream_pack_y);
                } else {
                    cudaMemcpy2DAsync(d_Y_final_dest_double_ptr + row_start,
                                      m * sizeof(double),
                                      d_Y_double_ptr,
                                      panel_height * sizeof(double),
                                      panel_height * sizeof(double),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }

                long int trailing_matrix_cols = n - (col_start + panel_width);
                if (trailing_matrix_cols > 0) {
                    T_compute* d_A_trailing_ptr = d_A_ptr + (size_t)(col_start + panel_width) * m + row_start;
                    T_compute* d_Temp1_ptr = thrust::raw_pointer_cast(buf_Temp1_fp32.data());
                    gemm_pair_graph_or_plain<T_compute>(cublasHandle, stream,
                                                        (int)panel_height, (int)trailing_matrix_cols, (int)new_panel_width,
                                                        d_W_ptr, (int)panel_height,
                                                        d_A_trailing_ptr, (int)m,
                                                        d_Temp1_ptr, (int)new_panel_width,
                                                        d_Y_ptr, (int)panel_height,
                                                        d_A_trailing_ptr, (int)m);
                    if (!guaranteed_skip_final_qr) {
                        T_compute* d_R_offdiag_dest_ptr = d_R_final_ptr + (size_t)(col_start + panel_width) * n + accumulated_rank;
                        cudaMemcpy2DAsync(d_R_offdiag_dest_ptr,
                                          n * sizeof(T_compute),
                                          d_A_trailing_ptr,
                                          m * sizeof(T_compute),
                                          new_panel_width * sizeof(T_compute),
                                          trailing_matrix_cols,
                                          cudaMemcpyDeviceToDevice, stream);
                    }
                }
                } else {
                // --fast: 回到纯 float 的原始 panel W/Y + trailing update 路径。
                buf_B.resize((size_t)panel_height * new_panel_width);
                T_compute* d_B_ptr = thrust::raw_pointer_cast(buf_B.data());
                bool use_geam_path = true; if (const char* eg = std::getenv("HQB_B_GEAM")) { use_geam_path = (eg[0] != '0'); }
                if (use_geam_path) {
                    BlasCaller<T_compute>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                panel_height, new_panel_width,
                                                &minus_one, d_Q_filtered_ptr, panel_height,
                                                &zero, d_Q_filtered_ptr, panel_height,
                                                d_B_ptr, panel_height);
                    dim3 threads_add_id(256);
                    dim3 blocks_add_id((std::min(panel_height, new_panel_width) + threads_add_id.x - 1) / threads_add_id.x);
                    add_identity_diagonal_kernel<T_compute><<<blocks_add_id, threads_add_id, 0, stream>>>(d_B_ptr, panel_height, new_panel_width, panel_height);
                } else {
                    if ((panel_height & 3) == 0) {
                        dim3 t_b(32, 8);
                        dim3 b_b((new_panel_width + t_b.x - 1) / t_b.x,
                                 ((int)panel_height / 4 + t_b.y - 1) / t_b.y);
                        i_minus_matrix_kernel_f4<<<b_b, t_b, 0, stream>>>(d_B_ptr, d_Q_filtered_ptr,(int)panel_height, (int)new_panel_width, (int)panel_height, (int)panel_height);
                    } else {
                        dim3 t_b(32, 8);
                        dim3 b_b((new_panel_width + t_b.x - 1) / t_b.x,
                                 (panel_height + t_b.y - 1) / t_b.y);
                        i_minus_matrix_kernel<T_compute><<<b_b, t_b, 0, stream>>>(d_B_ptr, d_Q_filtered_ptr,
                                                                                  (int)panel_height, (int)new_panel_width,
                                                                                  (int)panel_height, (int)panel_height);
                    }
                }
                buf_W.resize((size_t)panel_height * new_panel_width);
                T_compute* d_W_ptr = thrust::raw_pointer_cast(buf_W.data());
                cudaMemcpyAsync(d_W_ptr, d_B_ptr, (size_t)panel_height * new_panel_width * sizeof(T_compute), cudaMemcpyDeviceToDevice, stream);
                buf_Y.resize((size_t)panel_height * new_panel_width);
                T_compute* d_Y_ptr = thrust::raw_pointer_cast(buf_Y.data());
                int lwork_getrf = 0;
                SolverCaller<T_compute>::getrf_bufferSize(cusolverHandle, (int)panel_height, (int)new_panel_width, d_B_ptr, (int)panel_height, &lwork_getrf);
                buf_getrf_work.resize(lwork_getrf);
                T_compute* d_getrf_work_ptr = thrust::raw_pointer_cast(buf_getrf_work.data());
                int* d_devInfo_ptr = thrust::raw_pointer_cast(d_devInfo.data());
                SolverCaller<T_compute>::getrf(cusolverHandle, (int)panel_height, (int)new_panel_width, d_B_ptr, (int)panel_height, d_getrf_work_ptr, NULL, d_devInfo_ptr);
                cudaMemset(d_Y_ptr, 0, (size_t)panel_height * new_panel_width * sizeof(T_compute));
                dim3 threads_for_L(16, 16);
                dim3 blocks_for_L((new_panel_width + threads_for_L.x - 1) / threads_for_L.x, (panel_height + threads_for_L.y - 1) / threads_for_L.y);
                extract_L_factor_kernel<T_compute><<<blocks_for_L, threads_for_L, 0, stream>>>(d_Y_ptr, panel_height, d_B_ptr, panel_height, panel_height, new_panel_width);
                BlasCaller<T_compute>::trsm(cublasHandle,CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T, CUBLAS_DIAG_UNIT,
                                            panel_height, new_panel_width, &one, d_Y_ptr, panel_height, d_W_ptr, panel_height);
                if (need_panel_ready_event) {
                    cudaEventRecord(ev_panel_ready, stream);
                }
                T_compute* d_W_final_dest_ptr = d_W_final_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaStreamWaitEvent(stream_pack_w, ev_panel_ready, 0);
                    cudaMemcpy2DAsync(d_W_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_W_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_w);
                    cudaEventRecord(ev_pack_w_last, stream_pack_w);
                } else {
                    cudaMemcpy2DAsync(d_W_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_W_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }
                T_compute* d_Y_final_dest_ptr = d_Y_final_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaStreamWaitEvent(stream_pack_y, ev_panel_ready, 0);
                    cudaMemcpy2DAsync(d_Y_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_Y_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_y);
                    cudaEventRecord(ev_pack_y_last, stream_pack_y);
                } else {
                    cudaMemcpy2DAsync(d_Y_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_Y_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }
                long int trailing_matrix_cols = n - (col_start + panel_width);
                if (trailing_matrix_cols > 0) {
                    T_compute* d_A_trailing_ptr = d_A_ptr + (size_t)(col_start + panel_width) * m + row_start;
                    buf_Temp1_fp32.resize((size_t)new_panel_width * trailing_matrix_cols);
                    T_compute* d_Temp1_ptr = thrust::raw_pointer_cast(buf_Temp1_fp32.data());
                    gemm_pair_graph_or_plain<T_compute>(cublasHandle, stream,
                                                        (int)panel_height, (int)trailing_matrix_cols, (int)new_panel_width,
                                                        d_W_ptr, (int)panel_height,
                                                        d_A_trailing_ptr, (int)m,
                                                        d_Temp1_ptr, (int)new_panel_width,
                                                        d_Y_ptr, (int)panel_height,
                                                        d_A_trailing_ptr, (int)m);
                    if (!guaranteed_skip_final_qr) {
                        T_compute* d_R_offdiag_dest_ptr = d_R_final_ptr + (size_t)(col_start + panel_width) * n + accumulated_rank;
                        cudaMemcpy2DAsync(d_R_offdiag_dest_ptr,
                                          n * sizeof(T_compute),
                                          d_A_trailing_ptr,
                                          m * sizeof(T_compute),
                                          new_panel_width * sizeof(T_compute),
                                          trailing_matrix_cols,
                                          cudaMemcpyDeviceToDevice, stream);
                    }
                }
                }
                } else {
                // 构建Q_filtered的W和Y。1.计算I-Q_filtered=B，拷贝d_B为后续TRSM做准备。2.对B进行LU分解，得到L和U，然后提取L矩阵并将对角线置1得到Y。3.利用I-WY^T=Q_filtered,即WY^T=d_B_backup,利用TRSM得到W
                // 计算 I - Q_filtered
                buf_B.resize((size_t)panel_height * new_panel_width);
                T_compute* d_B_ptr = thrust::raw_pointer_cast(buf_B.data());
                // B = I - Q_filtered：提供两种路径，默认用 geam+add（更快）；若 HQB_B_GEAM=0 则回退到融合核
                bool use_geam_path = true; if (const char* eg = std::getenv("HQB_B_GEAM")) { use_geam_path = (eg[0] != '0'); }
                if (use_geam_path) {
                    BlasCaller<T_compute>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                panel_height, new_panel_width,
                                                &minus_one, d_Q_filtered_ptr, panel_height,
                                                &zero, d_Q_filtered_ptr, panel_height,
                                                d_B_ptr, panel_height);
                    dim3 threads_add_id(256);
                    dim3 blocks_add_id((std::min(panel_height, new_panel_width) + threads_add_id.x - 1) / threads_add_id.x);
                    add_identity_diagonal_kernel<T_compute><<<blocks_add_id, threads_add_id, 0, stream>>>(d_B_ptr, panel_height, new_panel_width, panel_height);
                } else {
                    if constexpr (std::is_same_v<T_compute, float>) {
                        if ((panel_height & 3) == 0) {
                            dim3 t_b(32, 8);
                            dim3 b_b((new_panel_width + t_b.x - 1) / t_b.x,
                                     ((int)panel_height / 4 + t_b.y - 1) / t_b.y);
                            i_minus_matrix_kernel_f4<<<b_b, t_b, 0, stream>>>(d_B_ptr, d_Q_filtered_ptr,(int)panel_height, (int)new_panel_width, (int)panel_height, (int)panel_height);
                        } else {
                            dim3 t_b(32, 8);
                            dim3 b_b((new_panel_width + t_b.x - 1) / t_b.x,
                                     (panel_height + t_b.y - 1) / t_b.y);
                            i_minus_matrix_kernel<T_compute><<<b_b, t_b, 0, stream>>>(d_B_ptr, d_Q_filtered_ptr,
                                                                                      (int)panel_height, (int)new_panel_width,
                                                                                      (int)panel_height, (int)panel_height);
                        }
                    } else if constexpr (std::is_same_v<T_compute, double>) {
                        if ((panel_height & 1) == 0) {
                            dim3 t_b(32, 8);
                            dim3 b_b((new_panel_width + t_b.x - 1) / t_b.x,
                                     ((int)panel_height / 2 + t_b.y - 1) / t_b.y);
                            i_minus_matrix_kernel_d2<<<b_b, t_b, 0, stream>>>(d_B_ptr, d_Q_filtered_ptr,
                                                                              (int)panel_height, (int)new_panel_width,
                                                                              (int)panel_height, (int)panel_height);
                        } else {
                            dim3 t_b(32, 8);
                            dim3 b_b((new_panel_width + t_b.x - 1) / t_b.x,(panel_height + t_b.y - 1) / t_b.y);
                            i_minus_matrix_kernel<T_compute><<<b_b, t_b, 0, stream>>>(d_B_ptr, d_Q_filtered_ptr,
                                                                                      (int)panel_height, (int)new_panel_width,
                                                                                      (int)panel_height, (int)panel_height);
                        }
                    } else {
                        dim3 t_b(32, 8);
                        dim3 b_b((new_panel_width + t_b.x - 1) / t_b.x, (panel_height + t_b.y - 1) / t_b.y);
                        i_minus_matrix_kernel<T_compute><<<b_b, t_b, 0, stream>>>(d_B_ptr, d_Q_filtered_ptr,
                                                                                  (int)panel_height, (int)new_panel_width,
                                                                                  (int)panel_height, (int)panel_height);
                    }
                }
                // 把B存一份副本d_W_ptr，方便后续进行TRSM求W
                buf_W.resize((size_t)panel_height * new_panel_width);
                T_compute* d_W_ptr = thrust::raw_pointer_cast(buf_W.data());
                cudaMemcpyAsync(d_W_ptr, d_B_ptr, (size_t)panel_height * new_panel_width * sizeof(T_compute), cudaMemcpyDeviceToDevice, stream);
                
                // 对 B 进行 LU 分解，求解Y
                buf_Y.resize((size_t)panel_height * new_panel_width);
                T_compute* d_Y_ptr = thrust::raw_pointer_cast(buf_Y.data());
                int lwork_getrf = 0;
                SolverCaller<T_compute>::getrf_bufferSize(cusolverHandle, (int)panel_height, (int)new_panel_width, d_B_ptr, (int)panel_height, &lwork_getrf);
                buf_getrf_work.resize(lwork_getrf);
                T_compute* d_getrf_work_ptr = thrust::raw_pointer_cast(buf_getrf_work.data());
                int* d_devInfo_ptr = thrust::raw_pointer_cast(d_devInfo.data()); 
                
                // LU 分解
                SolverCaller<T_compute>::getrf(cusolverHandle, (int)panel_height, (int)new_panel_width, d_B_ptr, (int)panel_height, d_getrf_work_ptr, NULL, d_devInfo_ptr);
                cudaMemset(d_Y_ptr, 0, (size_t)panel_height * new_panel_width * sizeof(T_compute));
                dim3 threads_for_L(16, 16); 
                dim3 blocks_for_L((new_panel_width + threads_for_L.x - 1) / threads_for_L.x, (panel_height + threads_for_L.y - 1) / threads_for_L.y);
                extract_L_factor_kernel<T_compute><<<blocks_for_L, threads_for_L, 0, stream>>>(d_Y_ptr, panel_height, d_B_ptr, panel_height, panel_height, new_panel_width);

                // 利用I-WY^T=Q求解WY^T=(I-Q)=B,用TRSM求解W。
                BlasCaller<T_compute>::trsm(cublasHandle,CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,CUBLAS_OP_T, CUBLAS_DIAG_UNIT, 
                                            panel_height, new_panel_width, &one, d_Y_ptr, panel_height, d_W_ptr, panel_height);

                // 标记本面板 W/Y 计算完成，供并行尾部更新等待
                if (need_panel_ready_event) {
                    cudaEventRecord(ev_panel_ready, stream);
                }

                // 将当前面板的 W 和 Y 组装到最终的全局矩阵中（用独立流并与面板就绪事件同步）
                T_compute* d_W_final_dest_ptr = d_W_final_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaStreamWaitEvent(stream_pack_w, ev_panel_ready, 0);
                    cudaMemcpy2DAsync(d_W_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_W_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_w);
                    cudaEventRecord(ev_pack_w_last, stream_pack_w);
                } else {
                    cudaMemcpy2DAsync(d_W_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_W_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }

                T_compute* d_Y_final_dest_ptr = d_Y_final_ptr + (size_t)accumulated_rank * m;
                if (use_pack_streams) {
                    cudaStreamWaitEvent(stream_pack_y, ev_panel_ready, 0);
                    cudaMemcpy2DAsync(d_Y_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_Y_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream_pack_y);
                    cudaEventRecord(ev_pack_y_last, stream_pack_y);
                } else {
                    cudaMemcpy2DAsync(d_Y_final_dest_ptr + row_start,
                                      m * sizeof(T_compute),
                                      d_Y_ptr,
                                      panel_height * sizeof(T_compute),
                                      panel_height * sizeof(T_compute),
                                      new_panel_width,
                                      cudaMemcpyDeviceToDevice, stream);
                }
                // 更新后续尾矩阵A=(I-WY^T)^T*A，QR计算的块是：d_A_ptr + row_start * m + col_start，这里尾矩阵更新的A：d_A_ptr + (row_start +k) * m + col_start
                // A=(I-WY^T)^T*A=(I-YW^T)A=A-YW^T*A
                long int trailing_matrix_cols = n - (col_start + panel_width);
                if (trailing_matrix_cols > 0) {
                    bool used_lowp_this_iter = false;
                    bool lowp_out_direct_flag = false; // default: convert path
                    size_t active_compute_stream_count = 0;
                    T_compute* d_A_trailing_ptr = d_A_ptr + (size_t)(col_start + panel_width) * m + row_start;
                if constexpr ((std::is_same_v<T_storage, __half> || std::is_same_v<T_storage, __nv_bfloat16>) && std::is_same_v<T_compute, float>) 
                {
                    // 自适应选择：仅在compute=float并且storage=half/bfloat16时转回低精度
                    long m_h = panel_height;
                    long m_r = new_panel_width;
                    long n_t = trailing_matrix_cols;
                    // 仅输出一次 trailing update 路径选择信息
                    static bool s_trailing_choice_printed = false;
                    bool use_lowp = hqbtune::decide_lowp(m_h, m_r, n_t);

                    if (use_lowp) {
                    used_lowp_this_iter = true;
                    if (debug_mode && !s_trailing_choice_printed) {
                        printf("[Info] Trailing update: using LOWP (storage=%s) m_h=%ld, m_r=%ld, n_t=%ld\n",
                               (std::is_same_v<T_storage, __half> ? "half" : "bfloat16"), m_h, m_r, n_t);
                        s_trailing_choice_printed = true;
                    }
                    // 高性能 FP16/BF16 Tensor Core 路径，对齐 K 维（m_h）到 16 的倍数，利于 Tensor Core
                    int m_h_pad = ((int)panel_height + 15) / 16 * 16;
                    // 分配 T_storage (half/bfloat16) 类型的临时缓冲区（按对齐后的行数）
                    buf_W_lowp.resize((size_t)m_h_pad * new_panel_width);
                    buf_Y_lowp.resize((size_t)m_h_pad * new_panel_width);
                    buf_A_trailing_lowp.resize((size_t)m_h_pad * trailing_matrix_cols);
                    buf_Temp1_lowp.resize((size_t)new_panel_width * trailing_matrix_cols);
                    
                    // 等待面板 W/Y 生成完毕后再进行低精转换
                    cudaStreamWaitEvent(stream_conv, ev_panel_ready, 0);
                    // 将 FP32 矩阵下转换为低精度，W 和 Y 是连续的，使用简单的一维内核
                    size_t num_elements_W = (size_t)panel_height * new_panel_width;
                    convert_kernel<T_compute, T_storage><<< (num_elements_W + 255) / 256, 256, 0, stream_conv >>>(d_W_ptr, thrust::raw_pointer_cast(buf_W_lowp.data()), num_elements_W);
                    // 清零 pad 行
                    if (m_h_pad > panel_height) {
                        dim3 tz1(32,8); dim3 bz1((new_panel_width + 31)/32, ((m_h_pad - panel_height) + 7)/8);
                        zero_rows_kernel<T_storage><<<bz1, tz1, 0, stream_conv>>>(thrust::raw_pointer_cast(buf_W_lowp.data()), m_h_pad, panel_height, m_h_pad, new_panel_width);
                    }

                    size_t num_elements_Y = (size_t)panel_height * new_panel_width;
                    convert_kernel<T_compute, T_storage><<< (num_elements_Y + 255) / 256, 256, 0, stream_conv >>>(d_Y_ptr, thrust::raw_pointer_cast(buf_Y_lowp.data()), num_elements_Y);
                    if (m_h_pad > panel_height) {
                        dim3 tz2(32,8); dim3 bz2((new_panel_width + 31)/32, ((m_h_pad - panel_height) + 7)/8);
                        zero_rows_kernel<T_storage><<<bz2, tz2, 0, stream_conv>>>(thrust::raw_pointer_cast(buf_Y_lowp.data()), m_h_pad, panel_height, m_h_pad, new_panel_width);
                    }
                    
                    const int VECLEN = 4;
                    size_t panel_height_main = (panel_height / VECLEN) * VECLEN; 
                    size_t panel_height_rem = panel_height % VECLEN;            
        
                    // 调用矢量化内核处理主体部分
                    if (panel_height_main > 0) {
                        dim3 threads_vec(16, 8); 
                        dim3 blocks_vec((trailing_matrix_cols + threads_vec.x - 1) / threads_vec.x,(panel_height_main / VECLEN + threads_vec.y - 1) / threads_vec.y);
                        convert_kernel_strided_vec4<T_compute, T_storage><<<blocks_vec, threads_vec, 0, stream_conv>>>(
                            d_A_trailing_ptr, m, 
                            thrust::raw_pointer_cast(buf_A_trailing_lowp.data()), m_h_pad, 
                            panel_height_main, trailing_matrix_cols);
                    }

                    // 调用标量内核处理尾部
                    if (panel_height_rem > 0) {
                        T_compute* src_rem_ptr = d_A_trailing_ptr + panel_height_main;
                        T_storage* dst_rem_ptr = thrust::raw_pointer_cast(buf_A_trailing_lowp.data()) + panel_height_main;

                        dim3 threads_scalar(16, 16);
                        dim3 blocks_scalar((trailing_matrix_cols + threads_scalar.x - 1) / threads_scalar.x,(panel_height_rem + threads_scalar.y - 1) / threads_scalar.y);
                        convert_kernel_strided<T_compute, T_storage><<<blocks_scalar, threads_scalar, 0, stream_conv>>>(
                            src_rem_ptr, m, 
                            dst_rem_ptr, m_h_pad, 
                            panel_height_rem, trailing_matrix_cols);
                    }
                    // 清零 pad 行
                    if (m_h_pad > panel_height) {
                        dim3 tz3(32,8); dim3 bz3((trailing_matrix_cols + 31)/32, ((m_h_pad - panel_height) + 7)/8);
                        zero_rows_kernel<T_storage><<<bz3, tz3, 0, stream_conv>>>(thrust::raw_pointer_cast(buf_A_trailing_lowp.data()), m_h_pad, panel_height, m_h_pad, trailing_matrix_cols);
                    }

                    // 转换完成，供后续计算流等待
                    cudaEventRecord(ev_conv_done, stream_conv);

                    // 3. 执行两次 GEMM（低精路径）。第二次 GEMM 的输出可选：- convert（默认）：输出到低精 then 转回 FP32 写回 A_trailing（精度优先）- direct ：直接输出到 FP32 的 A_trailing（速度优先，命令行 --fast）
                    bool lowp_out_direct = fast_mode;
                    lowp_out_direct_flag = lowp_out_direct;
                    const int tile_cols = base_tile_cols;
                    const bool use_parallel_lowp_trailing = use_parallel_trailing(trailing_matrix_cols, tile_cols);
                    if (use_parallel_lowp_trailing) {
                        const int n_chunks = (int)((trailing_matrix_cols + tile_cols - 1) / tile_cols);
                        active_compute_stream_count = std::min(compute_streams.size(), (size_t)n_chunks);
                        for (int c = 0; c < n_chunks; ++c) {
                            int sid = c % (int)compute_streams.size();
                            int j0  = c * tile_cols;
                            int wc  = std::min(tile_cols, (int)trailing_matrix_cols - j0);
                            const T_storage* A_lp_chunk = thrust::raw_pointer_cast(buf_A_trailing_lowp.data()) + (size_t)j0 * m_h_pad;
                            T_storage*       T_lp_chunk = thrust::raw_pointer_cast(buf_Temp1_lowp.data()) + (size_t)j0 * new_panel_width;
                            T_compute*       A_fp32_chunk = d_A_trailing_ptr + (size_t)j0 * m;
                            // 等待面板 W/Y 和低精转换完成
                            cudaStreamWaitEvent(compute_streams[sid], ev_panel_ready, 0);
                            cudaStreamWaitEvent(compute_streams[sid], ev_conv_done, 0);
                            // Temp1_chunk = W_lp^T * A_lp_chunk
                            gemmex_lt<T_storage, T_storage, T_storage, T_compute>(
                                compute_handles[sid], CUBLAS_OP_T, CUBLAS_OP_N,
                                (int)new_panel_width, wc, (int)m_h_pad,
                                &one,
                                thrust::raw_pointer_cast(buf_W_lowp.data()), (int)m_h_pad,
                                A_lp_chunk, (int)m_h_pad,
                                &zero,
                                T_lp_chunk, (int)new_panel_width,
                                compute_streams[sid]);
                            if (!lowp_out_direct) {
                                // A_lp_chunk -= Y_lp * Temp1_chunk（低精缓冲内）
                                gemmex_lt<T_storage, T_storage, T_storage, T_compute>(compute_handles[sid], CUBLAS_OP_N, CUBLAS_OP_N,
                                                                                      (int)panel_height, wc, (int)new_panel_width,
                                                                                      &minus_one,
                                                                                      thrust::raw_pointer_cast(buf_Y_lowp.data()), (int)m_h_pad,
                                                                                      T_lp_chunk, (int)new_panel_width,
                                                                                      &one,
                                                                                      const_cast<T_storage*>(A_lp_chunk), (int)m_h_pad,
                                                                                      compute_streams[sid]);
                            } else {
                                // 直接写回到 FP32 目标
                                gemmex_lt<T_storage, T_storage, T_compute, T_compute>(compute_handles[sid], CUBLAS_OP_N, CUBLAS_OP_N,
                                                                                      (int)panel_height, wc, (int)new_panel_width,
                                                                                      &minus_one,
                                                                                      thrust::raw_pointer_cast(buf_Y_lowp.data()), (int)m_h_pad,
                                                                                      T_lp_chunk, (int)new_panel_width,
                                                                                      &one,
                                                                                      A_fp32_chunk, (int)m,
                                                                                      compute_streams[sid]);
                            }
                        }
                        // 记录完成事件并在需要时统一回写
                        for (size_t si = 0; si < active_compute_stream_count; ++si) {
                            cudaEventRecord(ev_compute_done[si], compute_streams[si]);
                        }
                        if (!lowp_out_direct) {
                            for (size_t si = 0; si < active_compute_stream_count; ++si) {
                                cudaStreamWaitEvent(stream_conv, ev_compute_done[si], 0);
                            }
                            if (panel_height_main > 0) {
                                dim3 threads_vec(16, 8);
                                dim3 blocks_vec((trailing_matrix_cols + threads_vec.x - 1) / threads_vec.x,
                                                (int)((panel_height_main / VECLEN + threads_vec.y - 1) / threads_vec.y));
                                convert_kernel_strided_vec4<T_storage, T_compute><<<blocks_vec, threads_vec, 0, stream_conv>>>(
                                    thrust::raw_pointer_cast(buf_A_trailing_lowp.data()), m_h_pad,
                                    d_A_trailing_ptr, m,
                                    panel_height_main, trailing_matrix_cols);
                            }
                            if (panel_height_rem > 0) {
                                const T_storage* src_rem_ptr = thrust::raw_pointer_cast(buf_A_trailing_lowp.data()) + panel_height_main;
                                T_compute* dst_rem_ptr = d_A_trailing_ptr + panel_height_main;
                                dim3 threads_scalar(16, 16);
                                dim3 blocks_scalar((trailing_matrix_cols + threads_scalar.x - 1) / threads_scalar.x,
                                                   (panel_height_rem + threads_scalar.y - 1) / threads_scalar.y);
                                convert_kernel_strided<T_storage, T_compute><<<blocks_scalar, threads_scalar, 0, stream_conv>>>(
                                    src_rem_ptr, m_h_pad,
                                    dst_rem_ptr, m,
                                    panel_height_rem, trailing_matrix_cols);
                            }
                            cudaEventRecord(ev_back_done, stream_conv);
                        }
                        // 主流等待所有计算完成（或回写完成）在下方 R_offdiag 前处理
                    } else {
                        // 单流低精路径
                        // Temp1_lowp = W_lowp^T * A_trailing_lowp
                        gemmex_lt<T_storage, T_storage, T_storage, T_compute>(cublasHandle,
                                                                           CUBLAS_OP_T, CUBLAS_OP_N, 
                                                                           new_panel_width, trailing_matrix_cols, m_h_pad,
                                                                           &one, 
                                                                           thrust::raw_pointer_cast(buf_W_lowp.data()), m_h_pad,         
                                                                           thrust::raw_pointer_cast(buf_A_trailing_lowp.data()), m_h_pad,             
                                                                           &zero, 
                                                                           thrust::raw_pointer_cast(buf_Temp1_lowp.data()), new_panel_width);
                        if (!lowp_out_direct) {
                            // A_trailing_lowp = A_trailing_lowp - Y_lowp * Temp1_lowp（输出低精），然后转换回 FP32
                            gemmex_lt<T_storage, T_storage, T_storage, T_compute>(cublasHandle,
                                                                               CUBLAS_OP_N, CUBLAS_OP_N, 
                                                                               panel_height, trailing_matrix_cols, new_panel_width,
                                                                               &minus_one, 
                                                                               thrust::raw_pointer_cast(buf_Y_lowp.data()), m_h_pad,          
                                                                               thrust::raw_pointer_cast(buf_Temp1_lowp.data()), new_panel_width,  
                                                                               &one,       
                                                                               thrust::raw_pointer_cast(buf_A_trailing_lowp.data()), m_h_pad);
                            if (panel_height_main > 0) {
                                dim3 threads_vec(16, 8);
                                dim3 blocks_vec((trailing_matrix_cols + threads_vec.x - 1) / threads_vec.x,(panel_height_main / VECLEN + threads_vec.y - 1) / threads_vec.y);
                                convert_kernel_strided_vec4<T_storage, T_compute><<<blocks_vec, threads_vec, 0, stream_conv>>>(thrust::raw_pointer_cast(buf_A_trailing_lowp.data()), m_h_pad,
                                                                                                                          d_A_trailing_ptr, m,
                                                                                                                          panel_height_main, trailing_matrix_cols);
                            }
                            if (panel_height_rem > 0) {
                                const T_storage* src_rem_ptr = thrust::raw_pointer_cast(buf_A_trailing_lowp.data()) + panel_height_main;
                                T_compute* dst_rem_ptr = d_A_trailing_ptr + panel_height_main;
                                dim3 threads_scalar(16, 16);
                                dim3 blocks_scalar((trailing_matrix_cols + threads_scalar.x - 1) / threads_scalar.x,(panel_height_rem + threads_scalar.y - 1) / threads_scalar.y);
                                convert_kernel_strided<T_storage, T_compute><<<blocks_scalar, threads_scalar, 0, stream_conv>>>(src_rem_ptr, m_h_pad, 
                                                                                                                           dst_rem_ptr, m, 
                                                                                                                           panel_height_rem, trailing_matrix_cols);
                            }
                            cudaEventRecord(ev_back_done, stream_conv);
                        } else {
                            // 直接输出到 FP32 的 A_trailing
                            gemmex_lt<T_storage, T_storage, T_compute, T_compute>(cublasHandle,
                                                                               CUBLAS_OP_N, CUBLAS_OP_N, 
                                                                               panel_height, trailing_matrix_cols, new_panel_width,
                                                                               &minus_one, 
                                                                               thrust::raw_pointer_cast(buf_Y_lowp.data()), m_h_pad,          
                                                                               thrust::raw_pointer_cast(buf_Temp1_lowp.data()), new_panel_width,  
                                                                               &one,       
                                                                               d_A_trailing_ptr, m);
                        }
                    }
                    } else {
                        if (debug_mode && !s_trailing_choice_printed) {
                            printf("[Info] Trailing update: using FP32 (no conversion) m_h=%ld, m_r=%ld, n_t=%ld\n",
                                   m_h, m_r, n_t);
                            s_trailing_choice_printed = true;
                        }
                        // FP32
                        T_compute* d_Temp1_ptr = thrust::raw_pointer_cast(buf_Temp1_fp32.data());
                        gemm_pair_graph_or_plain<T_compute>(cublasHandle, stream,
                                                            (int)panel_height, (int)trailing_matrix_cols, (int)new_panel_width,
                                                            d_W_ptr, (int)panel_height,
                                                            d_A_trailing_ptr, (int)m,
                                                            d_Temp1_ptr, (int)new_panel_width,
                                                            d_Y_ptr, (int)panel_height,
                                                            d_A_trailing_ptr, (int)m);
                    }
                } 
    else 
    {
                    // Temp1 = W^T * A_trailing （安全版多流：仅 FP32/FP64 路径分片并行）
                    T_compute* d_Temp1_ptr = thrust::raw_pointer_cast(buf_Temp1_fp32.data());
                    const int tile_cols = base_tile_cols;
                    const bool use_parallel_fp_trailing = use_parallel_trailing(trailing_matrix_cols, tile_cols);
                    if (use_parallel_fp_trailing) {
                        const int n_chunks = (int)((trailing_matrix_cols + tile_cols - 1) / tile_cols);
                        active_compute_stream_count = std::min(compute_streams.size(), (size_t)n_chunks);
                        for (int c = 0; c < n_chunks; ++c) {
                            int sid = c % (int)compute_streams.size();
                            int j0 = c * tile_cols;
                            int wc = std::min(tile_cols, (int)trailing_matrix_cols - j0);
                            T_compute* A_chunk = d_A_trailing_ptr + (size_t)j0 * m;
                            T_compute* T_chunk = d_Temp1_ptr + (size_t)j0 * new_panel_width;
                            // 等待面板数据（W/Y）就绪
                            cudaStreamWaitEvent(compute_streams[sid], ev_panel_ready, 0);
                            gemm_pair_graph_or_plain<T_compute>(compute_handles[sid], compute_streams[sid],
                                                                (int)panel_height, wc, (int)new_panel_width,
                                                                d_W_ptr, (int)panel_height,
                                                                A_chunk, (int)m,
                                                                T_chunk, (int)new_panel_width,
                                                                d_Y_ptr, (int)panel_height,
                                                                A_chunk, (int)m);
                        }
                        // 等待所有计算流完成，再读取 R_offdiag 或进入下一面板
                        for (size_t si = 0; si < active_compute_stream_count; ++si) {
                            cudaEventRecord(ev_compute_done[si], compute_streams[si]);
                            cudaStreamWaitEvent(stream, ev_compute_done[si], 0);
                        }
                    } else {
                        gemm_pair_graph_or_plain<T_compute>(cublasHandle, stream,
                                                            (int)panel_height, (int)trailing_matrix_cols, (int)new_panel_width,
                                                            d_W_ptr, (int)panel_height,
                                                            d_A_trailing_ptr, (int)m,
                                                            d_Temp1_ptr, (int)new_panel_width,
                                                            d_Y_ptr, (int)panel_height,
                                                            d_A_trailing_ptr, (int)m);
                    }
     }
                    // 更新R_final对应部分
                    if (!guaranteed_skip_final_qr) {
                        T_compute* d_R_offdiag_dest_ptr = d_R_final_ptr + (size_t)(col_start + panel_width) * n + accumulated_rank;
                        if (used_lowp_this_iter && !lowp_out_direct_flag) {
                            // 确保回写完成后再读取 A_trailing（避免 CPU 阻塞）
                            cudaStreamWaitEvent(stream, ev_back_done, 0);
                        } else if (used_lowp_this_iter && lowp_out_direct_flag && active_compute_stream_count > 0) {
                            // 直接写回 FP32：需等待所有计算流完成
                            for (size_t si = 0; si < active_compute_stream_count; ++si) {
                                cudaStreamWaitEvent(stream, ev_compute_done[si], 0);
                            }
                        }
                        cudaMemcpy2DAsync(d_R_offdiag_dest_ptr,
                                          n * sizeof(T_compute),
                                          d_A_trailing_ptr,
                                          m * sizeof(T_compute),
                                          new_panel_width * sizeof(T_compute),
                                          trailing_matrix_cols,
                                          cudaMemcpyDeviceToDevice, stream);
                    }
                }
                }
                } // end direct-projection fast branch
                // 更新状态 
                accumulated_rank += new_panel_width;
                if (!g_force_full_rank && new_panel_width < panel_width) {
                    stop_condition_met = true;
                    final_rank = accumulated_rank;
                    if (g_unknown_rank_mode) {
                        printf("[AdaptiveRank] Stop: local cutoff at panel %zu reduced width from %ld to %ld.\n",
                               j, panel_width, new_panel_width);
                    }
                }
                if (g_unknown_rank_mode && g_energy_criterion_enabled && initial_matrix_norm_sq > 0.0) {
                    const long int next_row_start = row_start + panel_width;
                    const long int next_col_start = col_start + panel_width;
                    const int residual_rows = (next_row_start < m) ? (int)(m - next_row_start) : 0;
                    const int residual_cols = (next_col_start < n) ? (int)(n - next_col_start) : 0;
                    double residual_norm_sq = 0.0;
                    if (residual_rows > 0 && residual_cols > 0) {
                        const T_compute* d_residual_ptr = d_A_ptr + (size_t)next_col_start * (size_t)m + (size_t)next_row_start;
                        residual_norm_sq = device_matrix_norm_sq<T_compute>(stream, d_residual_ptr, (int)m, residual_rows, residual_cols);
                    }
                    const double captured_energy = std::max(0.0, 1.0 - residual_norm_sq / initial_matrix_norm_sq);
                    if (debug_mode) {
                        printf("[AdaptiveRank] panel %zu captured_energy=%.6e residual_ratio=%.6e\n",
                               j, captured_energy, residual_norm_sq / initial_matrix_norm_sq);
                    }
                    if (captured_energy >= 1.0 - g_energy_tau) {
                        stop_condition_met = true;
                        final_rank = accumulated_rank;
                        printf("[AdaptiveRank] Stop: captured energy %.6e reached target %.6e at panel %zu.\n",
                               captured_energy, 1.0 - g_energy_tau, j);
                    }
                }
                if (accumulated_rank >= real_rank) {
                    stop_condition_met = true;
                    
                    if (final_rank != accumulated_rank) { 
                       final_rank = accumulated_rank;   
                    }
                     printf("\n         !!! Stopping: Reached or exceeded theoretical rank. Final rank set to: %zu\n", final_rank);
                }
            if (stop_condition_met) {
                break;
            }
        }
        final_rank = accumulated_rank;
        float elapsed_time_QB = stopTimer();
            stage_time_qb_ms = static_cast<double>(elapsed_time_QB);
            printf("QB finished successfully in %f ms.\n", elapsed_time_QB);
            printf("[Rank] target_rank=%zu, kept_rank=%zu, final_rank=%zu\n",
                   (size_t)real_rank, (size_t)accumulated_rank, (size_t)final_rank);

            // 等待所有 W/Y 汇入全局缓冲的异步拷贝完成
            if (use_pack_streams) {
                cudaStreamWaitEvent(stream, ev_pack_w_last, 0);
                cudaStreamWaitEvent(stream, ev_pack_y_last, 0);
            }

            startTimer();
            thrust::device_vector<T_compute> d_Q_basis(m * final_rank);
            T_compute* d_Q_basis_ptr = thrust::raw_pointer_cast(d_Q_basis.data());
            thrust::device_vector<double> d_Q_basis_double_hold;
            const double* d_Q_basis_double_hold_ptr = nullptr;
            const bool single_panel_q = (final_rank <= (size_t)qb_block_k);
            const size_t qbasis_panel_count = (final_rank + (size_t)qb_block_k - 1) / (size_t)qb_block_k;
            if (single_panel_q) {
                // For a single kept panel, Q_final already is the thin Q basis.
                cudaMemcpyAsync(d_Q_basis_ptr,
                                d_Q_final_ptr,
                                sizeof(T_compute) * (size_t)m * (size_t)final_rank,
                                cudaMemcpyDeviceToDevice,
                                stream);
            } else {
                if constexpr (std::is_same_v<T_storage, float> && std::is_same_v<T_compute, float>) {
                    const bool use_double_qbasis_float_path =
                        (!use_ppopp_float_path) &&
                        (!force_flat_fast_float_path) &&
                        (debug_mode ||
                         g_accuracy == AccuracyLevel::High ||
                         qbasis_panel_count > 2 ||
                         final_rank > 96);
                    if (use_double_qbasis_float_path) {
                        if (debug_mode && !fast_mode) {
                            printf("[Info] Building Q_basis in double for multi-panel float QB path.\n");
                        } else if (debug_mode && fast_mode) {
                            printf("[Info] Building Q_basis in double on float fast path (reuse float W/Y, stabilize recursive W).\n");
                        }
                        thrust::device_vector<double> d_recursive_work_W_double((size_t)final_rank * (size_t)qb_block_k);
                        d_Q_basis_double_hold.resize((size_t)m * (size_t)final_rank);
                        double* d_recursive_work_W_double_ptr = thrust::raw_pointer_cast(d_recursive_work_W_double.data());
                        double* d_Q_basis_double_ptr = thrust::raw_pointer_cast(d_Q_basis_double_hold.data());
                        double* d_W_qbasis_double_ptr = fast_mode ? nullptr : d_W_final_accum_double_ptr;
                        double* d_Y_qbasis_double_ptr = fast_mode ? nullptr : d_Y_final_accum_double_ptr;
                        thrust::device_vector<double> d_W_final_double_fallback;
                        thrust::device_vector<double> d_Y_final_double_fallback;

                        if (d_W_qbasis_double_ptr == nullptr || d_Y_qbasis_double_ptr == nullptr) {
                            d_W_final_double_fallback.resize((size_t)m * (size_t)final_rank);
                            d_Y_final_double_fallback.resize((size_t)m * (size_t)final_rank);
                            d_W_qbasis_double_ptr = thrust::raw_pointer_cast(d_W_final_double_fallback.data());
                            d_Y_qbasis_double_ptr = thrust::raw_pointer_cast(d_Y_final_double_fallback.data());
                            convert_kernel<float, double><<<(int)((((size_t)m * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                d_W_final_ptr, d_W_qbasis_double_ptr, (size_t)m * (size_t)final_rank);
                            convert_kernel<float, double><<<(int)((((size_t)m * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                d_Y_final_ptr, d_Y_qbasis_double_ptr, (size_t)m * (size_t)final_rank);
                        }

                        recursive_compute_w<double>(cublasHandle,
                                                    (size_t)m,
                                                    (size_t)final_rank,
                                                    (size_t)qb_block_k,
                                                    d_W_qbasis_double_ptr, (size_t)m,
                                                    d_Y_qbasis_double_ptr, (size_t)m,
                                                    d_recursive_work_W_double_ptr);

                        cudaMemsetAsync(d_Q_basis_double_ptr, 0, sizeof(double) * (size_t)m * (size_t)final_rank, stream);
                        BlasCaller<double>::gemm(cublasHandle,
                                                 CUBLAS_OP_N, CUBLAS_OP_T,
                                                 (int)m, (int)final_rank, (int)final_rank,
                                                 &minus_one_d,
                                                 d_W_qbasis_double_ptr, (int)m,
                                                 d_Y_qbasis_double_ptr, (int)m,
                                                 &zero_d,
                                                 d_Q_basis_double_ptr, (int)m);

                        dim3 threads_diag(256);
                        dim3 blocks_diag((final_rank + 255) / 256);
                        add_identity_diagonal_kernel<double><<<blocks_diag, threads_diag, 0, stream>>>(
                            d_Q_basis_double_ptr, m, final_rank, m, 1.0);
                        CUDA_CHECK(cudaGetLastError());

                        convert_kernel<double, float><<<(int)((((size_t)m * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_Q_basis_double_ptr, d_Q_basis_ptr, (size_t)m * (size_t)final_rank);
                        d_Q_basis_double_hold_ptr = d_Q_basis_double_ptr;
                    } else {
                        if (debug_mode) {
                            printf("[Info] Building Q_basis in float for small multi-panel float QB path.\n");
                        }
                        thrust::device_vector<T_compute> d_recursive_work_W((size_t)final_rank * (size_t)qb_block_k);
                        T_compute* d_recursive_work_W_ptr = thrust::raw_pointer_cast(d_recursive_work_W.data());
                        recursive_compute_w<T_compute>(cublasHandle,
                                                       (size_t)m,
                                                       (size_t)final_rank,
                                                       (size_t)qb_block_k,
                                                       d_W_final_ptr, (size_t)m,
                                                       d_Y_final_ptr, (size_t)m,
                                                       d_recursive_work_W_ptr);

                        cudaMemsetAsync(d_Q_basis_ptr, 0, sizeof(T_compute) * (size_t)m * (size_t)final_rank, stream);
                        BlasCaller<T_compute>::gemm(cublasHandle,
                                                    CUBLAS_OP_N, CUBLAS_OP_T,
                                                    (int)m, (int)final_rank, (int)final_rank,
                                                    &minus_one,
                                                    d_W_final_ptr, (int)m,
                                                    d_Y_final_ptr, (int)m,
                                                    &zero,
                                                    d_Q_basis_ptr, (int)m);

                        dim3 threads_diag(256);
                        dim3 blocks_diag((final_rank + 255) / 256);
                        add_identity_diagonal_kernel<T_compute><<<blocks_diag, threads_diag, 0, stream>>>(
                            d_Q_basis_ptr, m, final_rank, m, T_compute(1.0));
                        CUDA_CHECK(cudaGetLastError());
                    }
                } else {
                    thrust::device_vector<T_compute> d_recursive_work_W((size_t)final_rank * (size_t)qb_block_k);
                    T_compute* d_recursive_work_W_ptr = thrust::raw_pointer_cast(d_recursive_work_W.data());
                    recursive_compute_w<T_compute>(cublasHandle,
                                                   (size_t)m,
                                                   (size_t)final_rank,
                                                   (size_t)qb_block_k,
                                                   d_W_final_ptr, (size_t)m,
                                                   d_Y_final_ptr, (size_t)m,
                                                   d_recursive_work_W_ptr);

                    cudaMemsetAsync(d_Q_basis_ptr, 0, sizeof(T_compute) * (size_t)m * (size_t)final_rank, stream);
                    BlasCaller<T_compute>::gemm(cublasHandle,
                                                CUBLAS_OP_N, CUBLAS_OP_T,
                                                (int)m, (int)final_rank, (int)final_rank,
                                                &minus_one,
                                                d_W_final_ptr, (int)m,
                                                d_Y_final_ptr, (int)m,
                                                &zero,
                                                d_Q_basis_ptr, (int)m);

                    dim3 threads_diag(256);
                    dim3 blocks_diag((final_rank + 255) / 256);
                    add_identity_diagonal_kernel<T_compute><<<blocks_diag, threads_diag, 0, stream>>>(
                        d_Q_basis_ptr, m, final_rank, m, T_compute(1.0));
                    CUDA_CHECK(cudaGetLastError());
                }
            }
            thrust::device_vector<T_compute> d_Q_basis_trim;
            const T_compute* d_Q_basis_effective_ptr = d_Q_basis_ptr;
            int ld_Q_basis_effective = (int)m;
            if (need_pad) {
                d_Q_basis_trim.resize((size_t)m_effective * (size_t)final_rank);
                dim3 tQtrim(16, 16);
                dim3 bQtrim((int)((final_rank + tQtrim.x - 1) / tQtrim.x),
                            (int)(((int)m_effective + tQtrim.y - 1) / tQtrim.y));
                convert_kernel_strided<T_compute, T_compute><<<bQtrim, tQtrim, 0, stream>>>(
                    d_Q_basis_ptr, (size_t)m,
                    thrust::raw_pointer_cast(d_Q_basis_trim.data()), (size_t)m_effective,
                    (size_t)m_effective, (size_t)final_rank);
                CUDA_CHECK(cudaGetLastError());
                d_Q_basis_effective_ptr = thrust::raw_pointer_cast(d_Q_basis_trim.data());
                ld_Q_basis_effective = (int)m_effective;
            }
            float elapsed_time_Q = stopTimer();
            stage_time_q_build_ms = static_cast<double>(elapsed_time_Q);
            printf("Q generation finished successfully in %f ms.\n", elapsed_time_Q);

            startTimer();
            const double ratio_n_over_rank = (final_rank > 0)
                                              ? static_cast<double>(n) / static_cast<double>(final_rank)
                                              : std::numeric_limits<double>::infinity();
            const bool skip_final_qr =
                force_flat_fast_float_path ? false : (ratio_n_over_rank > kRFinalQrRatioThreshold);
            // Keep the default float path in mixed precision all the way through
            // the bypassed-Step4 case as well; otherwise rank-64 style runs
            // prematurely fall back to pure float at R_proj = Q^T A.
            const bool use_float_mixed_path =
                std::is_same_v<T_storage, float> && std::is_same_v<T_compute, float> &&
                !fast_mode && !use_ppopp_float_path;
            const bool need_full_double_views =
                std::is_same_v<T_compute, float> && use_float_mixed_path;
            const char* rproj_mode_str =
                std::is_same_v<T_compute, double> ? "double" :
                (use_float_mixed_path ? "double" : "float_tf32");
            printf("[Route] solver=HQB path=generic storage=%s compute=%s final_rank=%zu step4=%s rproj=%s ratio_n_over_rank=%.3f threshold=%.3f\n",
                   precision_name<T_storage>(),
                   precision_name<T_compute>(),
                   final_rank,
                   skip_final_qr ? "bypass" : "qr",
                   rproj_mode_str,
                   ratio_n_over_rank,
                   kRFinalQrRatioThreshold);
            thrust::device_vector<double> d_A_effective_double;
            thrust::device_vector<double> d_Q_basis_effective_double;
            const double* d_A_effective_raw_double_ptr = nullptr;
            const double* d_Q_basis_effective_double_ptr = nullptr;
            if constexpr (std::is_same_v<T_compute, float>) {
                if (need_full_double_views) {
                    d_A_effective_double.resize((size_t)m_effective * (size_t)n);
                    convert_kernel<float, double><<<(int)((((size_t)m_effective * (size_t)n) + 255) / 256), 256, 0, stream>>>(
                        d_A_effective_raw, thrust::raw_pointer_cast(d_A_effective_double.data()), (size_t)m_effective * (size_t)n);
                    d_A_effective_raw_double_ptr = thrust::raw_pointer_cast(d_A_effective_double.data());
                    if (d_Q_basis_double_hold_ptr != nullptr) {
                        if (need_pad) {
                            d_Q_basis_effective_double.resize((size_t)m_effective * (size_t)final_rank);
                            dim3 tQtrimD(16, 16);
                            dim3 bQtrimD((int)((final_rank + tQtrimD.x - 1) / tQtrimD.x),
                                         (int)(((int)m_effective + tQtrimD.y - 1) / tQtrimD.y));
                            convert_kernel_strided<double, double><<<bQtrimD, tQtrimD, 0, stream>>>(
                                d_Q_basis_double_hold_ptr, (size_t)m,
                                thrust::raw_pointer_cast(d_Q_basis_effective_double.data()), (size_t)m_effective,
                                (size_t)m_effective, (size_t)final_rank);
                            d_Q_basis_effective_double_ptr = thrust::raw_pointer_cast(d_Q_basis_effective_double.data());
                        } else {
                            d_Q_basis_effective_double_ptr = d_Q_basis_double_hold_ptr;
                        }
                    } else {
                        d_Q_basis_effective_double.resize((size_t)m_effective * (size_t)final_rank);
                        convert_kernel<float, double><<<(int)((((size_t)m_effective * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_Q_basis_effective_ptr, thrust::raw_pointer_cast(d_Q_basis_effective_double.data()), (size_t)m_effective * (size_t)final_rank);
                        d_Q_basis_effective_double_ptr = thrust::raw_pointer_cast(d_Q_basis_effective_double.data());
                    }
                }
            } else {
                d_A_effective_raw_double_ptr = d_A_effective_raw;
                d_Q_basis_effective_double_ptr = d_Q_basis_effective_ptr;
            }

            // === Step 4: 为 SVD 生成最终 R ===
            // 先重算 R_proj = Q^T * A，再对 R_proj^T 做 QR
            thrust::device_vector<T_compute> d_R_proj((size_t)final_rank * (size_t)n);
            T_compute* d_R_proj_ptr = thrust::raw_pointer_cast(d_R_proj.data());
            thrust::device_vector<double> d_R_proj_double;
            const double* d_R_proj_double_ptr = nullptr;
            if constexpr (std::is_same_v<T_compute, float>) {
                if (use_float_mixed_path) {
                    d_R_proj_double.resize((size_t)final_rank * (size_t)n);
                    double* d_R_proj_double_mut_ptr = thrust::raw_pointer_cast(d_R_proj_double.data());
                    BlasCaller<double>::gemm(cublasHandle,
                                             CUBLAS_OP_T, CUBLAS_OP_N,
                                             (int)final_rank, (int)n, (int)m_effective,
                                             &one_d,
                                             d_Q_basis_effective_double_ptr, ld_Q_basis_effective,
                                             d_A_effective_raw_double_ptr, ld_A_effective,
                                             &zero_d,
                                             d_R_proj_double_mut_ptr, (int)final_rank);
                    convert_kernel<double, float><<<(int)((((size_t)final_rank * (size_t)n) + 255) / 256), 256, 0, stream>>>(
                        d_R_proj_double_mut_ptr, d_R_proj_ptr, (size_t)final_rank * (size_t)n);
                    d_R_proj_double_ptr = d_R_proj_double_mut_ptr;
                } else {
                    BlasCaller<T_compute>::gemm(cublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                (int)final_rank, (int)n, (int)m_effective,
                                                &one,
                                                d_Q_basis_effective_ptr, ld_Q_basis_effective,
                                                d_A_effective_raw, ld_A_effective,
                                                &zero,
                                                d_R_proj_ptr, (int)final_rank);
                }
            } else {
                BlasCaller<T_compute>::gemm(cublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            (int)final_rank, (int)n, (int)m_effective,
                                            &one,
                                            d_Q_basis_effective_ptr, ld_Q_basis_effective,
                                            d_A_effective_raw, ld_A_effective,
                                            &zero,
                                            d_R_proj_ptr, (int)final_rank);
            }
            float elapsed_time_rproj = stopTimer();
            stage_time_rproj_ms = static_cast<double>(elapsed_time_rproj);
            printf("R_proj / basis prep finished in %f ms.\n", elapsed_time_rproj);
            if (debug_mode) {
                const long int verify_m = m_effective;
                if constexpr (std::is_same_v<T_compute, float>) {
                    if (use_float_mixed_path) {
                        verify_qb_decomposition<double>(cublasHandle, verify_m, n, final_rank,
                                                        d_A_effective_double, d_Q_basis_effective_double_ptr,
                                                        d_R_proj_double_ptr, final_rank, stream);
                    } else {
                        const thrust::device_vector<T_compute>* A_verify_ptr = d_A_effective_ptr;
                        const T_compute* q_verify_ptr = d_Q_basis_effective_ptr;
                        verify_qb_decomposition<T_compute>(cublasHandle, verify_m, n, final_rank,
                                                           *A_verify_ptr, q_verify_ptr,
                                                           d_R_proj_ptr, final_rank, stream);
                    }
                } else {
                    const thrust::device_vector<T_compute>* A_verify_ptr = d_A_effective_ptr;
                    const T_compute* q_verify_ptr = d_Q_basis_effective_ptr;
                    // 与旧版一致：QB 验证使用重投影后的 R_proj
                    verify_qb_decomposition<T_compute>(cublasHandle, verify_m, n, final_rank, *A_verify_ptr, q_verify_ptr, d_R_proj_ptr, final_rank, stream);
                }
            }
            // 对 R_proj 做转置，再对转置后的矩阵执行 QR
            printf("=== Step 4: Generate Final R For SVD ===\n");
            if (skip_final_qr) {
                printf("[Step4] Mode: Bypass final QR (n/final_rank=%.3f > threshold %.3f).\n",
                       ratio_n_over_rank, kRFinalQrRatioThreshold);
                stage_time_step4_ms = 0.0;
            } else {
                if (force_flat_fast_float_path) {
                    printf("[Step4] Mode: Execute final QR (float --fast flat path).\n");
                } else {
                    printf("[Step4] Mode: Execute final QR (n/final_rank=%.3f <= threshold %.3f).\n",
                           ratio_n_over_rank, kRFinalQrRatioThreshold);
                }
                startTimer();
            }
            thrust::device_vector<T_compute> d_R_final_T(n * final_rank);
            T_compute* d_R_final_T_ptr = thrust::raw_pointer_cast(d_R_final_T.data());
            thrust::device_vector<T_compute> d_R_final_T_backup;
            T_compute* d_R_final_T_backup_ptr = nullptr;

            // 最终R的输出缓冲区
            thrust::device_vector<T_compute> d_R_true_final(final_rank * final_rank);
            T_compute* d_R_true_final_ptr = thrust::raw_pointer_cast(d_R_true_final.data());
            thrust::device_vector<double> d_R_final_T_double;
            double* d_R_final_T_double_ptr = nullptr;
            thrust::device_vector<double> d_R_final_T_backup_double;
            double* d_R_final_T_backup_double_ptr = nullptr;
            thrust::device_vector<double> d_R_true_final_double;
            double* d_R_true_final_double_ptr = nullptr;
            const double* d_Q_step4_double_verify_ptr = nullptr;
            if constexpr (std::is_same_v<T_compute, float>) {
                if (use_float_mixed_path) {
                    d_R_final_T_double.resize((size_t)n * (size_t)final_rank);
                    d_R_final_T_double_ptr = thrust::raw_pointer_cast(d_R_final_T_double.data());
                    d_R_true_final_double.resize((size_t)final_rank * (size_t)final_rank);
                    d_R_true_final_double_ptr = thrust::raw_pointer_cast(d_R_true_final_double.data());
                    if (skip_final_qr) {
                        BlasCaller<T_compute>::geam(cublasHandle,
                                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                                    (int)n, (int)final_rank,
                                                    &one,
                                                    d_R_proj_ptr, (int)final_rank,
                                                    &zero,
                                                    d_R_final_T_ptr, (int)n,
                                                    d_R_final_T_ptr, (int)n);
                        convert_kernel<float, double><<<(int)((((size_t)n * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_R_final_T_ptr, d_R_final_T_double_ptr, (size_t)n * (size_t)final_rank);
                    } else {
                        BlasCaller<double>::geam(cublasHandle,
                                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                                 (int)n, (int)final_rank,
                                                 &one_d,
                                                 d_R_proj_double_ptr, (int)final_rank,
                                                 &zero_d,
                                                 d_R_final_T_double_ptr, (int)n,
                                                 d_R_final_T_double_ptr, (int)n);
                    }
                } else {
                    BlasCaller<T_compute>::geam(cublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                (int)n, (int)final_rank,
                                                &one,
                                                d_R_proj_ptr, (int)final_rank,
                                                &zero,
                                                d_R_final_T_ptr, (int)n,
                                                d_R_final_T_ptr, (int)n);
                }
            } else {
                BlasCaller<T_compute>::geam(cublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            (int)n, (int)final_rank,
                                            &one,
                                            d_R_proj_ptr, (int)final_rank,
                                            &zero,
                                            d_R_final_T_ptr, (int)n,
                                            d_R_final_T_ptr, (int)n);
            }

            if (debug_mode && !skip_final_qr) {
                if constexpr (std::is_same_v<T_compute, float>) {
                    if (use_float_mixed_path) {
                        d_R_final_T_backup_double = d_R_final_T_double;
                        d_R_final_T_backup_double_ptr = thrust::raw_pointer_cast(d_R_final_T_backup_double.data());
                    } else {
                        d_R_final_T_backup = d_R_final_T;
                        d_R_final_T_backup_ptr = thrust::raw_pointer_cast(d_R_final_T_backup.data());
                    }
                } else {
                    d_R_final_T_backup = d_R_final_T;
                    d_R_final_T_backup_ptr = thrust::raw_pointer_cast(d_R_final_T_backup.data());
                }
            }

            bool step4_use_tsqr_final = false;
            if (!skip_final_qr) {
                // 根据条件选择TSQR或cuSOLVER QR
                // 这里的panelwidth对应的是final_rank，当final_rank <= 64时使用TSQR
                // 默认启用 TSQR-final；可通过 HQB_DISABLE_TSQR=1 关闭。
                bool use_tsqr_final = false;
                bool tsqr_env_ok_final = true;
                if (const char* en = std::getenv("HQB_ENABLE_TSQR")) { tsqr_env_ok_final = (en[0] != '0'); }
                if (const char* dis = std::getenv("HQB_DISABLE_TSQR")) { if (dis[0] != '0') tsqr_env_ok_final = false; }
                if (tsqr_env_ok_final && final_rank <= 64 && (n % final_rank) == 0) {
                    use_tsqr_final = true;
                }
                if constexpr (std::is_same_v<T_compute, float>) {
                    if (use_float_mixed_path) {
                        // 默认 float 精确路径统一走 double-QR，避免在 Step 4 再次降回 float。
                        use_tsqr_final = false;
                    }
                }
                if (use_tsqr_final) {
                    printf("[Info] Using TSQR for final R decomposition (final_rank=%zu <= 64 and n%%final_rank==0)\n", final_rank);
                } else {
                    if constexpr (std::is_same_v<T_compute, float>) {
                        if (use_float_mixed_path) {
                            printf("[Info] Using cuSOLVER QR for final R decomposition on float path (keep Step4 intermediates in double)\n");
                        } else {
                            printf("[Info] Using cuSOLVER QR for final R decomposition on float fast path\n");
                        }
                    } else {
                        printf("[Info] Using cuSOLVER QR for final R decomposition (final_rank=%zu > 64 or shape mismatch)\n", final_rank);
                    }
                }

                step4_use_tsqr_final = use_tsqr_final;
                if (use_tsqr_final) {
                    // 使用TSQR进行QR分解，TSQR内部包含Q构建，时间计入总时间
                    // 对于TSQR，需要对矩阵维度进行对齐以避免奇数维度问题
                    int n_aligned = n;
                    int final_rank_aligned = final_rank;

                    // 对齐到16的倍数（TSQR要求）
                    if (n % 16 != 0) {
                        n_aligned = ((n + 15) / 16) * 16;
                    }
                    if (final_rank % 16 != 0) {
                        final_rank_aligned = ((final_rank + 15) / 16) * 16;
                    }

                    if (n_aligned != n || final_rank_aligned != final_rank) {
                        printf("[Info] TSQR alignment: n %d->%d, final_rank %zu->%d\n",
                               n, n_aligned, final_rank, final_rank_aligned);
                    }

                    thrust::device_vector<T_compute> d_R_final_T_aligned(n_aligned * final_rank_aligned);
                    T_compute* d_R_final_T_aligned_ptr = thrust::raw_pointer_cast(d_R_final_T_aligned.data());

                    // 将原始数据复制到对齐的缓冲区
                    if (n_aligned != n || final_rank_aligned != final_rank) {
                        cudaMemset(d_R_final_T_aligned_ptr, 0, n_aligned * final_rank_aligned * sizeof(T_compute));
                        for (int j = 0; j < final_rank; j++) {
                            cudaMemcpy(d_R_final_T_aligned_ptr + j * n_aligned,
                                       d_R_final_T_ptr + j * n,
                                       n * sizeof(T_compute),
                                       cudaMemcpyDeviceToDevice);
                        }
                    } else {
                        cudaMemcpy(d_R_final_T_aligned_ptr, d_R_final_T_ptr, n * final_rank * sizeof(T_compute), cudaMemcpyDeviceToDevice);
                    }

                    thrust::device_vector<T_compute> d_work_tsqr_final(n_aligned * final_rank_aligned * 4); // 为TSQR分配工作空间
                    T_compute* d_work_tsqr_final_ptr = thrust::raw_pointer_cast(d_work_tsqr_final.data());

                    tsqr<T_compute>(cublasHandle,
                                    n_aligned, final_rank_aligned,
                                    d_R_final_T_aligned_ptr, n_aligned,
                                    d_R_true_final_ptr, final_rank,
                                    d_work_tsqr_final_ptr, n_aligned);
                    // 如果进行了对齐，则需要提取实际的 final_rank x final_rank 部分
                    if (final_rank_aligned != final_rank) {
                        // 提取实际的final_rank x final_rank部分
                        thrust::device_vector<T_compute> d_R_temp(final_rank * final_rank);
                        T_compute* d_R_temp_ptr = thrust::raw_pointer_cast(d_R_temp.data());

                        for (int j = 0; j < final_rank; j++) {
                            cudaMemcpy(d_R_temp_ptr + j * final_rank,
                                      d_R_true_final_ptr + j * final_rank_aligned,
                                      final_rank * sizeof(T_compute),
                                      cudaMemcpyDeviceToDevice);
                        }
                        cudaMemcpy(d_R_true_final_ptr, d_R_temp_ptr,
                                  final_rank * final_rank * sizeof(T_compute), cudaMemcpyDeviceToDevice);
                    }
                    float R_compute_time = stopTimer();
                    stage_time_step4_ms = static_cast<double>(R_compute_time);
                    printf("R_final generation (TSQR) finished in %f ms.\n", R_compute_time);

                    // 如果使用了float计算，则进行一次double精度的正交性修正（可选，调试模式下开启）
                    if constexpr (std::is_same_v<T_compute,float>) {
                        if (debug_mode) {
                            thrust::device_vector<double> d_Rd_T((size_t)n * final_rank);
                            double* d_Rd_T_ptr = thrust::raw_pointer_cast(d_Rd_T.data());
                            convert_kernel<float, double><<<(int)((((size_t)n*final_rank)+255)/256), 256, 0, stream>>>(
                                d_R_final_T_ptr, d_Rd_T_ptr, (size_t)n * (size_t)final_rank);
                            int lwork_geqrf_final = 0;
                            cusolverDnDgeqrf_bufferSize(cusolverHandle, (int)n, (int)final_rank, d_Rd_T_ptr, (int)n, &lwork_geqrf_final);
                            thrust::device_vector<double> d_work_geqrf(lwork_geqrf_final);
                            thrust::device_vector<double> d_tau_final((size_t)final_rank);
                            thrust::device_vector<int> d_devInfo_final(1);
                            cusolverDnDgeqrf(cusolverHandle, (int)n, (int)final_rank,
                                             d_Rd_T_ptr, (int)n,
                                             thrust::raw_pointer_cast(d_tau_final.data()),
                                             thrust::raw_pointer_cast(d_work_geqrf.data()), lwork_geqrf_final,
                                             thrust::raw_pointer_cast(d_devInfo_final.data()));
                            // Keep the debug QR pair self-consistent: Step 5 should use the same R
                            // whose Q is later materialized for final verification.
                            thrust::device_vector<double> d_R_true_final_dbg((size_t)final_rank * (size_t)final_rank);
                            dim3 threads_R_extract_dbg(16, 16);
                            dim3 blocks_R_extract_dbg((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                            extract_upper_triangle_kernel<double><<<blocks_R_extract_dbg, threads_R_extract_dbg, 0, stream>>>(
                                thrust::raw_pointer_cast(d_R_true_final_dbg.data()), (int)final_rank,
                                d_Rd_T_ptr, (int)n, (int)final_rank, (int)final_rank);
                            convert_kernel<double, float><<<(int)((((size_t)final_rank * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                thrust::raw_pointer_cast(d_R_true_final_dbg.data()), d_R_true_final_ptr, (size_t)final_rank * (size_t)final_rank);
                            int lwork_orgqr_final = 0;
                            cusolverDnDorgqr_bufferSize(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                        d_Rd_T_ptr, (int)n, thrust::raw_pointer_cast(d_tau_final.data()), &lwork_orgqr_final);
                            thrust::device_vector<double> d_work_orgqr(lwork_orgqr_final);
                            cusolverDnDorgqr(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                             d_Rd_T_ptr, (int)n,
                                             thrust::raw_pointer_cast(d_tau_final.data()),
                                             thrust::raw_pointer_cast(d_work_orgqr.data()), lwork_orgqr_final,
                                             thrust::raw_pointer_cast(d_devInfo_final.data()));
                            d_R_final_T_double.resize((size_t)n * (size_t)final_rank);
                            d_R_final_T_double_ptr = thrust::raw_pointer_cast(d_R_final_T_double.data());
                            cudaMemcpyAsync(d_R_final_T_double_ptr, d_Rd_T_ptr,
                                            sizeof(double) * (size_t)n * (size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            d_Q_step4_double_verify_ptr = d_R_final_T_double_ptr;
                            convert_kernel<double, float><<<(int)((((size_t)n*final_rank)+255)/256), 256, 0, stream>>>(
                                d_Rd_T_ptr, d_R_final_T_ptr, (size_t)n * (size_t)final_rank);
                        }
                    }
                } else {
                    // cusolver QR
                    cudaEvent_t R_start, R_end;
                    cudaEventCreate(&R_start);
                    cudaEventCreate(&R_end);
                    cudaEventRecord(R_start, stream);

                    if constexpr (std::is_same_v<T_compute,float>) {
                        if (use_float_mixed_path) {
                        // 使用 double 中间量执行 Step 4 QR，并仅在输出阶段回写到 float。
                        double* d_Rd_T_ptr = d_R_final_T_double_ptr;
                        int lwork_geqrf_final = 0;
                        cusolverDnDgeqrf_bufferSize(cusolverHandle, (int)n, (int)final_rank, d_Rd_T_ptr, (int)n, &lwork_geqrf_final);
                        thrust::device_vector<double> d_work_geqrf(lwork_geqrf_final);
                        thrust::device_vector<double> d_tau_final((size_t)final_rank);
                        thrust::device_vector<int> d_devInfo_final(1);
                        cusolverDnDgeqrf(cusolverHandle, (int)n, (int)final_rank,
                                         d_Rd_T_ptr, (int)n,
                                         thrust::raw_pointer_cast(d_tau_final.data()),
                                         thrust::raw_pointer_cast(d_work_geqrf.data()), lwork_geqrf_final,
                                         thrust::raw_pointer_cast(d_devInfo_final.data()));

                        // 提取 R，并保留 double 版本供后续 Step 5 直接使用。
                        dim3 threads_R_extract(16, 16);
                        dim3 blocks_R_extract((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                        extract_upper_triangle_kernel<double><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
                            d_R_true_final_double_ptr, (int)final_rank, d_Rd_T_ptr, (int)n, (int)final_rank, (int)final_rank);
                        convert_kernel<double, float><<<(int)((((size_t)final_rank*final_rank)+255)/256), 256, 0, stream>>>(
                            d_R_true_final_double_ptr, d_R_true_final_ptr, (size_t)final_rank * (size_t)final_rank);

                        cudaEventRecord(R_end, stream);
                        cudaEventSynchronize(R_end);
                        float R_compute_time;
                        cudaEventElapsedTime(&R_compute_time, R_start, R_end);
                        cudaEventDestroy(R_start);
                        cudaEventDestroy(R_end);
                        stage_time_step4_ms = static_cast<double>(R_compute_time);
                        printf("R_final generation (double-QR) finished in %f ms.\n", R_compute_time);

                        // 构建Q（仅调试模式）
                        if (debug_mode) {
                            cudaEvent_t qbuild_start, qbuild_end;
                            cudaEventCreate(&qbuild_start);
                            cudaEventCreate(&qbuild_end);
                            cudaEventRecord(qbuild_start, stream);
                            int lwork_orgqr_final = 0;
                            cusolverDnDorgqr_bufferSize(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                        d_Rd_T_ptr, (int)n, thrust::raw_pointer_cast(d_tau_final.data()), &lwork_orgqr_final);
                            thrust::device_vector<double> d_work_orgqr(lwork_orgqr_final);
                            cusolverDnDorgqr(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                             d_Rd_T_ptr, (int)n,
                                             thrust::raw_pointer_cast(d_tau_final.data()),
                                             thrust::raw_pointer_cast(d_work_orgqr.data()), lwork_orgqr_final,
                                             thrust::raw_pointer_cast(d_devInfo_final.data()));
                            cudaEventRecord(qbuild_end, stream);
                            cudaEventSynchronize(qbuild_end);
                            float qbuild_ms = 0.0f;
                            cudaEventElapsedTime(&qbuild_ms, qbuild_start, qbuild_end);
                            stage_time_step4_qbuild_ms += static_cast<double>(qbuild_ms);
                            cudaEventDestroy(qbuild_start);
                            cudaEventDestroy(qbuild_end);
                            d_Q_step4_double_verify_ptr = d_Rd_T_ptr;
                            convert_kernel<double, float><<<(int)((((size_t)n*final_rank)+255)/256), 256, 0, stream>>>(
                                d_Rd_T_ptr, d_R_final_T_ptr, (size_t)n * (size_t)final_rank);
                        }
                        } else {
                            int lwork_geqrf_final = 0;
                            SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, (int)n, (int)final_rank,
                                                                     d_R_final_T_ptr, (int)n, &lwork_geqrf_final);
                            thrust::device_vector<T_compute> d_work_geqrf(lwork_geqrf_final);
                            thrust::device_vector<T_compute> d_tau_final((size_t)final_rank);
                            thrust::device_vector<int> d_devInfo_final(1);
                            SolverCaller<T_compute>::geqrf(cusolverHandle, (int)n, (int)final_rank,
                                                          d_R_final_T_ptr, (int)n,
                                                          thrust::raw_pointer_cast(d_tau_final.data()),
                                                          thrust::raw_pointer_cast(d_work_geqrf.data()), lwork_geqrf_final,
                                                          thrust::raw_pointer_cast(d_devInfo_final.data()));
                            dim3 threads_R_extract(16, 16);
                            dim3 blocks_R_extract((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                            extract_upper_triangle_kernel<T_compute><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
                                d_R_true_final_ptr, (int)final_rank, d_R_final_T_ptr, (int)n, (int)final_rank, (int)final_rank);

                            cudaEventRecord(R_end, stream);
                            cudaEventSynchronize(R_end);
                            float R_compute_time;
                            cudaEventElapsedTime(&R_compute_time, R_start, R_end);
                            cudaEventDestroy(R_start);
                            cudaEventDestroy(R_end);
                            stage_time_step4_ms = static_cast<double>(R_compute_time);
                            printf("R_final generation finished successfully in %f ms.\n", R_compute_time);

                            if (debug_mode) {
                                cudaEvent_t qbuild_start, qbuild_end;
                                cudaEventCreate(&qbuild_start);
                                cudaEventCreate(&qbuild_end);
                                cudaEventRecord(qbuild_start, stream);
                                int lwork_orgqr_final = 0;
                                SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                                         d_R_final_T_ptr, (int)n, thrust::raw_pointer_cast(d_tau_final.data()), &lwork_orgqr_final);
                                thrust::device_vector<T_compute> d_work_orgqr(lwork_orgqr_final);
                                SolverCaller<T_compute>::orgqr(cusolverHandle,
                                                             (int)n, (int)final_rank, (int)final_rank,
                                                             d_R_final_T_ptr, (int)n,
                                                             thrust::raw_pointer_cast(d_tau_final.data()),
                                                             thrust::raw_pointer_cast(d_work_orgqr.data()),
                                                             lwork_orgqr_final,
                                                             thrust::raw_pointer_cast(d_devInfo_final.data()));
                                cudaEventRecord(qbuild_end, stream);
                                cudaEventSynchronize(qbuild_end);
                                float qbuild_ms = 0.0f;
                                cudaEventElapsedTime(&qbuild_ms, qbuild_start, qbuild_end);
                                stage_time_step4_qbuild_ms += static_cast<double>(qbuild_ms);
                                cudaEventDestroy(qbuild_start);
                                cudaEventDestroy(qbuild_end);
                            }
                        }
                    } else {
                        int lwork_geqrf_final = 0;
                        SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, (int)n, (int)final_rank,
                                                                 d_R_final_T_ptr, (int)n, &lwork_geqrf_final);
                        thrust::device_vector<T_compute> d_work_geqrf(lwork_geqrf_final);
                        thrust::device_vector<T_compute> d_tau_final((size_t)final_rank);
                        thrust::device_vector<int> d_devInfo_final(1);
                        // QR 
                        SolverCaller<T_compute>::geqrf(cusolverHandle, (int)n, (int)final_rank,
                                                      d_R_final_T_ptr, (int)n,
                                                      thrust::raw_pointer_cast(d_tau_final.data()),
                                                      thrust::raw_pointer_cast(d_work_geqrf.data()), lwork_geqrf_final,
                                                      thrust::raw_pointer_cast(d_devInfo_final.data()));
                        // Extract R
                        dim3 threads_R_extract(16, 16);
                        dim3 blocks_R_extract((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                        extract_upper_triangle_kernel<T_compute><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
                            d_R_true_final_ptr, (int)final_rank, d_R_final_T_ptr, (int)n, (int)final_rank, (int)final_rank);

                        cudaEventRecord(R_end, stream);
                        cudaEventSynchronize(R_end);
                        float R_compute_time;
                        cudaEventElapsedTime(&R_compute_time, R_start, R_end);
                        cudaEventDestroy(R_start);
                        cudaEventDestroy(R_end);
                        stage_time_step4_ms = static_cast<double>(R_compute_time);
                        printf("R_final generation finished successfully in %f ms.\n", R_compute_time);

                        // Q (debug)
                        if (debug_mode) {
                            cudaEvent_t qbuild_start, qbuild_end;
                            cudaEventCreate(&qbuild_start);
                            cudaEventCreate(&qbuild_end);
                            cudaEventRecord(qbuild_start, stream);
                            int lwork_orgqr_final = 0;
                            SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                                     d_R_final_T_ptr, (int)n, thrust::raw_pointer_cast(d_tau_final.data()), &lwork_orgqr_final);
                            thrust::device_vector<T_compute> d_work_orgqr(lwork_orgqr_final);
                            SolverCaller<T_compute>::orgqr(cusolverHandle,
                                                         (int)n, (int)final_rank, (int)final_rank,
                                                         d_R_final_T_ptr, (int)n,
                                                         thrust::raw_pointer_cast(d_tau_final.data()),
                                                         thrust::raw_pointer_cast(d_work_orgqr.data()),
                                                         lwork_orgqr_final,
                                                         thrust::raw_pointer_cast(d_devInfo_final.data()));
                            cudaEventRecord(qbuild_end, stream);
                            cudaEventSynchronize(qbuild_end);
                            float qbuild_ms = 0.0f;
                            cudaEventElapsedTime(&qbuild_ms, qbuild_start, qbuild_end);
                            stage_time_step4_qbuild_ms += static_cast<double>(qbuild_ms);
                            cudaEventDestroy(qbuild_start);
                            cudaEventDestroy(qbuild_end);
                        }
                    }
                }
            } else {
                // 直接提取 R_final 的上三角部分供后续 SVD 使用
                dim3 threads_R_extract(16, 16);
                dim3 blocks_R_extract((int)((final_rank + 15) / 16), (int)((final_rank + 15) / 16));
                extract_upper_triangle_kernel<T_compute><<<blocks_R_extract, threads_R_extract, 0, stream>>>(
                    d_R_true_final_ptr, (int)final_rank, d_R_proj_ptr, (int)final_rank, (int)final_rank, (int)final_rank);
                if constexpr (std::is_same_v<T_compute, float>) {
                    if (use_float_mixed_path) {
                        convert_kernel<float, double><<<(int)((((size_t)final_rank * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_R_true_final_ptr, d_R_true_final_double_ptr, (size_t)final_rank * (size_t)final_rank);
                    }
                }
                printf("[Info] Final QR bypassed: using existing upper-triangular R block.\n");
            }

            // QR分解后向误差验证
            if (debug_mode && !skip_final_qr && !step4_use_tsqr_final) {
                printf("=== Step 4: Verifying QR Decomposition (R_final_T ≈ Q*R_true_final) ===\n");
                // 设置数学模式为默认以避免潜在的数值问题
                cublasMath_t prev_math_mode; cublasGetMathMode(cublasHandle, &prev_math_mode);
                cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);

                if constexpr (std::is_same_v<T_compute, float>) {
                    if (use_float_mixed_path) {
                        double q_ortho_error = verify_orthogonality<double>(cublasHandle, n, final_rank, d_R_final_T_double_ptr, n);
                        printf("Q orthogonality error ||Q^T*Q - I||_F = %.12e\n", q_ortho_error);

                        thrust::device_vector<double> d_R_reconstructed((size_t)n * (size_t)final_rank);
                        double* d_R_reconstructed_ptr = thrust::raw_pointer_cast(d_R_reconstructed.data());
                        BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, final_rank, final_rank,
                                                 &one_d, d_R_final_T_double_ptr, n,
                                                 d_R_true_final_double_ptr, final_rank,
                                                 &zero_d, d_R_reconstructed_ptr, n);
                        BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                 n, final_rank,
                                                 &one_d, d_R_final_T_backup_double_ptr, n,
                                                 &minus_one_d, d_R_reconstructed_ptr, n,
                                                 d_R_reconstructed_ptr, n);

                        double reconstruction_error = 0.0, norm_R_original = 0.0;
                        cublasDnrm2(cublasHandle, n * final_rank, d_R_reconstructed_ptr, 1, &reconstruction_error);
                        cublasDnrm2(cublasHandle, n * final_rank, d_R_final_T_backup_double_ptr, 1, &norm_R_original);
                        printf("Reconstruction error ||R_final_T - Q*R||_F = %.12e\n", reconstruction_error);
                        double relative_error = (norm_R_original > 1e-18) ? (reconstruction_error / norm_R_original) : reconstruction_error;
                        printf("Relative reconstruction error ||R_final_T - Q*R||_F / ||R_final_T||_F = %.12e\n", relative_error);
                    } else {
                        T_compute q_ortho_error = verify_orthogonality<T_compute>(cublasHandle, n, final_rank, d_R_final_T_ptr, n);
                        printf("Q orthogonality error ||Q^T*Q - I||_F = %.12e\n", (double)q_ortho_error);

                        thrust::device_vector<T_compute> d_R_reconstructed(n * final_rank);
                        T_compute* d_R_reconstructed_ptr = thrust::raw_pointer_cast(d_R_reconstructed.data());
                        BlasCaller<T_compute>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                   n, final_rank, final_rank,
                                                   &one, d_R_final_T_ptr, n,
                                                   d_R_true_final_ptr, final_rank,
                                                   &zero, d_R_reconstructed_ptr, n);
                        BlasCaller<T_compute>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                   n, final_rank,
                                                   &one, d_R_final_T_backup_ptr, n,
                                                   &minus_one, d_R_reconstructed_ptr, n,
                                                   d_R_reconstructed_ptr, n);
                        T_compute reconstruction_error = 0, norm_R_original = 0;
                        cublasSnrm2(cublasHandle, n * final_rank, (const float*)d_R_reconstructed_ptr, 1, (float*)&reconstruction_error);
                        cublasSnrm2(cublasHandle, n * final_rank, (const float*)d_R_final_T_backup_ptr, 1, (float*)&norm_R_original);
                        printf("Reconstruction error ||R_final_T - Q*R||_F = %.12e\n", (double)reconstruction_error);
                        T_compute relative_error = (norm_R_original > 1e-12f) ? (reconstruction_error / norm_R_original) : reconstruction_error;
                        printf("Relative reconstruction error ||R_final_T - Q*R||_F / ||R_final_T||_F = %.12e\n", (double)relative_error);
                    }
                } else {
                    const T_compute one = T_compute(1.0);
                    const T_compute zero = T_compute(0.0);
                    const T_compute minus_one = T_compute(-1.0);

                    T_compute q_ortho_error = verify_orthogonality<T_compute>(cublasHandle, n, final_rank, d_R_final_T_ptr, n);
                    printf("Q orthogonality error ||Q^T*Q - I||_F = %.12e\n", (double)q_ortho_error);

                    // ||R_final_T - Q*R_true_final||_F
                    thrust::device_vector<T_compute> d_R_reconstructed(n * final_rank);
                    T_compute* d_R_reconstructed_ptr = thrust::raw_pointer_cast(d_R_reconstructed.data());

                    // Q * R_true_final
                    BlasCaller<T_compute>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, final_rank, final_rank,
                                               &one, d_R_final_T_ptr, n,
                                               d_R_true_final_ptr, final_rank,
                                               &zero, d_R_reconstructed_ptr, n);

                    // R_final_T_backup - Q*R_true_final
                    BlasCaller<T_compute>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                               n, final_rank,
                                               &one, d_R_final_T_backup_ptr, n,
                                               &minus_one, d_R_reconstructed_ptr, n,
                                               d_R_reconstructed_ptr, n);

                    T_compute reconstruction_error = 0, norm_R_original = 0;
                    if constexpr (std::is_same<T_compute, float>::value) {
                        cublasSnrm2(cublasHandle, n * final_rank, (const float*)d_R_reconstructed_ptr, 1, (float*)&reconstruction_error);
                        cublasSnrm2(cublasHandle, n * final_rank, (const float*)d_R_final_T_backup_ptr, 1, (float*)&norm_R_original);
                    } else {
                        cublasDnrm2(cublasHandle, n * final_rank, (const double*)d_R_reconstructed_ptr, 1, (double*)&reconstruction_error);
                        cublasDnrm2(cublasHandle, n * final_rank, (const double*)d_R_final_T_backup_ptr, 1, (double*)&norm_R_original);
                    }

                    printf("Reconstruction error ||R_final_T - Q*R||_F = %.12e\n", (double)reconstruction_error);
                    T_compute relative_error = (norm_R_original > 1e-12) ? (reconstruction_error / norm_R_original) : reconstruction_error;
                    printf("Relative reconstruction error ||R_final_T - Q*R||_F / ||R_final_T||_F = %.12e\n", (double)relative_error);
                }
                // Restore math mode
                cublasSetMathMode(cublasHandle, prev_math_mode);
            } else if (debug_mode) {
                if (skip_final_qr) {
                    printf("[Info] Step 4 verification skipped because final QR was bypassed.\n");
                } else if (step4_use_tsqr_final) {
                    printf("[Info] Step 4 verification skipped on TSQR-final path (Q factor is not materialized in d_R_final_T).\n");
                }
            }
            // 对R_final做 SVD 求解奇异值，奇异向量(double：直接 SVD；float：基于 EVD 的混合精度 SVD)
            printf("=== Step 5: SVD of small R (%zu x %zu) ===\n", final_rank, final_rank);
            startTimer();
            if constexpr (std::is_same_v<T_compute, double>) {
                // 直接在 double 精度下，用 cuSOLVER 对 R 做标准 SVD
                thrust::device_vector<double> d_R_copy(final_rank * final_rank);
                double* d_R_copy_ptr = thrust::raw_pointer_cast(d_R_copy.data());
                cudaMemcpyAsync(d_R_copy_ptr, d_R_true_final_ptr, sizeof(double) * final_rank * final_rank,
                                cudaMemcpyDeviceToDevice, stream);

                thrust::device_vector<double> d_S(final_rank);
                thrust::device_vector<double> d_U(final_rank * final_rank);
                thrust::device_vector<double> d_VT(final_rank * final_rank);
                thrust::device_vector<int> d_info(1);

                int lwork = 0;
                cusolverDnDgesvd_bufferSize(cusolverHandle, (int)final_rank, (int)final_rank, &lwork);
                thrust::device_vector<double> d_work(lwork);
                char jobu = need_vectors() ? 'A' : 'N';
                char jobvt = need_vectors() ? 'A' : 'N';
                cusolverStatus_t st = cusolverDnDgesvd(cusolverHandle,
                                                       jobu, jobvt,
                                                       (int)final_rank, (int)final_rank,
                                                       d_R_copy_ptr, (int)final_rank,
                                                       thrust::raw_pointer_cast(d_S.data()),
                                                       (jobu=='N' ? nullptr : thrust::raw_pointer_cast(d_U.data())), (int)final_rank,
                                                       (jobvt=='N' ? nullptr : thrust::raw_pointer_cast(d_VT.data())), (int)final_rank,
                                                       thrust::raw_pointer_cast(d_work.data()), lwork,
                                                       nullptr,
                                                       thrust::raw_pointer_cast(d_info.data()));
                if (st != CUSOLVER_STATUS_SUCCESS) {
                    printf("[Warn] cusolverDnDgesvd failed with status=%d\n", (int)st);
                }
                int info_host = 0; cudaMemcpy(&info_host, thrust::raw_pointer_cast(d_info.data()), sizeof(int), cudaMemcpyDeviceToHost);
                if (info_host != 0) printf("[Warn] Dgesvd info=%d\n", info_host);

                // 若只需要奇异值，则跳过向量的后处理
                thrust::device_vector<double> d_V; thrust::device_vector<double> d_V_tmp;
                if (need_vectors()) {
                    // 将 V^T 转置为 V
                    d_V.resize((size_t)final_rank * (size_t)final_rank);
                    d_V_tmp.resize((size_t)final_rank * (size_t)final_rank);
                    BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                             (int)final_rank, (int)final_rank,
                                             &one_d,
                                             thrust::raw_pointer_cast(d_VT.data()), (int)final_rank,
                                             &zero_d,
                                             thrust::raw_pointer_cast(d_VT.data()), (int)final_rank,
                                             thrust::raw_pointer_cast(d_V_tmp.data()), (int)final_rank);
                    cudaMemcpyAsync(thrust::raw_pointer_cast(d_V.data()),
                                    thrust::raw_pointer_cast(d_V_tmp.data()),
                                    sizeof(double) * (size_t)final_rank * final_rank,
                                    cudaMemcpyDeviceToDevice, stream);
                }

                if (debug_mode && need_vectors()) {
                    // ||R - U*S*V^T||_F
                    thrust::device_vector<double> d_US_dbg(final_rank * final_rank);
                    cudaMemcpyAsync(thrust::raw_pointer_cast(d_US_dbg.data()), thrust::raw_pointer_cast(d_U.data()), sizeof(double)*final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);
                    {
                        dim3 th(16,16); dim3 gr((int)((final_rank + th.x - 1)/th.x), (int)((final_rank + th.y - 1)/th.y));
                        scale_columns_by_vector_inplace_double<<<gr, th, 0, stream>>>(
                            thrust::raw_pointer_cast(d_US_dbg.data()), thrust::raw_pointer_cast(d_S.data()), (int)final_rank, (int)final_rank, (int)final_rank);
                    }
                    thrust::device_vector<double> d_R_recon(final_rank * final_rank);
                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                             (int)final_rank, (int)final_rank, (int)final_rank,
                                             &one_d,
                                             thrust::raw_pointer_cast(d_US_dbg.data()), (int)final_rank,
                                             thrust::raw_pointer_cast(d_V.data()), (int)final_rank,
                                             &zero_d,
                                             thrust::raw_pointer_cast(d_R_recon.data()), (int)final_rank);
                    // R - R_recon
                    BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             (int)final_rank, (int)final_rank,
                                             &one_d, d_R_true_final_ptr, (int)final_rank,
                                             &minus_one_d, thrust::raw_pointer_cast(d_R_recon.data()), (int)final_rank,
                                             thrust::raw_pointer_cast(d_R_recon.data()), (int)final_rank);
                    double r_res = 0.0;
                    cublasDnrm2(cublasHandle, (int)(final_rank*final_rank), thrust::raw_pointer_cast(d_R_recon.data()), 1, &r_res);
                    printf("R residual ||R - U*S*V^T||_F = %.6e\n", r_res);

                    // U/V的正交性
                    double u_ortho = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank, thrust::raw_pointer_cast(d_U.data()), (int)final_rank);
                    double v_ortho = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank, thrust::raw_pointer_cast(d_V.data()), (int)final_rank);
                    printf("U_R orthogonality ||U^T U - I||_F = %.6e\n", u_ortho);
                    printf("V_R orthogonality ||V^T V - I||_F = %.6e\n", v_ortho);

                    
                }

                // 组装并验证 A 重构（仅 debug）：A ≈ Q_basis * R_final，且 R_final_T ≈ Q_step4 * R。
                // 若对 R 做 SVD：R = U*S*V^T，则 R^T = V*S*U^T。
                // 因此 A_hat 应构造成：A_hat = (Q_basis*V) * S * (Q_step4*U)^T
                if (debug_mode) {
                    const int m_debug = (int)m_effective;
                    const int ld_q_debug = ld_Q_basis_effective;
                    const int ld_a_debug = ld_A_effective;
                    const double* d_A_effective_raw_double = nullptr;
                    const double* d_Q_basis_effective_double = nullptr;
                    thrust::device_vector<double> d_A_effective_tmp_double;
                    thrust::device_vector<double> d_Q_basis_tmp_double;
                    if constexpr (std::is_same_v<T_compute, double>) {
                        d_A_effective_raw_double = d_A_effective_raw;
                        d_Q_basis_effective_double = d_Q_basis_effective_ptr;
                    } else {
                        d_A_effective_tmp_double.resize((size_t)m_effective * (size_t)n);
                        d_Q_basis_tmp_double.resize((size_t)m_effective * (size_t)final_rank);
                        dim3 tA(16,16);
                        dim3 bA((int)((n + tA.x - 1)/tA.x), (int)(((int)m_effective + tA.y - 1)/tA.y));
                        convert_kernel_strided<T_compute, double><<<bA, tA, 0, stream>>>(
                            d_A_effective_raw, (size_t)ld_a_debug,
                            thrust::raw_pointer_cast(d_A_effective_tmp_double.data()), (size_t)m_effective,
                            (size_t)m_effective, (size_t)n);
                        dim3 tQd(16,16);
                        dim3 bQd((int)((final_rank + tQd.x - 1)/tQd.x), (int)(((int)m_effective + tQd.y - 1)/tQd.y));
                        convert_kernel_strided<T_compute, double><<<bQd, tQd, 0, stream>>>(
                            d_Q_basis_effective_ptr, (size_t)ld_q_debug,
                            thrust::raw_pointer_cast(d_Q_basis_tmp_double.data()), (size_t)m_effective,
                            (size_t)m_effective, (size_t)final_rank);
                        CUDA_CHECK(cudaGetLastError());
                        d_A_effective_raw_double = thrust::raw_pointer_cast(d_A_effective_tmp_double.data());
                        d_Q_basis_effective_double = thrust::raw_pointer_cast(d_Q_basis_tmp_double.data());
                    }
                    thrust::device_vector<double> d_Rd_T_dbg((size_t)n * (size_t)final_rank);
                    double* d_Rd_T_dbg_ptr = thrust::raw_pointer_cast(d_Rd_T_dbg.data());
                    cudaMemcpyAsync(d_Rd_T_dbg_ptr, d_R_final_T_ptr,
                                    sizeof(double)*(size_t)n*(size_t)final_rank,
                                    cudaMemcpyDeviceToDevice, stream);
                    int lwork_geqrf_dbg = 0;
                    SolverCaller<double>::geqrf_bufferSize(cusolverHandle, (int)n, (int)final_rank,
                                                           d_Rd_T_dbg_ptr, (int)n, &lwork_geqrf_dbg);
                    thrust::device_vector<double> d_work_geqrf_dbg(lwork_geqrf_dbg);
                    thrust::device_vector<double> d_tau_dbg((size_t)final_rank);
                    thrust::device_vector<int> d_devInfo_dbg(1);
                    SolverCaller<double>::geqrf(cusolverHandle, (int)n, (int)final_rank,
                                                d_Rd_T_dbg_ptr, (int)n,
                                                thrust::raw_pointer_cast(d_tau_dbg.data()),
                                                thrust::raw_pointer_cast(d_work_geqrf_dbg.data()), lwork_geqrf_dbg,
                                                thrust::raw_pointer_cast(d_devInfo_dbg.data()));
                    int lwork_orgqr_dbg = 0;
                    SolverCaller<double>::orgqr_bufferSize(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                           d_Rd_T_dbg_ptr, (int)n, thrust::raw_pointer_cast(d_tau_dbg.data()), &lwork_orgqr_dbg);
                    thrust::device_vector<double> d_work_orgqr_dbg(lwork_orgqr_dbg);
                    SolverCaller<double>::orgqr(cusolverHandle, (int)n, (int)final_rank, (int)final_rank,
                                                d_Rd_T_dbg_ptr, (int)n,
                                                thrust::raw_pointer_cast(d_tau_dbg.data()),
                                                thrust::raw_pointer_cast(d_work_orgqr_dbg.data()), lwork_orgqr_dbg,
                                                thrust::raw_pointer_cast(d_devInfo_dbg.data()));
                    double* d_Q_step4_ptr = d_Rd_T_dbg_ptr; // now holds Q
                    // VS = V * diag(S)
                    thrust::device_vector<double> d_VS_assem((size_t)final_rank * final_rank);
                    cudaMemcpyAsync(thrust::raw_pointer_cast(d_VS_assem.data()), thrust::raw_pointer_cast(d_V.data()), sizeof(double)*final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);
                    {
                        dim3 th(16,16); dim3 gr((int)((final_rank + th.x - 1)/th.x), (int)((final_rank + th.y - 1)/th.y));
                        scale_columns_by_vector_inplace_double<<<gr, th, 0, stream>>>(
                            thrust::raw_pointer_cast(d_VS_assem.data()), thrust::raw_pointer_cast(d_S.data()), (int)final_rank, (int)final_rank, (int)final_rank);
                    }
                    // Q_basis * VS
                    thrust::device_vector<double> d_QVS((size_t)m_effective * final_rank);
                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             m_debug, (int)final_rank, (int)final_rank,
                                             &one_d,
                                             d_Q_basis_effective_double, ld_q_debug,
                                             thrust::raw_pointer_cast(d_VS_assem.data()), (int)final_rank,
                                             &zero_d,
                                             thrust::raw_pointer_cast(d_QVS.data()), m_debug);
                    // Q4U = Q_step4 * U
                    thrust::device_vector<double> d_Q4U((size_t)n * final_rank);
                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             (int)n, (int)final_rank, (int)final_rank,
                                             &one_d,
                                             d_Q_step4_ptr, (int)n,
                                             thrust::raw_pointer_cast(d_U.data()), (int)final_rank,
                                             &zero_d,
                                             thrust::raw_pointer_cast(d_Q4U.data()), (int)n);
                    {
                    thrust::device_vector<double> d_A_rec((size_t)m_effective * (size_t)n);
                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                             m_debug, (int)n, (int)final_rank,
                                             &one_d,
                                             thrust::raw_pointer_cast(d_QVS.data()), m_debug,
                                             thrust::raw_pointer_cast(d_Q4U.data()), (int)n,
                                             &zero_d,
                                             thrust::raw_pointer_cast(d_A_rec.data()), m_debug);
                    BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                             m_debug, (int)n,
                                             &one_d, d_A_effective_raw_double, ld_a_debug,
                                             &minus_one_d, thrust::raw_pointer_cast(d_A_rec.data()), m_debug,
                                             thrust::raw_pointer_cast(d_A_rec.data()), m_debug);
                    double errA=0.0, normA=0.0;
                    cublasDnrm2(cublasHandle, (int)(m_effective*n), thrust::raw_pointer_cast(d_A_rec.data()), 1, &errA);
                    cublasDnrm2(cublasHandle, (int)(m_effective*n), d_A_effective_raw_double, 1, &normA);
                    printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %e\n", errA);
                    printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n", (normA>1e-18? errA/normA: errA));
                    }
                }
                else if (debug_mode) {
                    printf("[Info] Skipping debug reconstruction via Q_step4 because final QR was bypassed.\n");
                }

            } else {
                if constexpr (std::is_same_v<T_storage, __half> || std::is_same_v<T_storage, __nv_bfloat16>) {
                    thrust::device_vector<float> d_Rd(final_rank * final_rank);
                    float* d_Rd_ptr = thrust::raw_pointer_cast(d_Rd.data());
                    thrust::device_vector<float> d_R_T((size_t)final_rank * (size_t)final_rank);
                    BlasCaller<float>::geam(cublasHandle,
                                            CUBLAS_OP_T, CUBLAS_OP_N,
                                            (int)final_rank, (int)final_rank,
                                            &one,
                                            d_R_true_final_ptr, (int)final_rank,
                                            &zero,
                                            thrust::raw_pointer_cast(d_R_T.data()), (int)final_rank,
                                            thrust::raw_pointer_cast(d_R_T.data()), (int)final_rank);
                    cudaMemcpyAsync(d_Rd_ptr, thrust::raw_pointer_cast(d_R_T.data()),
                                    sizeof(float) * (size_t)final_rank * (size_t)final_rank,
                                    cudaMemcpyDeviceToDevice, stream);
                    thrust::device_vector<float> d_G(final_rank * final_rank);
                    float* d_G_ptr = thrust::raw_pointer_cast(d_G.data());
                    bool gram_is_RtR = skip_final_qr || (m > n);
                    if (gram_is_RtR) {
                        cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                    (int)final_rank, (int)final_rank,
                                    &one, d_Rd_ptr, (int)final_rank,
                                    &zero, d_G_ptr, (int)final_rank);
                    } else {
                        cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                    (int)final_rank, (int)final_rank,
                                    &one, d_Rd_ptr, (int)final_rank,
                                    &zero, d_G_ptr, (int)final_rank);
                    }
                    thrust::device_vector<float> d_eig(final_rank);
                    thrust::device_vector<int> d_info(1);
                    size_t lwork = 0, lhwork = 0;
                    auto jobz = need_vectors() ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
                    cusolverDnXsyevd_bufferSize(cusolverHandle, nullptr, jobz,
                                                CUBLAS_FILL_MODE_LOWER,
                                                (int)final_rank,
                                                CUDA_R_32F,
                                                d_G_ptr, (int)final_rank,
                                                CUDA_R_32F,
                                                thrust::raw_pointer_cast(d_eig.data()),
                                                CUDA_R_32F,
                                                &lwork, &lhwork);
                    thrust::device_vector<float> d_work(lwork);
                    void* hwork = nullptr; if (lhwork > 0) hwork = malloc(lhwork);
                    cusolverStatus_t st = cusolverDnXsyevd(cusolverHandle, nullptr, jobz,
                                                           CUBLAS_FILL_MODE_LOWER,
                                                           (int)final_rank,
                                                           CUDA_R_32F,
                                                           d_G_ptr, (int)final_rank,
                                                           CUDA_R_32F,
                                                           thrust::raw_pointer_cast(d_eig.data()),
                                                           CUDA_R_32F,
                                                           thrust::raw_pointer_cast(d_work.data()), (size_t)lwork,
                                                           hwork, (size_t)lhwork,
                                                           thrust::raw_pointer_cast(d_info.data()));
                    if (hwork) free(hwork);
                    if (st != CUSOLVER_STATUS_SUCCESS) printf("[Warn] Xsyevd(float) failed status=%d\n", (int)st);
                    thrust::device_vector<float> d_Sf(final_rank);
                    computeSingularValues_kernel<float><<<(int)((final_rank + 255)/256), 256, 0, stream>>>(
                        thrust::raw_pointer_cast(d_eig.data()), thrust::raw_pointer_cast(d_Sf.data()), (int)final_rank);

                    double stage_time_step5_verify_lowp_ms = 0.0;
                    if (need_vectors() && debug_mode) {
                        cudaEvent_t verify_start_evt, verify_end_evt;
                        cudaEventCreate(&verify_start_evt);
                        cudaEventCreate(&verify_end_evt);
                        cudaEventRecord(verify_start_evt, stream);
                        thrust::device_vector<float> d_U_R;
                        thrust::device_vector<float> d_V_R;
                        float* d_U_R_ptr = nullptr;
                        float* d_V_R_ptr = nullptr;

                        if (gram_is_RtR) {
                            d_V_R.resize((size_t)final_rank * (size_t)final_rank);
                            d_V_R_ptr = thrust::raw_pointer_cast(d_V_R.data());
                            cudaMemcpyAsync(d_V_R_ptr, d_G_ptr,
                                            sizeof(float) * (size_t)final_rank * (size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            d_U_R.resize((size_t)final_rank * (size_t)final_rank);
                            d_U_R_ptr = thrust::raw_pointer_cast(d_U_R.data());
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                    (int)final_rank, (int)final_rank, (int)final_rank,
                                                    &one, d_Rd_ptr, (int)final_rank,
                                                    d_V_R_ptr, (int)final_rank,
                                                    &zero, d_U_R_ptr, (int)final_rank);
                            dim3 th(16,16);
                            dim3 gr((int)((final_rank + th.x - 1)/th.x), (int)((final_rank + th.y - 1)/th.y));
                            divide_columns_by_diag_kernel<float><<<gr, th, 0, stream>>>(
                                d_U_R_ptr, thrust::raw_pointer_cast(d_Sf.data()),
                                (int)final_rank, (int)final_rank, (int)final_rank, 1e-12f);

                            // Match the stable float-mixed path more closely: re-orthogonalize U_R
                            // from RR^T, then backsolve V_R = R^T * U_R * Sigma^{-1}.
                            thrust::device_vector<float> d_Gu((size_t)final_rank * (size_t)final_rank);
                            float* d_Gu_ptr = thrust::raw_pointer_cast(d_Gu.data());
                            cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                        (int)final_rank, (int)final_rank,
                                        &one, d_Rd_ptr, (int)final_rank,
                                        &zero, d_Gu_ptr, (int)final_rank);
                            thrust::device_vector<float> d_eig_u(final_rank);
                            thrust::device_vector<int> d_info_u(1);
                            size_t lwork_u = 0, lhwork_u = 0;
                            cusolverDnXsyevd_bufferSize(cusolverHandle, nullptr, CUSOLVER_EIG_MODE_VECTOR,
                                                        CUBLAS_FILL_MODE_LOWER,
                                                        (int)final_rank,
                                                        CUDA_R_32F,
                                                        d_Gu_ptr, (int)final_rank,
                                                        CUDA_R_32F,
                                                        thrust::raw_pointer_cast(d_eig_u.data()),
                                                        CUDA_R_32F,
                                                        &lwork_u, &lhwork_u);
                            thrust::device_vector<float> d_work_u(lwork_u);
                            void* hwork_u = nullptr; if (lhwork_u > 0) hwork_u = malloc(lhwork_u);
                            cusolverDnXsyevd(cusolverHandle, nullptr, CUSOLVER_EIG_MODE_VECTOR,
                                             CUBLAS_FILL_MODE_LOWER,
                                             (int)final_rank,
                                             CUDA_R_32F,
                                             d_Gu_ptr, (int)final_rank,
                                             CUDA_R_32F,
                                             thrust::raw_pointer_cast(d_eig_u.data()),
                                             CUDA_R_32F,
                                             thrust::raw_pointer_cast(d_work_u.data()), (size_t)lwork_u,
                                             hwork_u, (size_t)lhwork_u,
                                             thrust::raw_pointer_cast(d_info_u.data()));
                            if (hwork_u) free(hwork_u);
                            cudaMemcpyAsync(d_U_R_ptr, d_Gu_ptr,
                                            sizeof(float) * (size_t)final_rank * (size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)final_rank, (int)final_rank, (int)final_rank,
                                                    &one, d_Rd_ptr, (int)final_rank,
                                                    d_U_R_ptr, (int)final_rank,
                                                    &zero, d_V_R_ptr, (int)final_rank);
                            divide_columns_by_diag_kernel<float><<<gr, th, 0, stream>>>(
                                d_V_R_ptr, thrust::raw_pointer_cast(d_Sf.data()),
                                (int)final_rank, (int)final_rank, (int)final_rank, 1e-12f);
                        } else {
                            d_U_R.resize((size_t)final_rank * (size_t)final_rank);
                            d_U_R_ptr = thrust::raw_pointer_cast(d_U_R.data());
                            cudaMemcpyAsync(d_U_R_ptr, d_G_ptr,
                                            sizeof(float) * (size_t)final_rank * (size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            d_V_R.resize((size_t)final_rank * (size_t)final_rank);
                            d_V_R_ptr = thrust::raw_pointer_cast(d_V_R.data());
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)final_rank, (int)final_rank, (int)final_rank,
                                                    &one, d_Rd_ptr, (int)final_rank,
                                                    d_U_R_ptr, (int)final_rank,
                                                    &zero, d_V_R_ptr, (int)final_rank);
                            dim3 th(16,16);
                            dim3 gr((int)((final_rank + th.x - 1)/th.x), (int)((final_rank + th.y - 1)/th.y));
                            divide_columns_by_diag_kernel<float><<<gr, th, 0, stream>>>(
                                d_V_R_ptr, thrust::raw_pointer_cast(d_Sf.data()),
                                (int)final_rank, (int)final_rank, (int)final_rank, 1e-12f);
                        }

                        const int m_debug = (int)m_effective;
                        const int ld_q_debug = ld_Q_basis_effective;
                        const int ld_a_debug = ld_A_effective;
                        const T_compute* q_basis_debug_ptr = d_Q_basis_effective_ptr;

                        float q_ortho_error = verify_orthogonality<float>(cublasHandle, m_debug, (int)final_rank,
                                                                          q_basis_debug_ptr, ld_q_debug);
                        float u_r_ortho_error = verify_orthogonality<float>(cublasHandle, (int)final_rank, (int)final_rank,
                                                                            d_U_R_ptr, (int)final_rank);
                        printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %e\n", (double)u_r_ortho_error);

                        if (skip_final_qr) {
                            thrust::device_vector<float> d_U_scaled_inv((size_t)final_rank * (size_t)final_rank);
                            cudaMemcpyAsync(thrust::raw_pointer_cast(d_U_scaled_inv.data()),
                                            d_U_R_ptr,
                                            sizeof(float) * (size_t)final_rank * (size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            {
                                dim3 thInv(16,16);
                                dim3 grInv((int)((final_rank + thInv.x - 1)/thInv.x),
                                           (int)((final_rank + thInv.y - 1)/thInv.y));
                                divide_columns_by_diag_kernel<float><<<grInv, thInv, 0, stream>>>(
                                    thrust::raw_pointer_cast(d_U_scaled_inv.data()),
                                    thrust::raw_pointer_cast(d_Sf.data()),
                                    (int)final_rank, (int)final_rank, (int)final_rank, 1e-12f);
                            }

                            thrust::device_vector<float> d_V_final((size_t)n * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)n, (int)final_rank, (int)final_rank,
                                                    &one, d_R_final_T_ptr, (int)n,
                                                    thrust::raw_pointer_cast(d_U_scaled_inv.data()), (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_V_final.data()), (int)n);

                            float v_ortho_error = verify_orthogonality<float>(cublasHandle, (int)n, (int)final_rank,
                                                                              thrust::raw_pointer_cast(d_V_final.data()), (int)n);
                            printf("Orthogonality of V ||V'*V - I||_F = %e\n", (double)v_ortho_error);
                            printf("Orthogonality of Q_basis ||Q_basis'*Q_basis - I||_F = %e\n", (double)q_ortho_error);

                            thrust::device_vector<float> d_U_final((size_t)m_effective * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)final_rank, (int)final_rank,
                                                    &one, q_basis_debug_ptr, ld_q_debug,
                                                    d_U_R_ptr, (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_U_final.data()), m_debug);
                            thrust::device_vector<float> d_Temp_U((size_t)m_effective * (size_t)final_rank);
                            cudaMemcpyAsync(thrust::raw_pointer_cast(d_Temp_U.data()),
                                            thrust::raw_pointer_cast(d_U_final.data()),
                                            sizeof(float) * (size_t)m_effective * (size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            {
                                dim3 thS(16,16);
                                dim3 grS((int)((final_rank + thS.x - 1)/thS.x), (int)(((int)m_effective + thS.y - 1)/thS.y));
                                scale_columns_by_vector_inplace_kernel<float><<<grS, thS, 0, stream>>>(
                                    thrust::raw_pointer_cast(d_Temp_U.data()),
                                    thrust::raw_pointer_cast(d_Sf.data()),
                                    m_debug, (int)final_rank, m_debug);
                            }
                            thrust::device_vector<float> d_Arec((size_t)m_effective * (size_t)n);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                    m_debug, (int)n, (int)final_rank,
                                                    &one, thrust::raw_pointer_cast(d_Temp_U.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_V_final.data()), (int)n,
                                                    &zero, thrust::raw_pointer_cast(d_Arec.data()), m_debug);
                            BlasCaller<float>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)n,
                                                    &one, d_A_effective_raw, ld_a_debug,
                                                    &minus_one, thrust::raw_pointer_cast(d_Arec.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_Arec.data()), m_debug);
                            float errA = 0.f, normA = 0.f;
                            cublasSnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                        thrust::raw_pointer_cast(d_Arec.data()), 1, &errA);
                            cublasSnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                        d_A_effective_raw, 1, &normA);
                            printf("Verifying Reconstruction Error ||A - Q_basis*U_R*S*V^T||_F...\n");
                            printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %e\n", (double)errA);
                            printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n",
                                   (normA > 1e-12f ? (double)errA / (double)normA : (double)errA));
                        } else {
                            thrust::device_vector<float> d_V_final_for_verify((size_t)n * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)n, (int)final_rank, (int)final_rank,
                                                    &one, d_R_final_T_ptr, (int)n,
                                                    d_U_R_ptr, (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_V_final_for_verify.data()), (int)n);
                            float v_ortho_error = verify_orthogonality<float>(cublasHandle, (int)n, (int)final_rank,
                                                                              thrust::raw_pointer_cast(d_V_final_for_verify.data()), (int)n);
                            printf("Orthogonality of V ||V'*V - I||_F = %e\n", (double)v_ortho_error);
                            printf("Orthogonality of Q_basis ||Q_basis'*Q_basis - I||_F = %e\n", (double)q_ortho_error);

                            thrust::device_vector<float> d_VS_temp((size_t)final_rank * (size_t)final_rank);
                            cudaMemcpyAsync(thrust::raw_pointer_cast(d_VS_temp.data()),
                                            d_V_R_ptr,
                                            sizeof(float) * (size_t)final_rank * (size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            {
                                dim3 thS2(16,16);
                                dim3 grS2((int)((final_rank + thS2.x - 1)/thS2.x), (int)((final_rank + thS2.y - 1)/thS2.y));
                                scale_columns_kernel<float><<<grS2, thS2, 0, stream>>>(
                                    thrust::raw_pointer_cast(d_VS_temp.data()),
                                    d_V_R_ptr,
                                    thrust::raw_pointer_cast(d_Sf.data()),
                                    (int)final_rank, (int)final_rank, (int)final_rank, (int)final_rank);
                            }
                            thrust::device_vector<float> d_QVS((size_t)m_effective * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)final_rank, (int)final_rank,
                                                    &one, q_basis_debug_ptr, ld_q_debug,
                                                    thrust::raw_pointer_cast(d_VS_temp.data()), (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_QVS.data()), m_debug);
                            thrust::device_vector<float> d_Arec((size_t)m_effective * (size_t)n);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                    m_debug, (int)n, (int)final_rank,
                                                    &one, thrust::raw_pointer_cast(d_QVS.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_V_final_for_verify.data()), (int)n,
                                                    &zero, thrust::raw_pointer_cast(d_Arec.data()), m_debug);
                            BlasCaller<float>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)n,
                                                    &one, d_A_effective_raw, ld_a_debug,
                                                    &minus_one, thrust::raw_pointer_cast(d_Arec.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_Arec.data()), m_debug);
                            float errA = 0.f, normA = 0.f;
                            cublasSnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                        thrust::raw_pointer_cast(d_Arec.data()), 1, &errA);
                            cublasSnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                        d_A_effective_raw, 1, &normA);
                            printf("Verifying Reconstruction Error ||A - Q_basis*U_R*S*V^T||_F...\n");
                            printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %e\n", (double)errA);
                            printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n",
                                   (normA > 1e-12f ? (double)errA / (double)normA : (double)errA));
                        }

                        cudaEventRecord(verify_end_evt, stream);
                        cudaEventSynchronize(verify_end_evt);
                        float step5_verify_lowp_ms = 0.0f;
                        cudaEventElapsedTime(&step5_verify_lowp_ms, verify_start_evt, verify_end_evt);
                        stage_time_step5_verify_lowp_ms = static_cast<double>(step5_verify_lowp_ms);
                        cudaEventDestroy(verify_start_evt);
                        cudaEventDestroy(verify_end_evt);
                    }

                    float elapsed_svd = stopTimer();
                    stage_time_svd_ms = static_cast<double>(elapsed_svd);
                    if (stage_time_step5_verify_lowp_ms > 0.0) {
                        stage_time_svd_ms = std::max(0.0, stage_time_svd_ms - stage_time_step5_verify_lowp_ms);
                    }

                    const double compression_time_ms = stage_time_qb_ms + stage_time_q_build_ms;
                    const double svd_tail_time_ms = stage_time_rproj_ms + stage_time_step4_ms + stage_time_svd_ms;
                    const double total_pipeline_ms = compression_time_ms + svd_tail_time_ms;
                    printf("QB finished successfully in %.6f ms.\n", stage_time_qb_ms);
                    printf("Q generation finished successfully in %.6f ms.\n", stage_time_q_build_ms);
                    printf("R_proj / basis prep finished in %.6f ms.\n", stage_time_rproj_ms);
                    if (stage_time_step4_ms > 0.0) {
                        printf("R factorization finished in %.6f ms.\n", stage_time_step4_ms);
                    } else {
                        printf("R factorization finished in 0.000000 ms.\n");
                    }
                    if (stage_time_step4_qbuild_ms > 0.0) {
                        printf("Step4 Q-build (debug only) finished in %.6f ms.\n", stage_time_step4_qbuild_ms);
                    }
                    printf("EVD-based SVD finished successfully in %.6f ms.\n", stage_time_svd_ms);
                    printf("Compression time (QB + Q): %.6f ms.\n", compression_time_ms);
                    printf("SVD tail time (R_proj + R_factor + EVD-SVD): %.6f ms.\n", svd_tail_time_ms);
                    printf("Total time (QB + Q + R_proj + R_factor + EVD-SVD): %.6f ms.\n", total_pipeline_ms);

                    if (lowp_runtime_possible) {
                        cudaEventDestroy(ev_conv_done);
                        cudaEventDestroy(ev_back_done);
                    }
                    if (use_pack_streams) {
                        cudaEventDestroy(ev_pack_w_last);
                        cudaEventDestroy(ev_pack_y_last);
                        cudaStreamDestroy(stream_pack_w);
                        cudaStreamDestroy(stream_pack_y);
                    }
                    if (need_panel_ready_event) {
                        cudaEventDestroy(ev_panel_ready);
                    }
                    if (lowp_runtime_possible) {
                        cudaStreamDestroy(stream_conv);
                    }
                    if (par_enable) {
                        for (size_t si = 0; si < compute_streams.size(); ++si) {
                            cudaEventDestroy(ev_compute_done[si]);
                            cublasDestroy(compute_handles[si]);
                            cudaStreamDestroy(compute_streams[si]);
                        }
                    }
                    return;
                }
                // EVD-based SVD route aligned with the paper pseudocode:
                // p = final_rank, q = n, B \approx R_proj (or its QR-compressed core when q/p <= T).
                const size_t core_p = final_rank;
                const size_t core_q = (size_t)n;
                const bool evd_qr_on_bt = !skip_final_qr;            // q / p <= T
                // EVD based SVD：先将 R 转置成 R^T, 然后转换成FP64进行EVD-based SVD
                thrust::device_vector<double> d_Rd(final_rank * final_rank);
                double* d_Rd_ptr = thrust::raw_pointer_cast(d_Rd.data());
                if constexpr (std::is_same_v<T_compute, float>) {
                    if (use_float_mixed_path) {
                    BlasCaller<double>::geam(cublasHandle,
                                             CUBLAS_OP_T, CUBLAS_OP_N,
                                             (int)final_rank, (int)final_rank,
                                             &one_d,
                                             d_R_true_final_double_ptr, (int)final_rank,
                                             &zero_d,
                                             d_Rd_ptr, (int)final_rank,
                                             d_Rd_ptr, (int)final_rank);
                    } else {
                        thrust::device_vector<T_compute> d_R_T((size_t)final_rank * final_rank);
                        BlasCaller<T_compute>::geam(cublasHandle,
                                                    CUBLAS_OP_T, CUBLAS_OP_N,
                                                    (int)final_rank, (int)final_rank,
                                                    &one,
                                                    d_R_true_final_ptr, (int)final_rank,
                                                    &zero,
                                                    thrust::raw_pointer_cast(d_R_T.data()), (int)final_rank,
                                                    thrust::raw_pointer_cast(d_R_T.data()), (int)final_rank);
                        size_t nn = (size_t)final_rank * final_rank;
                        convert_kernel<T_compute, double><<<(int)((nn + 255)/256), 256, 0, stream>>>(thrust::raw_pointer_cast(d_R_T.data()), d_Rd_ptr, nn);
                    }
                } else {
                    // R_T = R^T
                    thrust::device_vector<T_compute> d_R_T((size_t)final_rank * final_rank);
                    BlasCaller<T_compute>::geam(cublasHandle,
                                                CUBLAS_OP_T, CUBLAS_OP_N,
                                                (int)final_rank, (int)final_rank,
                                                &one,
                                                d_R_true_final_ptr, (int)final_rank,
                                                &zero,
                                                thrust::raw_pointer_cast(d_R_T.data()), (int)final_rank,
                                                thrust::raw_pointer_cast(d_R_T.data()), (int)final_rank);
                    size_t nn = (size_t)final_rank * final_rank;
                    convert_kernel<T_compute, double><<<(int)((nn + 255)/256), 256, 0, stream>>>(thrust::raw_pointer_cast(d_R_T.data()), d_Rd_ptr, nn);
                }

                thrust::device_vector<double> d_Bt_double;
                const double* d_Bt_double_ptr = nullptr;
                if (skip_final_qr) {
                    if constexpr (std::is_same_v<T_compute, double>) {
                        d_Bt_double_ptr = d_R_final_T_ptr;
                    } else if (use_float_mixed_path) {
                        d_Bt_double_ptr = d_R_final_T_double_ptr;
                    } else {
                        d_Bt_double.resize((size_t)n * (size_t)final_rank);
                        double* d_Bt_double_mut_ptr = thrust::raw_pointer_cast(d_Bt_double.data());
                        convert_kernel<float, double><<<(int)((((size_t)n * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                            d_R_final_T_ptr, d_Bt_double_mut_ptr, (size_t)n * (size_t)final_rank);
                        d_Bt_double_ptr = d_Bt_double_mut_ptr;
                    }
                }

                thrust::device_vector<double> d_G(final_rank * final_rank);
                double* d_G_ptr = thrust::raw_pointer_cast(d_G.data());
                printf("[Step5] EVD route: %s (p=%zu, q=%zu, q/p=%.3f, T=%.3f).\n",
                       evd_qr_on_bt ? "LEFT_SIDE via QR(B^T)" : "LEFT_SIDE via B*B^T",
                       core_p, core_q,
                       (core_p > 0 ? (double)core_q / (double)core_p : std::numeric_limits<double>::infinity()),
                       kRFinalQrRatioThreshold);
                if (skip_final_qr) {
                    // LEFT_SIDE without QR compression: G = B B^T, where B^T is stored in d_R_final_T.
                    cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                (int)final_rank, (int)n,
                                &one_d,
                                d_Bt_double_ptr, (int)n,
                                &zero_d,
                                d_G_ptr, (int)final_rank);
                } else {
                    // LEFT_SIDE with QR(B^T): d_Rd_ptr stores \hat{R}^T, so op=T yields \hat{R}\hat{R}^T.
                    cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                (int)final_rank, (int)final_rank,
                                &one_d,
                                d_Rd_ptr, (int)final_rank,
                                &zero_d,
                                d_G_ptr, (int)final_rank);
                }

                // 对 G 做 EVD：得到特征值（升序）与特征向量（列）
                thrust::device_vector<double> d_eig(final_rank);
                size_t lwork_evd = 0, lhwork_evd = 0;
                auto jobz = need_vectors() ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
                cusolverDnXsyevd_bufferSize(cusolverHandle, nullptr, jobz,
                                            CUBLAS_FILL_MODE_LOWER,
                                            (int)final_rank,
                                            CUDA_R_64F,
                                            d_G_ptr, (int)final_rank,
                                            CUDA_R_64F,
                                            thrust::raw_pointer_cast(d_eig.data()),
                                            CUDA_R_64F,
                                            &lwork_evd, &lhwork_evd);
                thrust::device_vector<double> d_work(lwork_evd);
                void* hwork = nullptr; if (lhwork_evd > 0) hwork = malloc(lhwork_evd);
                thrust::device_vector<int> d_info(1);
                cusolverStatus_t st = cusolverDnXsyevd(cusolverHandle, nullptr, jobz,
                                                       CUBLAS_FILL_MODE_LOWER,
                                                       (int)final_rank,
                                                       CUDA_R_64F,
                                                       d_G_ptr, (int)final_rank,
                                                       CUDA_R_64F,
                                                       thrust::raw_pointer_cast(d_eig.data()),
                                                       CUDA_R_64F,
                                                       thrust::raw_pointer_cast(d_work.data()), (size_t)lwork_evd,
                                                       hwork, (size_t)lhwork_evd,
                                                       thrust::raw_pointer_cast(d_info.data()));
                if (st != CUSOLVER_STATUS_SUCCESS) printf("[Warn] Xsyevd failed status=%d\n", (int)st);
                int info=0; cudaMemcpy(&info, thrust::raw_pointer_cast(d_info.data()), sizeof(int), cudaMemcpyDeviceToHost);
                if (info != 0) printf("[Warn] Xsyevd info=%d\n", info);
                if (hwork) free(hwork);

                // 由特征值得到奇异值：S = sqrt(λ)
                thrust::device_vector<double> d_Sd(final_rank);
                computeSingularValues_double<<<(int)((final_rank + 255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_eig.data()), thrust::raw_pointer_cast(d_Sd.data()), (int)final_rank);

                // 若只需要奇异值，则不进行 U/V 的构造
                if (need_vectors()) {
                    // 按分支构造 U_R / V_R
                    thrust::device_vector<double> d_Ud_best((size_t)final_rank * final_rank);
                    thrust::device_vector<double> d_Vd_best((size_t)final_rank * final_rank);
                    double* d_Ud_best_ptr = thrust::raw_pointer_cast(d_Ud_best.data());
                    double* d_Vd_best_ptr = thrust::raw_pointer_cast(d_Vd_best.data());
                    if (skip_final_qr) {
                        // LEFT_SIDE on the full core B: only the small left singular vectors U_R are
                        // materialized here. The large V is recovered later from B^T * (U_R * Sigma^{-1}).
                        cudaMemcpyAsync(d_Ud_best_ptr, d_G_ptr,
                                        sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                        cudaMemcpyDeviceToDevice, stream);
                        cudaMemsetAsync(d_Vd_best_ptr, 0,
                                        sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                        stream);
                        dim3 th_id(256);
                        dim3 gr_id((int)((final_rank + th_id.x - 1) / th_id.x));
                        add_identity_diagonal_kernel<double><<<gr_id, th_id, 0, stream>>>(
                            d_Vd_best_ptr, (int)final_rank, (int)final_rank, (int)final_rank, 1.0);
                        if (debug_mode) {
                            double uo = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank,
                                                                     d_Ud_best_ptr, (int)final_rank);
                            printf("U_R orthogonality ||U^T U - I||_F = %.6e\n", uo);
                            printf("[Info] LEFT_SIDE with QR bypass: large V is recovered later via B^T * (U_R * Sigma^{-1}).\n");
                        }
                    } else {
                        // d_G 的列即 U_R（double）
                        double* d_Ud_ptr = d_G_ptr;
                    // V_R = R^T * U_R，然后每列除以 Σ
                    thrust::device_vector<double> d_Vd(final_rank * final_rank);
                    double* d_Vd_ptr = thrust::raw_pointer_cast(d_Vd.data());
                    // d_Rd_ptr 已经是 R^T，直接使用不转置
                    cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                (int)final_rank, (int)final_rank, (int)final_rank,
                                &one_d,
                                d_Rd_ptr, (int)final_rank,
                                d_Ud_ptr, (int)final_rank,
                                &zero_d,
                                d_Vd_ptr, (int)final_rank);
                    dim3 th(16,16);
                    dim3 gr((int)((final_rank + th.x - 1)/th.x), (int)((final_rank + th.y - 1)/th.y));
                    divide_columns_by_diag_double<<<gr, th, 0, stream>>>(
                        d_Vd_ptr, thrust::raw_pointer_cast(d_Sd.data()), (int)final_rank, (int)final_rank, (int)final_rank, 1e-12);
                    // 将当前 U/V 写入统一持有者
                    cudaMemcpyAsync(d_Ud_best_ptr, d_Ud_ptr, sizeof(double)*final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);
                    cudaMemcpyAsync(d_Vd_best_ptr, d_Vd_ptr, sizeof(double)*final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);

                    // 残差与正交性检查（在低精度等级下跳过）
                    if (debug_mode) {
                        thrust::device_vector<double> d_US(final_rank * final_rank);
                        cudaMemcpyAsync(
                            thrust::raw_pointer_cast(d_US.data()),
                            d_Ud_ptr,
                            sizeof(double) * final_rank * final_rank,
                            cudaMemcpyDeviceToDevice,
                            stream);
                        {
                            dim3 th2(16,16);
                            dim3 gr2((int)((final_rank + th2.x - 1)/th2.x), (int)((final_rank + th2.y - 1)/th2.y));
                            scale_columns_by_vector_inplace_double<<<gr2, th2, 0, stream>>>(
                                thrust::raw_pointer_cast(d_US.data()),
                                thrust::raw_pointer_cast(d_Sd.data()),
                                (int)final_rank, (int)final_rank, (int)final_rank);
                        }
                        // Build R^T reconstruction: R^T = V * S * U^T
                        thrust::device_vector<double> d_Rt_rec(final_rank * final_rank);
                        {
                            thrust::device_vector<double> d_VS(final_rank * final_rank);
                            cudaMemcpyAsync(
                                thrust::raw_pointer_cast(d_VS.data()),
                                d_Vd_ptr,
                                sizeof(double) * final_rank * final_rank,
                                cudaMemcpyDeviceToDevice,
                                stream);
                            dim3 ths(16,16);
                            dim3 grs((int)((final_rank + ths.x - 1)/ths.x), (int)((final_rank + ths.y - 1)/ths.y));
                            scale_columns_by_vector_inplace_double<<<grs, ths, 0, stream>>>(
                                thrust::raw_pointer_cast(d_VS.data()),
                                thrust::raw_pointer_cast(d_Sd.data()),
                                (int)final_rank, (int)final_rank, (int)final_rank);
                            BlasCaller<double>::gemm(
                                cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                (int)final_rank, (int)final_rank, (int)final_rank,
                                &one_d,
                                thrust::raw_pointer_cast(d_VS.data()), (int)final_rank,
                                d_Ud_ptr, (int)final_rank,
                                &zero_d,
                                thrust::raw_pointer_cast(d_Rt_rec.data()), (int)final_rank);
                        }
                        // d_Rt_rec <- R^T - (V*S*U^T)
                        BlasCaller<double>::geam(
                            cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            (int)final_rank, (int)final_rank,
                            &one_d, d_Rd_ptr, (int)final_rank,
                            &minus_one_d, thrust::raw_pointer_cast(d_Rt_rec.data()), (int)final_rank,
                            thrust::raw_pointer_cast(d_Rt_rec.data()), (int)final_rank);
                        double rres = 0.0;
                        cublasDnrm2(cublasHandle, (int)(final_rank * final_rank),
                                    thrust::raw_pointer_cast(d_Rt_rec.data()), 1, &rres);
                        printf("R^T residual ||R^T - V*S*U^T||_F = %.6e\n", rres);
                        double uo = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank, d_Ud_ptr, (int)final_rank);
                        double vo = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank, d_Vd_ptr, (int)final_rank);
                        printf("U_R orthogonality ||U^T U - I||_F = %.6e\n", uo);
                        printf("V_R orthogonality ||V^T V - I||_F = %.6e\n", vo);
                    }
                    // 根据精度等级控制是否进行二次EVD
                    if constexpr (std::is_same_v<T_compute,float>) {
                        bool do_refine = false;
                        if (g_accuracy == AccuracyLevel::High) {
                            do_refine = true;
                        } else if (g_accuracy == AccuracyLevel::Medium) {
                            const double tol_ref = 5e-5;
                            double vo_before = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank, d_Vd_ptr, (int)final_rank);
                            do_refine = (vo_before > tol_ref);
                            if (debug_mode) printf("[Refine] V_R (double)=%.3e %s %.3e. %s.\n",
                                                   vo_before, do_refine? ">" : "<=", tol_ref,
                                                   do_refine? "Triggering EVD(R^T R)" : "Skip refine");
                        }
                        if (do_refine) {
                            if (debug_mode) printf("[Refine] V_R: Triggering EVD(R^T R) due to accuracy policy.\n");
                            // 对有问题的V进行二次EVD，V_check=(R^T*U)*Σ^{-1}
                            thrust::device_vector<double> d_Gv((size_t)final_rank * final_rank);
                            double* d_Gv_ptr = thrust::raw_pointer_cast(d_Gv.data());
                            cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank,
                                        &one_d, d_Rd_ptr, (int)final_rank,
                                        &zero_d, d_Gv_ptr, (int)final_rank);
                            size_t lw3=0, lhw3=0; thrust::device_vector<double> d_eig3(final_rank);
                        auto jobz_ref_v = need_vectors() ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
                        cusolverDnXsyevd_bufferSize(cusolverHandle, nullptr, jobz_ref_v,
                                                    CUBLAS_FILL_MODE_LOWER,
                                                    (int)final_rank, CUDA_R_64F,
                                                    d_Gv_ptr, (int)final_rank,
                                                    CUDA_R_64F, thrust::raw_pointer_cast(d_eig3.data()),
                                                    CUDA_R_64F, &lw3, &lhw3);
                            thrust::device_vector<double> d_work3(lw3); void* hw3=nullptr; if (lhw3>0) hw3=malloc(lhw3);
                            thrust::device_vector<int> di3(1);
                        cusolverDnXsyevd(cusolverHandle, nullptr, jobz_ref_v,
                                         CUBLAS_FILL_MODE_LOWER,
                                         (int)final_rank, CUDA_R_64F,
                                         d_Gv_ptr, (int)final_rank,
                                         CUDA_R_64F, thrust::raw_pointer_cast(d_eig3.data()),
                                         CUDA_R_64F, thrust::raw_pointer_cast(d_work3.data()), (size_t)lw3,
                                         hw3, (size_t)lhw3, thrust::raw_pointer_cast(di3.data()));
                            if (hw3) free(hw3);
                            // 重写 V
                            cudaMemcpyAsync(d_Vd_ptr, d_Gv_ptr, sizeof(double)*final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);
                            // V_check = (R^T * U) * Σ^{-1}
                            thrust::device_vector<double> d_Vcheck((size_t)final_rank * final_rank);
                            double* d_Vcheck_ptr = thrust::raw_pointer_cast(d_Vcheck.data());
                            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                        (int)final_rank, (int)final_rank, (int)final_rank,
                                        &one_d, d_Rd_ptr, (int)final_rank,
                                        d_Ud_ptr, (int)final_rank,
                                        &zero_d, d_Vcheck_ptr, (int)final_rank);
                            {
                                dim3 ths(16,16); dim3 grs((int)((final_rank + ths.x - 1)/ths.x), (int)((final_rank + ths.y - 1)/ths.y));
                                divide_columns_by_diag_double<<<grs, ths, 0, stream>>>(
                                    d_Vcheck_ptr, thrust::raw_pointer_cast(d_Sd.data()), (int)final_rank, (int)final_rank, (int)final_rank, 1e-12);
                            }
                            // 对V进行逐列点积并符号对齐
                            {
                                thrust::device_vector<double> d_corr((size_t)final_rank * final_rank);
                                BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                                         (int)final_rank, (int)final_rank, (int)final_rank,
                                                         &one_d, d_Vd_ptr, (int)final_rank,
                                                         d_Vcheck_ptr, (int)final_rank,
                                                         &zero_d, thrust::raw_pointer_cast(d_corr.data()), (int)final_rank);
                                std::vector<double> h_corr((size_t)final_rank * final_rank);
                                cudaMemcpy(h_corr.data(), thrust::raw_pointer_cast(d_corr.data()), sizeof(double)*(size_t)final_rank*final_rank, cudaMemcpyDeviceToHost);
                                std::vector<int> used_row((size_t)final_rank, 0);
                                std::vector<int> src_for_dest((size_t)final_rank, -1);
                                for (int j = 0; j < (int)final_rank; ++j) {
                                    int best_i = -1; double best_v = -1.0;
                                    for (int i = 0; i < (int)final_rank; ++i) {
                                        if (used_row[i]) continue;
                                        double v = fabs(h_corr[(size_t)j * (size_t)final_rank + (size_t)i]);
                                        if (v > best_v) { best_v = v; best_i = i; }
                                    }
                                    if (best_i < 0) best_i = j;
                                    used_row[best_i] = 1; src_for_dest[j] = best_i;
                                }
                                thrust::device_vector<double> d_Vperm((size_t)final_rank * final_rank);
                                double* d_Vperm_ptr = thrust::raw_pointer_cast(d_Vperm.data());
                                for (int j = 0; j < (int)final_rank; ++j) {
                                    int src = src_for_dest[j];
                                    cudaMemcpy2DAsync(
                                        d_Vperm_ptr + (size_t)j * (size_t)final_rank, (size_t)final_rank * sizeof(double),
                                        d_Vd_ptr    + (size_t)src * (size_t)final_rank, (size_t)final_rank * sizeof(double),
                                        (size_t)final_rank * sizeof(double), 1,
                                        cudaMemcpyDeviceToDevice, stream);
                                }
                                cudaMemcpyAsync(d_Vd_ptr, d_Vperm_ptr, sizeof(double)*(size_t)final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);
                            }
                            // 符号对齐 V to V_check
                            thrust::device_vector<double> d_dots(final_rank), d_sgn(final_rank);
                            column_dot_double<<<(int)final_rank, 256, 0, stream>>>(
                                d_Vd_ptr, d_Vcheck_ptr, thrust::raw_pointer_cast(d_dots.data()),
                                (int)final_rank, (int)final_rank, (int)final_rank);
                            build_sign_vector_double<<<(int)((final_rank + 255)/256), 256, 0, stream>>>(
                                thrust::raw_pointer_cast(d_dots.data()), thrust::raw_pointer_cast(d_sgn.data()), (int)final_rank);
                            {
                                dim3 ths2(16,16); dim3 grs2((int)((final_rank + ths2.x - 1)/ths2.x), (int)((final_rank + ths2.y - 1)/ths2.y));
                                scale_columns_by_vector_inplace_double<<<grs2, ths2, 0, stream>>>(
                                    d_Vd_ptr, thrust::raw_pointer_cast(d_sgn.data()), (int)final_rank, (int)final_rank, (int)final_rank);
                            }
                            // 更新统一持有者（V 被重算/对齐，U 未改）
                            cudaMemcpyAsync(d_Vd_best_ptr, d_Vd_ptr, sizeof(double)*final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);
                            cudaMemcpyAsync(d_Ud_best_ptr, d_Ud_ptr, sizeof(double)*final_rank*final_rank, cudaMemcpyDeviceToDevice, stream);
                            // 已在后续统一打印 double 的 before/after，这里不再打印 float 的 after-refine
                        } else {
                            if (debug_mode) printf("[Refine] V_R: Skip refine (accuracy policy).\n");
                        }

                    
                        if (debug_mode && g_accuracy != AccuracyLevel::Low) {
                            double uo_after = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank, d_Ud_ptr, (int)final_rank);
                            double vo_after = verify_orthogonality<double>(cublasHandle, (int)final_rank, (int)final_rank, d_Vd_ptr, (int)final_rank);
                            printf("U_R orthogonality (double) after  refine = %.6e\n", uo_after);
                            printf("V_R orthogonality (double) after  refine = %.6e\n", vo_after);
                        }
                    }
                }
                

                // Q_basis 正交性、U_R 正交性、V 正交性、以及重构误差 ||A - Q_basis*U_R*S*V^T||_F
                if (need_vectors()) {
                    thrust::device_vector<float> d_Sf(final_rank);
                    convert_kernel<double, float><<<(int)((final_rank + 255)/256), 256, 0, stream>>>(
                        thrust::raw_pointer_cast(d_Sd.data()), thrust::raw_pointer_cast(d_Sf.data()), (size_t)final_rank);
                    // 使用统一持有者
                    thrust::device_vector<double> d_Ud_hold = d_Ud_best;
                    thrust::device_vector<double> d_Vd_hold = d_Vd_best;
                    // 转回 float
                    thrust::device_vector<float> d_U_R((size_t)final_rank * final_rank), d_V_R((size_t)final_rank * final_rank);
                    convert_kernel<double, float><<<(int)(((size_t)final_rank*final_rank + 255)/256), 256, 0, stream>>>(
                        thrust::raw_pointer_cast(d_Ud_hold.data()), thrust::raw_pointer_cast(d_U_R.data()), (size_t)final_rank*final_rank);
                    convert_kernel<double, float><<<(int)(((size_t)final_rank*final_rank + 255)/256), 256, 0, stream>>>(
                        thrust::raw_pointer_cast(d_Vd_hold.data()), thrust::raw_pointer_cast(d_V_R.data()), (size_t)final_rank*final_rank);

                    if (debug_mode) {
                        cudaEvent_t verify_start_evt, verify_end_evt;
                        cudaEventCreate(&verify_start_evt);
                        cudaEventCreate(&verify_end_evt);
                        cudaEventRecord(verify_start_evt, stream);
                        const int m_debug = (int)m_effective;
                        const int ld_q_debug = ld_Q_basis_effective;
                        const int ld_a_debug = ld_A_effective;
                        const T_compute* q_basis_debug_ptr = d_Q_basis_effective_ptr;
                        cublasMath_t prev_math_mode_verify; cublasGetMathMode(cublasHandle, &prev_math_mode_verify);
                        cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
                        // 直接使用 float 做验证，避免不必要的精度转换
                        // 验证 Q_basis 正交性（统一以 double 口径计算）。
                        double q_ortho_error = 0.0;
                        if constexpr (std::is_same_v<T_compute,double>) {
                            q_ortho_error = verify_orthogonality<double>(cublasHandle, m_debug, (int)final_rank,
                                                                         (double*)q_basis_debug_ptr, ld_q_debug);
                        } else {
                            thrust::device_vector<double> d_Q_basis_d((size_t)m_effective * (size_t)final_rank);
                            double* d_Q_basis_d_ptr = thrust::raw_pointer_cast(d_Q_basis_d.data());
                            convert_kernel<float, double><<<(int)((((size_t)m_effective*(size_t)final_rank)+255)/256), 256, 0, stream>>>(
                                q_basis_debug_ptr, d_Q_basis_d_ptr, (size_t)m_effective * (size_t)final_rank);
                            q_ortho_error = verify_orthogonality<double>(cublasHandle, m_debug, (int)final_rank,
                                                                         d_Q_basis_d_ptr, ld_q_debug);
                        }
                        printf("Orthogonality of Q_basis ||Q_basis'*Q_basis - I||_F = %e\n", (double)q_ortho_error);
                        thrust::device_vector<double> d_step4_q_verify_double;
                        const double* d_step4_q_verify_double_ptr = nullptr;
                        if constexpr (std::is_same_v<T_compute, float>) {
                            if (use_float_mixed_path &&
                                d_A_effective_raw_double_ptr != nullptr &&
                                d_Q_basis_effective_double_ptr != nullptr) {
                                if (!skip_final_qr) {
                                    if (d_Q_step4_double_verify_ptr != nullptr) {
                                        d_step4_q_verify_double_ptr = d_Q_step4_double_verify_ptr;
                                    } else {
                                        d_step4_q_verify_double.resize((size_t)n * (size_t)final_rank);
                                        double* d_step4_q_verify_double_mut_ptr = thrust::raw_pointer_cast(d_step4_q_verify_double.data());
                                        convert_kernel<float, double><<<(int)((((size_t)n * (size_t)final_rank) + 255) / 256), 256, 0, stream>>>(
                                            d_R_final_T_ptr, d_step4_q_verify_double_mut_ptr, (size_t)n * (size_t)final_rank);
                                        d_step4_q_verify_double_ptr = d_step4_q_verify_double_mut_ptr;
                                    }

                                    thrust::device_vector<double> d_VS_double((size_t)final_rank * (size_t)final_rank);
                                    cudaMemcpyAsync(thrust::raw_pointer_cast(d_VS_double.data()),
                                                    thrust::raw_pointer_cast(d_Vd_hold.data()),
                                                    sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                                    cudaMemcpyDeviceToDevice, stream);
                                    {
                                        dim3 thd(16,16); dim3 grd((int)((final_rank + thd.x - 1)/thd.x), (int)((final_rank + thd.y - 1)/thd.y));
                                        scale_columns_by_vector_inplace_double<<<grd, thd, 0, stream>>>(
                                            thrust::raw_pointer_cast(d_VS_double.data()),
                                            thrust::raw_pointer_cast(d_Sd.data()),
                                            (int)final_rank, (int)final_rank, (int)final_rank);
                                    }
                                    thrust::device_vector<double> d_QVS_double((size_t)m_effective * (size_t)final_rank);
                                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                             m_debug, (int)final_rank, (int)final_rank,
                                                             &one_d, d_Q_basis_effective_double_ptr, ld_q_debug,
                                                             thrust::raw_pointer_cast(d_VS_double.data()), (int)final_rank,
                                                             &zero_d, thrust::raw_pointer_cast(d_QVS_double.data()), m_debug);
                                    thrust::device_vector<double> d_V_final_double((size_t)n * (size_t)final_rank);
                                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                             (int)n, (int)final_rank, (int)final_rank,
                                                             &one_d, d_step4_q_verify_double_ptr, (int)n,
                                                             thrust::raw_pointer_cast(d_Ud_hold.data()), (int)final_rank,
                                                             &zero_d, thrust::raw_pointer_cast(d_V_final_double.data()), (int)n);
                                    thrust::device_vector<double> d_Arec_double((size_t)m_effective * (size_t)n);
                                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                             m_debug, (int)n, (int)final_rank,
                                                             &one_d, thrust::raw_pointer_cast(d_QVS_double.data()), m_debug,
                                                             thrust::raw_pointer_cast(d_V_final_double.data()), (int)n,
                                                             &zero_d, thrust::raw_pointer_cast(d_Arec_double.data()), m_debug);
                                    BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                             m_debug, (int)n,
                                                             &one_d, d_A_effective_raw_double_ptr, ld_a_debug,
                                                             &minus_one_d, thrust::raw_pointer_cast(d_Arec_double.data()), m_debug,
                                                             thrust::raw_pointer_cast(d_Arec_double.data()), m_debug);
                                    double errA_double = 0.0, normA_double = 0.0;
                                    cublasDnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                                thrust::raw_pointer_cast(d_Arec_double.data()), 1, &errA_double);
                                    cublasDnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                                d_A_effective_raw_double_ptr, 1, &normA_double);
                                    printf("Reconstruction error (double verify) ||A - Q*U_R*S*V^T||_F = %e\n", errA_double);
                                    printf("Relative reconstruction error (double verify) ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n",
                                           (normA_double > 1e-18 ? errA_double / normA_double : errA_double));
                                } else if (d_R_final_T_double_ptr != nullptr) {
                                    thrust::device_vector<double> d_U_final_double((size_t)m_effective * (size_t)final_rank);
                                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                             m_debug, (int)final_rank, (int)final_rank,
                                                             &one_d, d_Q_basis_effective_double_ptr, ld_q_debug,
                                                             thrust::raw_pointer_cast(d_Ud_hold.data()), (int)final_rank,
                                                             &zero_d, thrust::raw_pointer_cast(d_U_final_double.data()), m_debug);

                                    thrust::device_vector<double> d_Temp_U_double((size_t)m_effective * (size_t)final_rank);
                                    cudaMemcpyAsync(thrust::raw_pointer_cast(d_Temp_U_double.data()),
                                                    thrust::raw_pointer_cast(d_U_final_double.data()),
                                                    sizeof(double) * (size_t)m_effective * (size_t)final_rank,
                                                    cudaMemcpyDeviceToDevice, stream);
                                    {
                                        dim3 thS_d(16,16);
                                        dim3 grS_d((int)((final_rank + thS_d.x - 1)/thS_d.x), (int)(((int)m_effective + thS_d.y - 1)/thS_d.y));
                                        scale_columns_by_vector_inplace_double<<<grS_d, thS_d, 0, stream>>>(
                                            thrust::raw_pointer_cast(d_Temp_U_double.data()),
                                            thrust::raw_pointer_cast(d_Sd.data()),
                                            m_debug, (int)final_rank, m_debug);
                                    }

                                    thrust::device_vector<double> d_U_scaled_inv_double((size_t)final_rank * (size_t)final_rank);
                                    cudaMemcpyAsync(thrust::raw_pointer_cast(d_U_scaled_inv_double.data()),
                                                    thrust::raw_pointer_cast(d_Ud_hold.data()),
                                                    sizeof(double) * (size_t)final_rank * (size_t)final_rank,
                                                    cudaMemcpyDeviceToDevice, stream);
                                    {
                                        dim3 thInv_d(16,16);
                                        dim3 grInv_d((int)((final_rank + thInv_d.x - 1)/thInv_d.x),
                                                     (int)((final_rank + thInv_d.y - 1)/thInv_d.y));
                                        divide_columns_by_diag_kernel<double><<<grInv_d, thInv_d, 0, stream>>>(
                                            thrust::raw_pointer_cast(d_U_scaled_inv_double.data()),
                                            thrust::raw_pointer_cast(d_Sd.data()),
                                            (int)final_rank, (int)final_rank, (int)final_rank, 1e-18);
                                    }

                                    thrust::device_vector<double> d_V_final_double((size_t)n * (size_t)final_rank);
                                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                             (int)n, (int)final_rank, (int)final_rank,
                                                             &one_d, d_R_final_T_double_ptr, (int)n,
                                                             thrust::raw_pointer_cast(d_U_scaled_inv_double.data()), (int)final_rank,
                                                             &zero_d, thrust::raw_pointer_cast(d_V_final_double.data()), (int)n);

                                    thrust::device_vector<double> d_Arec_double((size_t)m_effective * (size_t)n);
                                    BlasCaller<double>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                             m_debug, (int)n, (int)final_rank,
                                                             &one_d, thrust::raw_pointer_cast(d_Temp_U_double.data()), m_debug,
                                                             thrust::raw_pointer_cast(d_V_final_double.data()), (int)n,
                                                             &zero_d, thrust::raw_pointer_cast(d_Arec_double.data()), m_debug);
                                    BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                             m_debug, (int)n,
                                                             &one_d, d_A_effective_raw_double_ptr, ld_a_debug,
                                                             &minus_one_d, thrust::raw_pointer_cast(d_Arec_double.data()), m_debug,
                                                             thrust::raw_pointer_cast(d_Arec_double.data()), m_debug);
                                    double errA_double = 0.0, normA_double = 0.0;
                                    cublasDnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                                thrust::raw_pointer_cast(d_Arec_double.data()), 1, &errA_double);
                                    cublasDnrm2(cublasHandle, (int)((size_t)m_effective * (size_t)n),
                                                d_A_effective_raw_double_ptr, 1, &normA_double);
                                    printf("Reconstruction error (double verify) ||A - Q*U_R*S*V^T||_F = %e\n", errA_double);
                                    printf("Relative reconstruction error (double verify) ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n",
                                           (normA_double > 1e-18 ? errA_double / normA_double : errA_double));
                                }
                            }
                        }
                        // 估算最终重构验证的临时内存占用
                        size_t bytes_V_final = (size_t)n * (size_t)final_rank * sizeof(float);
                        size_t bytes_QUS     = (size_t)m_effective * (size_t)final_rank * sizeof(float);
                        size_t bytes_Arec    = (size_t)m_effective * (size_t)n * sizeof(float);
                        size_t total_bytes   = bytes_V_final + bytes_QUS + bytes_Arec;
                        if (skip_final_qr) {
                            double gb_need = (double)total_bytes / (1024.0*1024.0*1024.0);
                            printf("[Info] Forcing final A reconstruction in verification (QR bypass, alloc ~%.2f GiB).\n", gb_need);
                            // U_final = Q_basis * U_R
                            thrust::device_vector<float> d_U_final((size_t)m_effective * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)final_rank, (int)final_rank,
                                                    &one, q_basis_debug_ptr, ld_q_debug,
                                                    thrust::raw_pointer_cast(d_U_R.data()), (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_U_final.data()), m_debug);

                            // Temp = U_final * S
                            thrust::device_vector<float> d_Temp_U((size_t)m_effective * (size_t)final_rank);
                            cudaMemcpyAsync(thrust::raw_pointer_cast(d_Temp_U.data()),
                                            thrust::raw_pointer_cast(d_U_final.data()),
                                            sizeof(float)*(size_t)m_effective*(size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            {
                                dim3 thS(16,16);
                                dim3 grS((int)((final_rank + thS.x - 1)/thS.x), (int)(((int)m_effective + thS.y - 1)/thS.y));
                                scale_columns_kernel<float><<<grS, thS, 0, stream>>>(
                                    thrust::raw_pointer_cast(d_Temp_U.data()),
                                    thrust::raw_pointer_cast(d_U_final.data()),
                                    thrust::raw_pointer_cast(d_Sf.data()),
                                    m_debug, (int)final_rank, m_debug, m_debug);
                            }

                            // V_final = V_R = R_final^T * (U_R * Σ^{-1})
                            thrust::device_vector<float> d_U_scaled_inv((size_t)final_rank * (size_t)final_rank);
                            cudaMemcpyAsync(thrust::raw_pointer_cast(d_U_scaled_inv.data()),
                                            thrust::raw_pointer_cast(d_U_R.data()),
                                            sizeof(float)*(size_t)final_rank*(size_t)final_rank,
                                            cudaMemcpyDeviceToDevice, stream);
                            {
                                dim3 thInv(16,16);
                                dim3 grInv((int)((final_rank + thInv.x - 1)/thInv.x),
                                           (int)((final_rank + thInv.y - 1)/thInv.y));
                                divide_columns_by_diag_kernel<float><<<grInv, thInv, 0, stream>>>(
                                    thrust::raw_pointer_cast(d_U_scaled_inv.data()),
                                    thrust::raw_pointer_cast(d_Sf.data()),
                                    (int)final_rank, (int)final_rank, (int)final_rank, 1e-12f);
                            }
                            thrust::device_vector<float> d_V_final((size_t)n * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)n, (int)final_rank, (int)final_rank,
                                                    &one, d_R_final_T_ptr, (int)n,
                                                    thrust::raw_pointer_cast(d_U_scaled_inv.data()), (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_V_final.data()), (int)n);

                            // A_rec = (Q_basis*U_R)*S*V_R^T
                            thrust::device_vector<float> d_Arec((size_t)m_effective * (size_t)n);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                    m_debug, (int)n, (int)final_rank,
                                                    &one, thrust::raw_pointer_cast(d_Temp_U.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_V_final.data()), (int)n,
                                                    &zero, thrust::raw_pointer_cast(d_Arec.data()), m_debug);

                            // 误差 ||A - A_rec||_F
                            BlasCaller<float>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)n,
                                                    &one, d_A_effective_raw, ld_a_debug,
                                                    &minus_one, thrust::raw_pointer_cast(d_Arec.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_Arec.data()), m_debug);
                            float errA = 0.f, normA = 0.f;
                            cublasSnrm2(cublasHandle, (int)(m_effective*n), thrust::raw_pointer_cast(d_Arec.data()), 1, &errA);
                            cublasSnrm2(cublasHandle, (int)(m_effective*n), d_A_effective_raw, 1, &normA);
                            printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %e\n", (double)errA);
                            printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n", (normA>1e-12f? (double)errA/(double)normA : (double)errA));
                        } else {
                            double gb_need = (double)total_bytes / (1024.0*1024.0*1024.0);
                            printf("[Info] Forcing final A reconstruction in verification (alloc ~%.2f GiB).\n", gb_need);
                            // Small-R SVD is formed on R_true_final^T, so the verification
                            // must reconstruct A via (Q_basis * (V_R * S)) * (Q_step4 * U_R)^T.
                            thrust::device_vector<float> d_V_final_for_verify((size_t)n * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    (int)n, (int)final_rank, (int)final_rank,
                                                    &one, d_R_final_T_ptr, (int)n,
                                                    thrust::raw_pointer_cast(d_U_R.data()), (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_V_final_for_verify.data()), (int)n);

                            thrust::device_vector<float> d_VS_temp((size_t)final_rank * (size_t)final_rank);
                            cudaMemcpyAsync(thrust::raw_pointer_cast(d_VS_temp.data()), thrust::raw_pointer_cast(d_V_R.data()), sizeof(float)*(size_t)final_rank*(size_t)final_rank, cudaMemcpyDeviceToDevice, stream);
                            {
                                dim3 thS2(16,16); dim3 grS2((int)((final_rank + thS2.x - 1)/thS2.x), (int)((final_rank + thS2.y - 1)/thS2.y));
                                scale_columns_kernel<float><<<grS2, thS2, 0, stream>>>(
                                    thrust::raw_pointer_cast(d_VS_temp.data()),
                                    thrust::raw_pointer_cast(d_V_R.data()),
                                    thrust::raw_pointer_cast(d_Sf.data()),
                                    (int)final_rank, (int)final_rank, (int)final_rank, (int)final_rank);
                            }
                            thrust::device_vector<float> d_QVS((size_t)m_effective * (size_t)final_rank);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)final_rank, (int)final_rank,
                                                    &one, q_basis_debug_ptr, ld_q_debug,
                                                    thrust::raw_pointer_cast(d_VS_temp.data()), (int)final_rank,
                                                    &zero, thrust::raw_pointer_cast(d_QVS.data()), m_debug);
                            thrust::device_vector<float> d_Arec((size_t)m_effective * (size_t)n);
                            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                                    m_debug, (int)n, (int)final_rank,
                                                    &one, thrust::raw_pointer_cast(d_QVS.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_V_final_for_verify.data()), (int)n,
                                                    &zero, thrust::raw_pointer_cast(d_Arec.data()), m_debug);
                            BlasCaller<float>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                                    m_debug, (int)n,
                                                    &one, d_A_effective_raw, ld_a_debug,
                                                    &minus_one, thrust::raw_pointer_cast(d_Arec.data()), m_debug,
                                                    thrust::raw_pointer_cast(d_Arec.data()), m_debug);
                            float errA=0.f, normA=0.f;
                            cublasSnrm2(cublasHandle, (int)(m_effective*n), thrust::raw_pointer_cast(d_Arec.data()), 1, &errA);
                            cublasSnrm2(cublasHandle, (int)(m_effective*n), d_A_effective_raw, 1, &normA);
                            printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %e\n", (double)errA);
                            printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %e\n", (normA>1e-12f? (double)errA/(double)normA : (double)errA));
                        }
                        cudaEventRecord(verify_end_evt, stream);
                        cudaEventSynchronize(verify_end_evt);
                        float step5_verify_ms = 0.0f;
                        cudaEventElapsedTime(&step5_verify_ms, verify_start_evt, verify_end_evt);
                        stage_time_step5_verify_ms += static_cast<double>(step5_verify_ms);
                        printf("[Verify] Step 5 debug verification finished in %f ms.\n", step5_verify_ms);
                        cudaEventDestroy(verify_start_evt);
                        cudaEventDestroy(verify_end_evt);
                        cublasSetMathMode(cublasHandle, prev_math_mode_verify);
                    }
                }
            }
            float elapsed_svd = stopTimer();
            stage_time_svd_ms = static_cast<double>(elapsed_svd);
            if (stage_time_step5_verify_ms > 0.0) {
                stage_time_svd_ms = std::max(0.0, stage_time_svd_ms - stage_time_step5_verify_ms);
            }
            printf("SVD stage finished in %f ms.\n", (float)stage_time_svd_ms);

            double total_pipeline_ms =
                stage_time_qb_ms + stage_time_q_build_ms + stage_time_rproj_ms + stage_time_step4_ms + stage_time_svd_ms;
            printf("=== HQB Timing Summary ===\n");
            printf("  QB stage          : %.6f ms\n", stage_time_qb_ms);
            printf("  Q construction    : %.6f ms\n", stage_time_q_build_ms);
            printf("  R_proj / prep     : %.6f ms\n", stage_time_rproj_ms);
            if (stage_time_step4_ms > 0.0) {
                printf("  R factor (Step 4) : %.6f ms\n", stage_time_step4_ms);
            } else {
                printf("  R factor (Step 4) : bypassed\n");
            }
            if (stage_time_step4_qbuild_ms > 0.0) {
                printf("  Step4 Q-build(debug): %.6f ms\n", stage_time_step4_qbuild_ms);
            }
            printf("  SVD of small R    : %.6f ms\n", stage_time_svd_ms);
            if (stage_time_step5_verify_ms > 0.0) {
                printf("  Step5 verify(debug): %.6f ms\n", stage_time_step5_verify_ms);
            }
            printf("  Total (QB+Q+Rproj+R+SVD): %.6f ms\n", total_pipeline_ms);


            // 销毁辅助流与事件
            if (lowp_runtime_possible) {
                cudaEventDestroy(ev_conv_done);
                cudaEventDestroy(ev_back_done);
            }
            if (use_pack_streams) {
                cudaEventDestroy(ev_pack_w_last);
                cudaEventDestroy(ev_pack_y_last);
                cudaStreamDestroy(stream_pack_w);
                cudaStreamDestroy(stream_pack_y);
            }
            if (need_panel_ready_event) {
                cudaEventDestroy(ev_panel_ready);
            }
            if (lowp_runtime_possible) {
                cudaStreamDestroy(stream_conv);
            }
            if (par_enable) {
                for (size_t si = 0; si < compute_streams.size(); ++si) {
                    cudaEventDestroy(ev_compute_done[si]);
                    cublasDestroy(compute_handles[si]);
                    cudaStreamDestroy(compute_streams[si]);
                }
            }
}
}

// 直接 SVD 路径：用于小规模或满秩矩阵
template<typename T_storage, typename T_compute>
void DirectSVD(cublasHandle_t cublasHandle, cusolverDnHandle_t cusolverHandle, cudaStream_t stream)
{
    const long int m_in = m;
    const long int n_in = n;
    const long int mn = std::min(m_in, n_in);
    const T_compute one = T_compute(1.0);
    const T_compute zero = T_compute(0.0);
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    const double one_d = 1.0, zero_d = 0.0, minus_one_d = -1.0;
    const char* storage_prec_str = precision_name<T_storage>();
    printf("[Route] solver=DirectSVD storage=%s compute=%s m=%ld n=%ld reason=%s\n",
           precision_name<T_storage>(),
           precision_name<T_compute>(),
           m_in, n_in,
           (g_rank_type == RankType::Full) ? "rank_type=full" : "small_matrix");
    thrust::device_vector<T_compute> d_A;
    std::vector<T_compute> h_S_expected;
    if (!g_use_custom_matrix) {
        // 1) 生成待分解矩阵 A（与 HQB 路径使用相同的生成器）
        thrust::device_vector<T_storage> d_A_storage;
        long int ldA_storage = 0;
        create_low_rank_matrix<T_storage, T_compute>(d_A_storage, h_S_expected, m_in, n_in, (size_t)rank, spectrum_type,
                                                     static_cast<T_compute>(s_max), static_cast<T_compute>(s_min),
                                                     cublasHandle, cusolverHandle, stream, ldA_storage);
        d_A.resize((size_t)m_in * (size_t)n_in);
        {
            dim3 t(16,16);
            dim3 b((n_in + t.x - 1)/t.x, (m_in + t.y - 1)/t.y);
            convert_kernel_strided<T_storage, T_compute><<<b, t, 0, stream>>>(
                thrust::raw_pointer_cast(d_A_storage.data()), (size_t)ldA_storage,
                thrust::raw_pointer_cast(d_A.data()), (size_t)m_in,
                (size_t)m_in, (size_t)n_in);
        }
        d_A_storage.clear(); d_A_storage.shrink_to_fit();
    } else {
        printf("=== Step 1: 从用户文件 '%s' 加载矩阵 ===\n", g_matrix_file_path.c_str());
        if (!load_matrix_from_file<T_compute>(stream,
                                              m_in,
                                              m_in,
                                              n_in,
                                              g_matrix_file_path,
                                              g_matrix_file_format,
                                              g_matrix_file_layout,
                                              d_A)) {
            printf("[Error] Aborting DirectSVD due to matrix loading failure.\n");
            return;
        }
    }

    // 2）若为长方阵，则先通过 QR 化简为方阵再分解
    thrust::device_vector<T_compute> d_Q; 
    thrust::device_vector<T_compute> d_Rsquare; int rs_dim = 0; 
    if (m_in > n_in) {
        thrust::device_vector<T_compute> d_tau(n_in);
        int lwork_geqrf=0, lwork_orgqr=0; thrust::device_vector<int> d_info(1);
        SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, (int)m_in, (int)n_in, thrust::raw_pointer_cast(d_A.data()), (int)m_in, &lwork_geqrf);
        thrust::device_vector<T_compute> d_work(lwork_geqrf);
        SolverCaller<T_compute>::geqrf(cusolverHandle, (int)m_in, (int)n_in, thrust::raw_pointer_cast(d_A.data()), (int)m_in,
                                       thrust::raw_pointer_cast(d_tau.data()), thrust::raw_pointer_cast(d_work.data()), lwork_geqrf,
                                       thrust::raw_pointer_cast(d_info.data()));
        // 得到n*n的R
        d_Rsquare.resize((size_t)n_in * (size_t)n_in);
        {
            dim3 th(16,16); dim3 gr((n_in + 15)/16, (n_in + 15)/16);
            extract_upper_triangle_kernel<T_compute><<<gr, th, 0, stream>>>(
                thrust::raw_pointer_cast(d_Rsquare.data()), (int)n_in,
                thrust::raw_pointer_cast(d_A.data()), (int)m_in, (int)n_in, (int)n_in);
        }
        // 仅在 debug 下构造薄 Q (m x n)，避免非验证路径的多余计算
        if (debug_mode) {
            SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, (int)m_in, (int)n_in, (int)n_in, thrust::raw_pointer_cast(d_A.data()), (int)m_in,
                                                     thrust::raw_pointer_cast(d_tau.data()), &lwork_orgqr);
            thrust::device_vector<T_compute> d_work2(lwork_orgqr);
            SolverCaller<T_compute>::orgqr(cusolverHandle, (int)m_in, (int)n_in, (int)n_in, thrust::raw_pointer_cast(d_A.data()), (int)m_in,
                                           thrust::raw_pointer_cast(d_tau.data()), thrust::raw_pointer_cast(d_work2.data()), lwork_orgqr,
                                           thrust::raw_pointer_cast(d_info.data()));
            d_Q = std::move(d_A);
        }
        rs_dim = (int)n_in;
    } else if (m_in < n_in) {
        // At = A^T,转置后变为n*m, n>m
        thrust::device_vector<T_compute> d_At((size_t)n_in * (size_t)m_in);
        BlasCaller<T_compute>::geam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, (int)n_in, (int)m_in,
                                    &one, thrust::raw_pointer_cast(d_A.data()), (int)m_in,
                                    &zero, thrust::raw_pointer_cast(d_A.data()), (int)m_in,
                                    thrust::raw_pointer_cast(d_At.data()), (int)n_in);
        // 转置后对 At 做 QR 分解
        thrust::device_vector<T_compute> d_tau(m_in);
        int lwork_geqrf=0, lwork_orgqr=0; thrust::device_vector<int> d_info(1);
        SolverCaller<T_compute>::geqrf_bufferSize(cusolverHandle, (int)n_in, (int)m_in, thrust::raw_pointer_cast(d_At.data()), (int)n_in, &lwork_geqrf);
        thrust::device_vector<T_compute> d_work(lwork_geqrf);
        SolverCaller<T_compute>::geqrf(cusolverHandle, (int)n_in, (int)m_in, thrust::raw_pointer_cast(d_At.data()), (int)n_in,
                                       thrust::raw_pointer_cast(d_tau.data()), thrust::raw_pointer_cast(d_work.data()), lwork_geqrf,
                                       thrust::raw_pointer_cast(d_info.data()));
        // 获得方阵R：m*m
        d_Rsquare.resize((size_t)m_in * (size_t)m_in);
        {
            dim3 th(16,16); dim3 gr((m_in + 15)/16, (m_in + 15)/16);
            extract_upper_triangle_kernel<T_compute><<<gr, th, 0, stream>>>(
                thrust::raw_pointer_cast(d_Rsquare.data()), (int)m_in,
                thrust::raw_pointer_cast(d_At.data()), (int)n_in, (int)m_in, (int)m_in);
        }
        // 仅在 debug 下构造薄 Q_at (n x m)
        if (debug_mode) {
            SolverCaller<T_compute>::orgqr_bufferSize(cusolverHandle, (int)n_in, (int)m_in, (int)m_in, thrust::raw_pointer_cast(d_At.data()), (int)n_in,
                                                     thrust::raw_pointer_cast(d_tau.data()), &lwork_orgqr);
            thrust::device_vector<T_compute> d_work2(lwork_orgqr);
            SolverCaller<T_compute>::orgqr(cusolverHandle, (int)n_in, (int)m_in, (int)m_in, thrust::raw_pointer_cast(d_At.data()), (int)n_in,
                                           thrust::raw_pointer_cast(d_tau.data()), thrust::raw_pointer_cast(d_work2.data()), lwork_orgqr,
                                           thrust::raw_pointer_cast(d_info.data()));
            d_Q = std::move(d_At);
        }
        rs_dim = (int)m_in;
    } else {
        // m=n方阵直接用A
        d_Rsquare = std::move(d_A);
        rs_dim = (int)mn;
    }

    if constexpr (std::is_same_v<T_storage, __half> || std::is_same_v<T_storage, __nv_bfloat16>) {
        startTimer();
        const float minus_one_f = -1.0f;
        thrust::device_vector<float> d_Mf((size_t)rs_dim * (size_t)rs_dim);
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_Mf.data()),
                        thrust::raw_pointer_cast(d_Rsquare.data()),
                        sizeof(float) * (size_t)rs_dim * (size_t)rs_dim,
                        cudaMemcpyDeviceToDevice, stream);

        thrust::device_vector<float> d_G((size_t)rs_dim * (size_t)rs_dim);
        cublasSsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                    rs_dim, rs_dim,
                    &one,
                    thrust::raw_pointer_cast(d_Mf.data()), rs_dim,
                    &zero,
                    thrust::raw_pointer_cast(d_G.data()), rs_dim);
        thrust::device_vector<float> d_eig(rs_dim);
        size_t lw = 0, lhw = 0;
        auto jobz = need_vectors() ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
        cusolverDnXsyevd_bufferSize(cusolverHandle, nullptr, jobz,
                                    CUBLAS_FILL_MODE_LOWER,
                                    rs_dim, CUDA_R_32F,
                                    thrust::raw_pointer_cast(d_G.data()), rs_dim,
                                    CUDA_R_32F, thrust::raw_pointer_cast(d_eig.data()),
                                    CUDA_R_32F, &lw, &lhw);
        thrust::device_vector<float> d_work(lw);
        void* hwork = nullptr; if (lhw > 0) hwork = malloc(lhw);
        thrust::device_vector<int> d_info_svd(1);
        cusolverDnXsyevd(cusolverHandle, nullptr, jobz,
                         CUBLAS_FILL_MODE_LOWER,
                         rs_dim, CUDA_R_32F,
                         thrust::raw_pointer_cast(d_G.data()), rs_dim,
                         CUDA_R_32F, thrust::raw_pointer_cast(d_eig.data()),
                         CUDA_R_32F, thrust::raw_pointer_cast(d_work.data()), (size_t)lw,
                         hwork, (size_t)lhw, thrust::raw_pointer_cast(d_info_svd.data()));
        if (hwork) free(hwork);

        thrust::device_vector<float> d_Sf(rs_dim);
        computeSingularValues_kernel<float><<<(int)((rs_dim + 255)/256), 256, 0, stream>>>(
            thrust::raw_pointer_cast(d_eig.data()), thrust::raw_pointer_cast(d_Sf.data()), rs_dim);
        thrust::device_vector<float> d_Uf((size_t)rs_dim * (size_t)rs_dim);
        float* d_Vf_ptr = thrust::raw_pointer_cast(d_G.data());
        BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                rs_dim, rs_dim, rs_dim,
                                &one, thrust::raw_pointer_cast(d_Mf.data()), rs_dim,
                                d_Vf_ptr, rs_dim,
                                &zero, thrust::raw_pointer_cast(d_Uf.data()), rs_dim);
        {
            dim3 th(16,16); dim3 gr((int)((rs_dim + th.x - 1)/th.x), (int)((rs_dim + th.y - 1)/th.y));
            divide_columns_by_diag_kernel<float><<<gr, th, 0, stream>>>(
                thrust::raw_pointer_cast(d_Uf.data()), thrust::raw_pointer_cast(d_Sf.data()), rs_dim, rs_dim, rs_dim, 1e-12f);
        }

        if (debug_mode && need_vectors()) {
            float u_ortho = verify_orthogonality<float>(cublasHandle, rs_dim, rs_dim, thrust::raw_pointer_cast(d_Uf.data()), rs_dim);
            float v_ortho = verify_orthogonality<float>(cublasHandle, rs_dim, rs_dim, d_Vf_ptr, rs_dim);
            printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", (double)u_ortho);
            printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", (double)v_ortho);
            thrust::device_vector<float> d_US((size_t)rs_dim * (size_t)rs_dim);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Uf.data()),
                            sizeof(float) * (size_t)rs_dim * (size_t)rs_dim, cudaMemcpyDeviceToDevice, stream);
            {
                dim3 th(16,16); dim3 gr((int)((rs_dim + th.x - 1)/th.x), (int)((rs_dim + th.y - 1)/th.y));
                scale_columns_by_vector_inplace_kernel<float><<<gr, th, 0, stream>>>(
                    thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Sf.data()), rs_dim, rs_dim, rs_dim);
            }
            thrust::device_vector<float> d_Arec((size_t)rs_dim * (size_t)rs_dim);
            BlasCaller<float>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                    rs_dim, rs_dim, rs_dim,
                                    &one, thrust::raw_pointer_cast(d_US.data()), rs_dim,
                                    d_Vf_ptr, rs_dim,
                                    &zero, thrust::raw_pointer_cast(d_Arec.data()), rs_dim);
            BlasCaller<float>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    rs_dim, rs_dim,
                                    &one, thrust::raw_pointer_cast(d_Rsquare.data()), rs_dim,
                                    &minus_one_f, thrust::raw_pointer_cast(d_Arec.data()), rs_dim,
                                    thrust::raw_pointer_cast(d_Arec.data()), rs_dim);
            float err = 0.0f, normA = 0.0f;
            cublasSnrm2(cublasHandle, rs_dim * rs_dim, thrust::raw_pointer_cast(d_Arec.data()), 1, &err);
            cublasSnrm2(cublasHandle, rs_dim * rs_dim, thrust::raw_pointer_cast(d_Rsquare.data()), 1, &normA);
            printf("Verifying Reconstruction Error ||A - U_R*S*V^T||_F...\n");
            printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %.6e\n", (double)err);
            printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %.6e\n",
                   (normA > 1e-12f ? (double)err / (double)normA : (double)err));
        }
        float elapsed_direct = stopTimer();
        printf("DirectSVD stage finished in %f ms.\n", elapsed_direct);
        return;
    }

    // 对化简得到的方阵执行 SVD（double）
    startTimer();
    thrust::device_vector<double> d_Md((size_t)rs_dim * (size_t)rs_dim);
    if constexpr (std::is_same_v<T_compute, double>) {
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_Md.data()), thrust::raw_pointer_cast(d_Rsquare.data()), sizeof(double)*(size_t)rs_dim*(size_t)rs_dim, cudaMemcpyDeviceToDevice, stream);
    } else {
        convert_kernel<T_compute, double><<<(int)((((size_t)rs_dim*(size_t)rs_dim)+255)/256), 256, 0, stream>>>(
            thrust::raw_pointer_cast(d_Rsquare.data()), thrust::raw_pointer_cast(d_Md.data()), (size_t)rs_dim*(size_t)rs_dim);
    }

    // 若计算类型为 double，直接用 cusolverDnDgesvd 对 M 执行标准 SVD
    if constexpr (std::is_same_v<T_compute, double>) {
        thrust::device_vector<double> d_Sd(rs_dim);
        thrust::device_vector<double> d_Ud((size_t)rs_dim * (size_t)rs_dim);
        thrust::device_vector<double> d_VT((size_t)rs_dim * (size_t)rs_dim);
        thrust::device_vector<int> d_info_svd(1);
        int lwork_svd = 0;
        cusolverDnDgesvd_bufferSize(cusolverHandle, rs_dim, rs_dim, &lwork_svd);
        thrust::device_vector<double> d_work_svd(lwork_svd);
        // 仅在 debug 且 m==n 的场景保留一份未修改的 M，用于事后重构误差对比
        thrust::device_vector<double> d_Mref;
        if (debug_mode && m_in == n_in) {
            d_Mref.resize((size_t)rs_dim * (size_t)rs_dim);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_Mref.data()),
                            thrust::raw_pointer_cast(d_Md.data()),
                            sizeof(double)*(size_t)rs_dim*(size_t)rs_dim,
                            cudaMemcpyDeviceToDevice, stream);
        }
        {
        // 精细控制：按需求设置 jobu/jobvt
        char jobu = want_U()  ? 'A' : 'N';
        char jobvt = want_Vt() ? 'A' : 'N';
        cusolverDnDgesvd(cusolverHandle, jobu, jobvt, rs_dim, rs_dim,
                         thrust::raw_pointer_cast(d_Md.data()), rs_dim,
                         thrust::raw_pointer_cast(d_Sd.data()),
                         (jobu=='N' ? nullptr : thrust::raw_pointer_cast(d_Ud.data())), rs_dim,
                         (jobvt=='N' ? nullptr : thrust::raw_pointer_cast(d_VT.data())), rs_dim,
                         thrust::raw_pointer_cast(d_work_svd.data()), lwork_svd,
                         nullptr, thrust::raw_pointer_cast(d_info_svd.data()));
        }
        // 仅在需要右奇异向量时执行：V = (V^T)^T 转置获得 V
        thrust::device_vector<double> d_Vd;
        if (want_Vt()) {
            d_Vd.resize((size_t)rs_dim * (size_t)rs_dim);
            BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                                     rs_dim, rs_dim,
                                     &one_d, thrust::raw_pointer_cast(d_VT.data()), rs_dim,
                                     &zero_d, thrust::raw_pointer_cast(d_VT.data()), rs_dim,
                                     thrust::raw_pointer_cast(d_Vd.data()), rs_dim);
        }

        if (debug_mode && m_in > n_in && want_U() && want_Vt()) {
            // Build Qd
            thrust::device_vector<double> d_Qd((size_t)m_in * (size_t)n_in);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_Qd.data()), thrust::raw_pointer_cast(d_Q.data()), sizeof(double)*(size_t)m_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
            // U_final = Q * U
            thrust::device_vector<double> d_Uf((size_t)m_in * (size_t)n_in);
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)m_in, (int)n_in, (int)n_in,
                        &one_d, thrust::raw_pointer_cast(d_Qd.data()), (int)m_in,
                        thrust::raw_pointer_cast(d_Ud.data()), (int)rs_dim,
                        &zero_d, thrust::raw_pointer_cast(d_Uf.data()), (int)m_in);
            if (debug_mode) {
                double q_ortho = verify_orthogonality<double>(cublasHandle, (int)m_in, (int)n_in, thrust::raw_pointer_cast(d_Qd.data()), (int)m_in);
                double u_ortho = verify_orthogonality<double>(cublasHandle, (int)n_in, (int)n_in, thrust::raw_pointer_cast(d_Ud.data()), (int)n_in);
                double v_ortho = verify_orthogonality<double>(cublasHandle, (int)n_in, (int)n_in, thrust::raw_pointer_cast(d_Vd.data()), (int)n_in);
                printf("Orthogonality of Q_basis ||Q_basis'*Q_basis - I||_F = %.6e\n", q_ortho);
                printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", u_ortho);
                printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", v_ortho);
                // Reconstruction vs A_ref = Q*R
                thrust::device_vector<double> d_US((size_t)m_in * (size_t)n_in);
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Uf.data()), sizeof(double)*(size_t)m_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
                {
                    dim3 th2(16,16); dim3 gr2((int)((n_in + th2.x - 1)/th2.x), (int)((m_in + th2.y - 1)/th2.y));
                    scale_columns_by_vector_inplace_double<<<gr2, th2, 0, stream>>>(
                        thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Sd.data()), (int)m_in, (int)n_in, (int)n_in);
                }
                thrust::device_vector<double> d_Arec((size_t)m_in * (size_t)n_in);
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                            (int)m_in, (int)n_in, (int)n_in,
                            &one_d, thrust::raw_pointer_cast(d_US.data()), (int)m_in,
                            thrust::raw_pointer_cast(d_Vd.data()), (int)n_in,
                            &zero_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);
                thrust::device_vector<double> d_Rd((size_t)n_in * (size_t)n_in);
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_Rd.data()), thrust::raw_pointer_cast(d_Rsquare.data()), sizeof(double)*(size_t)n_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
                thrust::device_vector<double> d_Aref((size_t)m_in * (size_t)n_in);
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                            (int)m_in, (int)n_in, (int)n_in,
                            &one_d, thrust::raw_pointer_cast(d_Qd.data()), (int)m_in,
                            thrust::raw_pointer_cast(d_Rd.data()), (int)n_in,
                            &zero_d, thrust::raw_pointer_cast(d_Aref.data()), (int)m_in);
                BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         (int)m_in, (int)n_in,
                                         &one_d, thrust::raw_pointer_cast(d_Aref.data()), (int)m_in,
                                         &minus_one_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in,
                                         thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);
                double err=0.0, normA=0.0;
                cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Arec.data()), 1, &err);
                cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Aref.data()), 1, &normA);
                printf("Verifying Reconstruction Error ||A - Q*U_R*S*V^T||_F...\n");
                printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %.6e\n", err);
                printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %.6e\n", (normA>1e-18? err/normA : err));
            }
        } else if (m_in < n_in) {
            // V_final = Q_at * U, U_final = V
            thrust::device_vector<double> d_Qd((size_t)n_in * (size_t)m_in);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_Qd.data()), thrust::raw_pointer_cast(d_Q.data()), sizeof(double)*(size_t)n_in*(size_t)m_in, cudaMemcpyDeviceToDevice, stream);
            thrust::device_vector<double> d_Vf((size_t)n_in * (size_t)m_in);
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)n_in, (int)m_in, (int)m_in,
                        &one_d, thrust::raw_pointer_cast(d_Qd.data()), (int)n_in,
                        thrust::raw_pointer_cast(d_Ud.data()), (int)rs_dim,
                        &zero_d, thrust::raw_pointer_cast(d_Vf.data()), (int)n_in);
            if (debug_mode) {
                double u_ortho = verify_orthogonality<double>(cublasHandle, (int)m_in, (int)m_in, thrust::raw_pointer_cast(d_Vd.data()), (int)m_in);
                double v_ortho = verify_orthogonality<double>(cublasHandle, (int)n_in, (int)m_in, thrust::raw_pointer_cast(d_Vf.data()), (int)n_in);
                printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", u_ortho);
                printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", v_ortho);
                // A reconstruction
                thrust::device_vector<double> d_US((size_t)m_in * (size_t)m_in);
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Vd.data()), sizeof(double)*(size_t)m_in*(size_t)m_in, cudaMemcpyDeviceToDevice, stream);
                {
                    dim3 th2(16,16); dim3 gr2((int)((m_in + th2.x - 1)/th2.x), (int)((m_in + th2.y - 1)/th2.y));
                    scale_columns_by_vector_inplace_double<<<gr2, th2, 0, stream>>>(
                        thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Sd.data()), (int)m_in, (int)m_in, (int)m_in);
                }
                thrust::device_vector<double> d_Arec((size_t)m_in * (size_t)n_in);
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                            (int)m_in, (int)n_in, (int)m_in,
                            &one_d, thrust::raw_pointer_cast(d_US.data()), (int)m_in,
                            thrust::raw_pointer_cast(d_Vf.data()), (int)n_in,
                            &zero_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);
                // baseline A from d_A
                thrust::device_vector<double> d_Ad((size_t)m_in * (size_t)n_in);
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_Ad.data()), thrust::raw_pointer_cast(d_A.data()), sizeof(double)*(size_t)m_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
                BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         (int)m_in, (int)n_in,
                                         &one_d, thrust::raw_pointer_cast(d_Ad.data()), (int)m_in,
                                         &minus_one_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in,
                                         thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);
                double err=0.0, normA=0.0;
                cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Arec.data()), 1, &err);
                cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Ad.data()), 1, &normA);
                printf("Verifying Reconstruction Error ||A - U_R*S*V^T||_F...\n");
                printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %.6e\n", err);
                printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %.6e\n", (normA>1e-18? err/normA : err));
            }
        } else {
            if (debug_mode) {
                double u_ortho = verify_orthogonality<double>(cublasHandle, (int)rs_dim, (int)rs_dim, thrust::raw_pointer_cast(d_Ud.data()), (int)rs_dim);
                double v_ortho = verify_orthogonality<double>(cublasHandle, (int)rs_dim, (int)rs_dim, thrust::raw_pointer_cast(d_Vd.data()), (int)rs_dim);
                printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", u_ortho);
                printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", v_ortho);
                // Reconstruction on square M
                thrust::device_vector<double> d_US((size_t)rs_dim * (size_t)rs_dim);
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Ud.data()), sizeof(double)*(size_t)rs_dim*(size_t)rs_dim, cudaMemcpyDeviceToDevice, stream);
                {
                    dim3 th2(16,16); dim3 gr2((int)((rs_dim + th2.x - 1)/th2.x), (int)((rs_dim + th2.y - 1)/th2.y));
                    scale_columns_by_vector_inplace_double<<<gr2, th2, 0, stream>>>(
                        thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Sd.data()), (int)rs_dim, (int)rs_dim, (int)rs_dim);
                }
                thrust::device_vector<double> d_Arec((size_t)rs_dim * (size_t)rs_dim);
                cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                            (int)rs_dim, (int)rs_dim, (int)rs_dim,
                            &one_d, thrust::raw_pointer_cast(d_US.data()), (int)rs_dim,
                            thrust::raw_pointer_cast(d_Vd.data()), (int)rs_dim,
                            &zero_d, thrust::raw_pointer_cast(d_Arec.data()), (int)rs_dim);
                BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                         (int)rs_dim, (int)rs_dim,
                                         &one_d, (m_in==n_in ? thrust::raw_pointer_cast(d_Mref.data()) : thrust::raw_pointer_cast(d_Md.data())), (int)rs_dim,
                                         &minus_one_d, thrust::raw_pointer_cast(d_Arec.data()), (int)rs_dim,
                                         thrust::raw_pointer_cast(d_Arec.data()), (int)rs_dim);
                double err=0.0, normA=0.0;
                cublasDnrm2(cublasHandle, (int)((size_t)rs_dim*(size_t)rs_dim), thrust::raw_pointer_cast(d_Arec.data()), 1, &err);
                cublasDnrm2(cublasHandle, (int)((size_t)rs_dim*(size_t)rs_dim), (m_in==n_in ? thrust::raw_pointer_cast(d_Mref.data()) : thrust::raw_pointer_cast(d_Md.data())), 1, &normA);
                printf("Verifying Reconstruction Error ||A - U_R*S*V^T||_F...\n");
                printf("Reconstruction error ||A - U_R*S*V^T||_F = %.6e\n", err);
                printf("Relative reconstruction error ||A - U_R*S*V^T||_F / ||A||_F = %.6e\n", (normA>1e-18? err/normA : err));
            }
        }
        float elapsed_direct = stopTimer();
        printf("DirectSVD stage finished in %f ms.\n", elapsed_direct);
        return;
    }
    // Float/低精路径：按需求选择 Gram 矩阵
    // 只要 U → G = M M^T；只要 V^T → G = M^T M；若两侧都要 → 可任选其一并推导另一侧（此处沿用 M^T M）
    thrust::device_vector<double> d_G((size_t)rs_dim * (size_t)rs_dim);
    if (want_U() && !want_Vt()) {
        // G = M * M^T
        cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                    rs_dim, rs_dim, rs_dim,
                    &one_d,
                    thrust::raw_pointer_cast(d_Md.data()), rs_dim,
                    thrust::raw_pointer_cast(d_Md.data()), rs_dim,
                    &zero_d,
                    thrust::raw_pointer_cast(d_G.data()), rs_dim);
    } else {
        // 默认：G = M^T * M（得到 V）
        cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
                    rs_dim, rs_dim, rs_dim,
                    &one_d,
                    thrust::raw_pointer_cast(d_Md.data()), rs_dim,
                    thrust::raw_pointer_cast(d_Md.data()), rs_dim,
                    &zero_d,
                    thrust::raw_pointer_cast(d_G.data()), rs_dim);
    }
    // 对 G 做特征分解 EVD(G)
    thrust::device_vector<double> d_eig(rs_dim);
    size_t lw=0, lhw=0; thrust::device_vector<int> d_info(1);
    {
    auto jobz = need_vectors() ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
    cusolverDnXsyevd_bufferSize(cusolverHandle, nullptr, jobz,
                                CUBLAS_FILL_MODE_LOWER,
                                rs_dim, CUDA_R_64F,
                                thrust::raw_pointer_cast(d_G.data()), rs_dim,
                                CUDA_R_64F, thrust::raw_pointer_cast(d_eig.data()),
                                CUDA_R_64F, &lw, &lhw);
    thrust::device_vector<double> d_work(lw); void* hw=nullptr; if (lhw>0) hw=malloc(lhw);
    cusolverDnXsyevd(cusolverHandle, nullptr, jobz,
                     CUBLAS_FILL_MODE_LOWER,
                     rs_dim, CUDA_R_64F,
                     thrust::raw_pointer_cast(d_G.data()), rs_dim,
                     CUDA_R_64F, thrust::raw_pointer_cast(d_eig.data()),
                     CUDA_R_64F, thrust::raw_pointer_cast(d_work.data()), (size_t)lw,
                     hw, (size_t)lhw, thrust::raw_pointer_cast(d_info.data()));
    if (hw) free(hw);
    // S = sqrt(lambda)
    thrust::device_vector<double> d_Sd(rs_dim);
    computeSingularValues_double<<<(int)((rs_dim + 255)/256), 256, 0, stream>>>(
        thrust::raw_pointer_cast(d_eig.data()), thrust::raw_pointer_cast(d_Sd.data()), rs_dim);
    // 根据选择的 Gram，直接得到需要的一侧特征向量
    double* d_Vd_ptr = nullptr; // 右奇异向量 V（列）
    thrust::device_vector<double> d_Ud((size_t)rs_dim * (size_t)rs_dim); // 左奇异向量 U（列）
    if (want_U() && !want_Vt()) {
        // d_G 的列即 U
        cudaMemcpyAsync(thrust::raw_pointer_cast(d_Ud.data()), thrust::raw_pointer_cast(d_G.data()), sizeof(double)*(size_t)rs_dim*(size_t)rs_dim, cudaMemcpyDeviceToDevice, stream);
        // 无需推导 V
    } else {
        // d_G 的列即 V
        if (want_Vt()) d_Vd_ptr = thrust::raw_pointer_cast(d_G.data());
        if (want_U()) {
            // U = M * V，再按列除以 S
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        rs_dim, rs_dim, rs_dim,
                        &one_d,
                        thrust::raw_pointer_cast(d_Md.data()), rs_dim,
                        d_Vd_ptr, rs_dim,
                        &zero_d,
                        thrust::raw_pointer_cast(d_Ud.data()), rs_dim);
            dim3 th(16,16); dim3 gr((rs_dim + th.x - 1)/th.x, (rs_dim + th.y - 1)/th.y);
            divide_columns_by_diag_double<<<gr, th, 0, stream>>>(
                thrust::raw_pointer_cast(d_Ud.data()), thrust::raw_pointer_cast(d_Sd.data()), rs_dim, rs_dim, rs_dim, 1e-18);
        }
    }

    // 4) 利用上一步得到的薄 Q（若存在）装配最终的 U/V
    if (debug_mode && m_in > n_in && want_U() && want_Vt()) {
        // U_final = Q (m x n) * U_R (n x n), V_final = V_R (n x n)
        thrust::device_vector<double> d_Qd((size_t)m_in * (size_t)n_in);
        if constexpr (std::is_same_v<T_compute, double>) {
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_Qd.data()), thrust::raw_pointer_cast(d_Q.data()), sizeof(double)*(size_t)m_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
        } else {
            convert_kernel<T_compute, double><<<(int)((((size_t)m_in*(size_t)n_in)+255)/256), 256, 0, stream>>>(
                thrust::raw_pointer_cast(d_Q.data()), thrust::raw_pointer_cast(d_Qd.data()), (size_t)m_in*(size_t)n_in);
        }
        thrust::device_vector<double> d_Uf((size_t)m_in * (size_t)n_in);
        cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)m_in, (int)n_in, (int)n_in,
                    &one_d, thrust::raw_pointer_cast(d_Qd.data()), (int)m_in,
                    thrust::raw_pointer_cast(d_Ud.data()), (int)rs_dim,
                    &zero_d, thrust::raw_pointer_cast(d_Uf.data()), (int)m_in);
        if (debug_mode && want_U() && want_Vt()) {
            printf("[DirectSVD-EVD] m>n: U(m x n), S(n), V(n x n) computed via EVD.\n");
            // Orthogonality checks
            double q_ortho = verify_orthogonality<double>(cublasHandle, (int)m_in, (int)n_in, thrust::raw_pointer_cast(d_Qd.data()), (int)m_in);
            double u_ortho = verify_orthogonality<double>(cublasHandle, (int)n_in, (int)n_in, thrust::raw_pointer_cast(d_Ud.data()), (int)n_in);
            double v_ortho = verify_orthogonality<double>(cublasHandle, (int)n_in, (int)n_in, d_Vd_ptr, (int)n_in);
            printf("Orthogonality of Q_basis ||Q_basis'*Q_basis - I||_F = %.6e\n", q_ortho);
            printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", u_ortho);
            printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", v_ortho);
            // Reconstruction: A ≈ (Q*U_R) * S * V^T
            thrust::device_vector<double> d_US((size_t)m_in * (size_t)n_in);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Uf.data()), sizeof(double)*(size_t)m_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
            {
                dim3 th2(16,16); dim3 gr2((int)((n_in + th2.x - 1)/th2.x), (int)((m_in + th2.y - 1)/th2.y));
                scale_columns_by_vector_inplace_double<<<gr2, th2, 0, stream>>>(
                    thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Sd.data()), (int)m_in, (int)n_in, (int)n_in);
            }
            thrust::device_vector<double> d_Arec((size_t)m_in * (size_t)n_in);
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)m_in, (int)n_in, (int)n_in,
                        &one_d, thrust::raw_pointer_cast(d_US.data()), (int)m_in,
                        d_Vd_ptr, (int)n_in,
                        &zero_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);
            // A_ref = Q * R
            thrust::device_vector<double> d_Rd((size_t)n_in * (size_t)n_in);
            if constexpr (std::is_same_v<T_compute, double>) {
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_Rd.data()), thrust::raw_pointer_cast(d_Rsquare.data()), sizeof(double)*(size_t)n_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
            } else {
                convert_kernel<T_compute, double><<<(int)((((size_t)n_in*(size_t)n_in)+255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_Rsquare.data()), thrust::raw_pointer_cast(d_Rd.data()), (size_t)n_in*(size_t)n_in);
            }
            thrust::device_vector<double> d_Aref((size_t)m_in * (size_t)n_in);
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                        (int)m_in, (int)n_in, (int)n_in,
                        &one_d, thrust::raw_pointer_cast(d_Qd.data()), (int)m_in,
                        thrust::raw_pointer_cast(d_Rd.data()), (int)n_in,
                        &zero_d, thrust::raw_pointer_cast(d_Aref.data()), (int)m_in);
            // Error = Aref - Arec
            BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     (int)m_in, (int)n_in,
                                     &one_d, thrust::raw_pointer_cast(d_Aref.data()), (int)m_in,
                                     &minus_one_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in,
                                     thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);
            double err=0.0, normA=0.0;
            cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Arec.data()), 1, &err);
            cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Aref.data()), 1, &normA);
            printf("Verifying Reconstruction Error ||A - Q*U_R*S*V^T||_F...\n");
            printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %.6e\n", err);
            printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %.6e\n", (normA>1e-18? err/normA : err));
        }
    } else if (m_in < n_in) {
        // U_final = V_R (m x m), V_final = Q_at (n x m) * U_R (m x m)
        thrust::device_vector<double> d_Qd((size_t)n_in * (size_t)m_in);
        if constexpr (std::is_same_v<T_compute, double>) {
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_Qd.data()), thrust::raw_pointer_cast(d_Q.data()), sizeof(double)*(size_t)n_in*(size_t)m_in, cudaMemcpyDeviceToDevice, stream);
        } else {
            convert_kernel<T_compute, double><<<(int)((((size_t)n_in*(size_t)m_in)+255)/256), 256, 0, stream>>>(
                thrust::raw_pointer_cast(d_Q.data()), thrust::raw_pointer_cast(d_Qd.data()), (size_t)n_in*(size_t)m_in);
        }
        thrust::device_vector<double> d_Vf((size_t)n_in * (size_t)m_in);
        cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                    (int)n_in, (int)m_in, (int)m_in,
                    &one_d, thrust::raw_pointer_cast(d_Qd.data()), (int)n_in,
                    thrust::raw_pointer_cast(d_Ud.data()), (int)rs_dim,
                    &zero_d, thrust::raw_pointer_cast(d_Vf.data()), (int)n_in);
        if (debug_mode) {
            printf("[DirectSVD-EVD] m<n: U(m x m), S(m), V(n x m) computed via EVD.\n");
            double u_ortho = verify_orthogonality<double>(cublasHandle, (int)m_in, (int)m_in, d_Vd_ptr, (int)m_in);
            double v_ortho = verify_orthogonality<double>(cublasHandle, (int)n_in, (int)m_in, thrust::raw_pointer_cast(d_Vf.data()), (int)n_in);
            printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", u_ortho);
            printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", v_ortho);
            // A ≈ U_final * S * V_final^T = (V_M)*S*(Q_at*U_M)^T
            thrust::device_vector<double> d_US((size_t)m_in * (size_t)m_in);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_US.data()), d_Vd_ptr, sizeof(double)*(size_t)m_in*(size_t)m_in, cudaMemcpyDeviceToDevice, stream);
            {
                dim3 th2(16,16); dim3 gr2((int)((m_in + th2.x - 1)/th2.x), (int)((m_in + th2.y - 1)/th2.y));
                scale_columns_by_vector_inplace_double<<<gr2, th2, 0, stream>>>(
                    thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Sd.data()), (int)m_in, (int)m_in, (int)m_in);
            }
            thrust::device_vector<double> d_Arec((size_t)m_in * (size_t)n_in);
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)m_in, (int)n_in, (int)m_in,
                        &one_d, thrust::raw_pointer_cast(d_US.data()), (int)m_in,
                        thrust::raw_pointer_cast(d_Vf.data()), (int)n_in,
                        &zero_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);

            thrust::device_vector<double> d_Ad((size_t)m_in * (size_t)n_in);
            if constexpr (std::is_same_v<T_compute, double>) {
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_Ad.data()), thrust::raw_pointer_cast(d_A.data()), sizeof(double)*(size_t)m_in*(size_t)n_in, cudaMemcpyDeviceToDevice, stream);
            } else {
                convert_kernel<T_compute, double><<<(int)((((size_t)m_in*(size_t)n_in)+255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_Ad.data()), (size_t)m_in*(size_t)n_in);
            }
            // Error = A - Arec
            BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     (int)m_in, (int)n_in,
                                     &one_d, thrust::raw_pointer_cast(d_Ad.data()), (int)m_in,
                                     &minus_one_d, thrust::raw_pointer_cast(d_Arec.data()), (int)m_in,
                                     thrust::raw_pointer_cast(d_Arec.data()), (int)m_in);
            double err=0.0, normA=0.0;
            cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Arec.data()), 1, &err);
            cublasDnrm2(cublasHandle, (int)((size_t)m_in*(size_t)n_in), thrust::raw_pointer_cast(d_Ad.data()), 1, &normA);
            printf("Verifying Reconstruction Error ||A - U_R*S*V^T||_F...\n");
            printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %.6e\n", err);
            printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %.6e\n", (normA>1e-18? err/normA : err));
        }
    } else {
        if (debug_mode) {
            constexpr bool lowp_storage_path =
                std::is_same_v<T_storage, __half> || std::is_same_v<T_storage, __nv_bfloat16>;
            // 由于 EVD 计算的数值稳定性问题，可能需要对 U 进行二次细化
            double uo_before = verify_orthogonality<double>(cublasHandle, (int)rs_dim, (int)rs_dim, thrust::raw_pointer_cast(d_Ud.data()), (int)rs_dim);
            const double tol_ref = 5e-5;
            bool do_refine = false;
            if (g_accuracy == AccuracyLevel::High) do_refine = true;
            else if (g_accuracy == AccuracyLevel::Medium && uo_before > tol_ref) do_refine = true;
            if (do_refine) {
                // 二次EVD细化 U
                thrust::device_vector<double> d_Gu((size_t)rs_dim * (size_t)rs_dim);
                cublasDsyrk(cublasHandle, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                            (int)rs_dim, (int)rs_dim,
                            &one_d,
                            thrust::raw_pointer_cast(d_Md.data()), (int)rs_dim,
                            &zero_d,
                            thrust::raw_pointer_cast(d_Gu.data()), (int)rs_dim);
                size_t lw2=0, lhw2=0; thrust::device_vector<double> d_eig2(rs_dim); thrust::device_vector<int> di2(1);
                auto jobz_ref2 = need_vectors() ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
                cusolverDnXsyevd_bufferSize(cusolverHandle, nullptr, jobz_ref2,
                                            CUBLAS_FILL_MODE_LOWER,
                                            (int)rs_dim, CUDA_R_64F,
                                            thrust::raw_pointer_cast(d_Gu.data()), (int)rs_dim,
                                            CUDA_R_64F, thrust::raw_pointer_cast(d_eig2.data()),
                                            CUDA_R_64F, &lw2, &lhw2);
                thrust::device_vector<double> d_work2(lw2); void* hw2=nullptr; if (lhw2>0) hw2=malloc(lhw2);
                cusolverDnXsyevd(cusolverHandle, nullptr, jobz_ref2,
                                 CUBLAS_FILL_MODE_LOWER,
                                 (int)rs_dim, CUDA_R_64F,
                                 thrust::raw_pointer_cast(d_Gu.data()), (int)rs_dim,
                                 CUDA_R_64F, thrust::raw_pointer_cast(d_eig2.data()),
                                 CUDA_R_64F, thrust::raw_pointer_cast(d_work2.data()), (size_t)lw2,
                                 hw2, (size_t)lhw2, thrust::raw_pointer_cast(di2.data()));
                if (hw2) free(hw2);
                // 确保 Gu 的列与 Ud 同号
                thrust::device_vector<double> d_dots(rs_dim), d_sgn(rs_dim);
                column_dot_double<<<(int)rs_dim, 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_Gu.data()), thrust::raw_pointer_cast(d_Ud.data()),
                    thrust::raw_pointer_cast(d_dots.data()), (int)rs_dim, (int)rs_dim, (int)rs_dim);
                build_sign_vector_double<<<(int)((rs_dim + 255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_dots.data()), thrust::raw_pointer_cast(d_sgn.data()), (int)rs_dim);
                {
                    dim3 ths(16,16); dim3 grs((int)((rs_dim + ths.x - 1)/ths.x), (int)((rs_dim + ths.y - 1)/ths.y));
                    scale_columns_by_vector_inplace_double<<<grs, ths, 0, stream>>>(
                        thrust::raw_pointer_cast(d_Gu.data()), thrust::raw_pointer_cast(d_sgn.data()), (int)rs_dim, (int)rs_dim, (int)rs_dim);
                }
                // 替换 Ud
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_Ud.data()), thrust::raw_pointer_cast(d_Gu.data()), sizeof(double)*(size_t)rs_dim*(size_t)rs_dim, cudaMemcpyDeviceToDevice, stream);
            }
            // 最终正负号统一（确保奇异值非负）
            double u_ortho = verify_orthogonality<double>(cublasHandle, (int)rs_dim, (int)rs_dim, thrust::raw_pointer_cast(d_Ud.data()), (int)rs_dim);
            double v_ortho = verify_orthogonality<double>(cublasHandle, (int)rs_dim, (int)rs_dim, d_Vd_ptr, (int)rs_dim);
            // A ≈ U*S*V^T 
            thrust::device_vector<double> d_US((size_t)rs_dim * (size_t)rs_dim);
            cudaMemcpyAsync(thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Ud.data()), sizeof(double)*(size_t)rs_dim*(size_t)rs_dim, cudaMemcpyDeviceToDevice, stream);
            {
                dim3 th2(16,16); dim3 gr2((int)((rs_dim + th2.x - 1)/th2.x), (int)((rs_dim + th2.y - 1)/th2.y));
                scale_columns_by_vector_inplace_double<<<gr2, th2, 0, stream>>>(
                    thrust::raw_pointer_cast(d_US.data()), thrust::raw_pointer_cast(d_Sd.data()), (int)rs_dim, (int)rs_dim, (int)rs_dim);
            }
            thrust::device_vector<double> d_Arec((size_t)rs_dim * (size_t)rs_dim);
            cublasDgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                        (int)rs_dim, (int)rs_dim, (int)rs_dim,
                        &one_d, thrust::raw_pointer_cast(d_US.data()), (int)rs_dim,
                        d_Vd_ptr, (int)rs_dim,
                        &zero_d, thrust::raw_pointer_cast(d_Arec.data()), (int)rs_dim);
            // Error = A - Arec
            BlasCaller<double>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     (int)rs_dim, (int)rs_dim,
                                     &one_d, thrust::raw_pointer_cast(d_Md.data()), (int)rs_dim,
                                     &minus_one_d, thrust::raw_pointer_cast(d_Arec.data()), (int)rs_dim,
                                     thrust::raw_pointer_cast(d_Arec.data()), (int)rs_dim);
            double err=0.0, normA=0.0;
            cublasDnrm2(cublasHandle, (int)((size_t)rs_dim*(size_t)rs_dim), thrust::raw_pointer_cast(d_Arec.data()), 1, &err);
            cublasDnrm2(cublasHandle, (int)((size_t)rs_dim*(size_t)rs_dim), thrust::raw_pointer_cast(d_Md.data()), 1, &normA);
            if constexpr (lowp_storage_path) {
                thrust::device_vector<T_storage> d_U_storage((size_t)rs_dim * (size_t)rs_dim);
                thrust::device_vector<T_storage> d_V_storage((size_t)rs_dim * (size_t)rs_dim);
                thrust::device_vector<T_storage> d_S_storage((size_t)rs_dim);
                convert_kernel<double, T_storage><<<(int)((((size_t)rs_dim*(size_t)rs_dim)+255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_Ud.data()), thrust::raw_pointer_cast(d_U_storage.data()), (size_t)rs_dim*(size_t)rs_dim);
                convert_kernel<double, T_storage><<<(int)((((size_t)rs_dim*(size_t)rs_dim)+255)/256), 256, 0, stream>>>(
                    d_Vd_ptr, thrust::raw_pointer_cast(d_V_storage.data()), (size_t)rs_dim*(size_t)rs_dim);
                convert_kernel<double, T_storage><<<(int)(((size_t)rs_dim+255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_Sd.data()), thrust::raw_pointer_cast(d_S_storage.data()), (size_t)rs_dim);

                thrust::device_vector<T_compute> d_Uq((size_t)rs_dim * (size_t)rs_dim);
                thrust::device_vector<T_compute> d_Vq((size_t)rs_dim * (size_t)rs_dim);
                thrust::device_vector<T_compute> d_Sq((size_t)rs_dim);
                convert_kernel<T_storage, T_compute><<<(int)((((size_t)rs_dim*(size_t)rs_dim)+255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_U_storage.data()), thrust::raw_pointer_cast(d_Uq.data()), (size_t)rs_dim*(size_t)rs_dim);
                convert_kernel<T_storage, T_compute><<<(int)((((size_t)rs_dim*(size_t)rs_dim)+255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_V_storage.data()), thrust::raw_pointer_cast(d_Vq.data()), (size_t)rs_dim*(size_t)rs_dim);
                convert_kernel<T_storage, T_compute><<<(int)(((size_t)rs_dim+255)/256), 256, 0, stream>>>(
                    thrust::raw_pointer_cast(d_S_storage.data()), thrust::raw_pointer_cast(d_Sq.data()), (size_t)rs_dim);

                float u_ortho_q = verify_orthogonality<T_compute>(cublasHandle, (int)rs_dim, (int)rs_dim, thrust::raw_pointer_cast(d_Uq.data()), (int)rs_dim);
                float v_ortho_q = verify_orthogonality<T_compute>(cublasHandle, (int)rs_dim, (int)rs_dim, thrust::raw_pointer_cast(d_Vq.data()), (int)rs_dim);
                printf("[DirectSVD-EVD] %s-quantized factor view:\n", storage_prec_str);
                printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", (double)u_ortho_q);
                printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", (double)v_ortho_q);

                thrust::device_vector<T_compute> d_USq((size_t)rs_dim * (size_t)rs_dim);
                cudaMemcpyAsync(thrust::raw_pointer_cast(d_USq.data()), thrust::raw_pointer_cast(d_Uq.data()),
                                sizeof(T_compute)*(size_t)rs_dim*(size_t)rs_dim, cudaMemcpyDeviceToDevice, stream);
                {
                    dim3 thq(16,16); dim3 grq((int)((rs_dim + thq.x - 1)/thq.x), (int)((rs_dim + thq.y - 1)/thq.y));
                    scale_columns_by_vector_inplace_kernel<T_compute><<<grq, thq, 0, stream>>>(
                        thrust::raw_pointer_cast(d_USq.data()), thrust::raw_pointer_cast(d_Sq.data()), (int)rs_dim, (int)rs_dim, (int)rs_dim);
                }
                thrust::device_vector<T_compute> d_Arec_q((size_t)rs_dim * (size_t)rs_dim);
                const T_compute one_q = T_compute(1.0f);
                const T_compute zero_q = T_compute(0.0f);
                const T_compute minus_one_q = T_compute(-1.0f);
                BlasCaller<T_compute>::gemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
                                            (int)rs_dim, (int)rs_dim, (int)rs_dim,
                                            &one_q, thrust::raw_pointer_cast(d_USq.data()), (int)rs_dim,
                                            thrust::raw_pointer_cast(d_Vq.data()), (int)rs_dim,
                                            &zero_q, thrust::raw_pointer_cast(d_Arec_q.data()), (int)rs_dim);
                BlasCaller<T_compute>::geam(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                            (int)rs_dim, (int)rs_dim,
                                            &one_q, thrust::raw_pointer_cast(d_Rsquare.data()), (int)rs_dim,
                                            &minus_one_q, thrust::raw_pointer_cast(d_Arec_q.data()), (int)rs_dim,
                                            thrust::raw_pointer_cast(d_Arec_q.data()), (int)rs_dim);
                float err_q = 0.0f, normA_q = 0.0f;
                cublasSnrm2(cublasHandle, (int)((size_t)rs_dim*(size_t)rs_dim), thrust::raw_pointer_cast(d_Arec_q.data()), 1, &err_q);
                cublasSnrm2(cublasHandle, (int)((size_t)rs_dim*(size_t)rs_dim), thrust::raw_pointer_cast(d_Rsquare.data()), 1, &normA_q);
                printf("Verifying Reconstruction Error ||A_%s - U_R*S*V^T||_F...\n", storage_prec_str);
                printf("Reconstruction error ||A_%s - Q*U_R*S*V^T||_F = %.6e\n", storage_prec_str, (double)err_q);
                printf("Relative reconstruction error ||A_%s - Q*U_R*S*V^T||_F / ||A_%s||_F = %.6e\n",
                       storage_prec_str, storage_prec_str,
                       (normA_q > 1e-12f ? (double)err_q / (double)normA_q : (double)err_q));
            } else {
                printf("Orthogonality of U_R ||U_R'*U_R - I||_F = %.6e\n", u_ortho);
                printf("Orthogonality of V ||V'*V - I||_F = %.6e\n", v_ortho);
                printf("Verifying Reconstruction Error ||A - U_R*S*V^T||_F...\n");
                printf("Reconstruction error ||A - Q*U_R*S*V^T||_F = %.6e\n", err);
                printf("Relative reconstruction error ||A - Q*U_R*S*V^T||_F / ||A||_F = %.6e\n", (normA>1e-18? err/normA : err));
            }
        }
    }
    float elapsed_direct = stopTimer();
    printf("DirectSVD stage finished in %f ms.\n", elapsed_direct);
}
}

int main(int argc, char *argv[])
{
    if (parseArguments(argc, argv) == -1) {
        return 1;
    }

    cublasHandle_t cublasHandle;
    CUDA_CHECK(cudaFree(0));
    CUBLAS_CHECK(cublasCreate(&cublasHandle));
    cusolverDnHandle_t cusolverHandle;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverHandle));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(cublasHandle, stream));
    // 允许 Tensor Core（不强制 TF32），确保混合精度 GEMM 使用 Tensor Ops
    CUBLAS_CHECK(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));
    //cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverHandle, stream));
    g_stream = stream; 

    // 路径决策：满秩或小矩阵（阈值=512）走 DirectSVD，否则走 HQB_SVD
    bool use_direct_path = false;
    const int direct_svd_threshold = 512;
    const char* direct_reason = "hqb_default";
    if (g_unknown_rank_mode) {
        use_direct_path = false;
        direct_reason = "unknown_rank_mode";
    } else if (g_rank_type == RankType::Full) {
        use_direct_path = true;
        direct_reason = "rank_type=full";
    } else if (std::min(m, n) < direct_svd_threshold) {
        use_direct_path = true;
        direct_reason = "small_matrix";
    }
    printf("[RouteInit] solver=%s precision=%s m=%ld n=%ld rank=%ld threshold=%d reason=%s\n",
           use_direct_path ? "DirectSVD" : "HQB",
           precision_str.c_str(),
           m, n, rank,
           direct_svd_threshold,
           direct_reason);

    // 根据精度与路径调度
    if (precision_str == "float") {
        if (use_direct_path) DirectSVD<float, float>(cublasHandle, cusolverHandle, stream);
        else HQB_SVD<float, float>(cublasHandle, cusolverHandle, stream);
    } else if (precision_str == "double") {
        if (use_direct_path) DirectSVD<double, double>(cublasHandle, cusolverHandle, stream);
        else HQB_SVD<double, double>(cublasHandle, cusolverHandle, stream);
    } else if (precision_str == "half") {
        if (use_direct_path) DirectSVD<__half, float>(cublasHandle, cusolverHandle, stream);
        else HQB_SVD<__half, float>(cublasHandle, cusolverHandle, stream);
    } else if (precision_str == "bfloat16") {
        if (use_direct_path) DirectSVD<__nv_bfloat16, float>(cublasHandle, cusolverHandle, stream);
        else HQB_SVD<__nv_bfloat16, float>(cublasHandle, cusolverHandle, stream);
    }

    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverHandle);
    cudaStreamDestroy(stream);
    
    printf("\nExecution finished.\n");
    return 0;
}
