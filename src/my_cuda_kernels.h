#ifndef MY_CUDA_KERNELS_H  
#define MY_CUDA_KERNELS_H  

#include <thrust/device_vector.h> 
#include <thrust/copy.h>         
#include <unordered_map>
#include <mutex>
#include <vector>     
#include <cstdio>    
#include <algorithm>  
#include <cmath>      
#include <type_traits> 
#include <cublas_v2.h> 
#include <cublasLt.h>
#include <curand.h>    
#include <cusolverDn.h> 
#include <random>
#include <unordered_map>
#include <mutex>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdlib>
#include <cuda_runtime.h>

template<typename T> struct CudaDataType;
template<> struct CudaDataType<float> { static const cudaDataType_t value = CUDA_R_32F; };
template<> struct CudaDataType<double> { static const cudaDataType_t value = CUDA_R_64F; };
template<> struct CudaDataType<__half> { static const cudaDataType_t value = CUDA_R_16F; };
template<> struct CudaDataType<__nv_bfloat16> { static const cudaDataType_t value = CUDA_R_16BF; };
template<typename T> struct CudaComputeType;
template<> struct CudaComputeType<float> { static const cublasComputeType_t value = CUBLAS_COMPUTE_32F; };
template<> struct CudaComputeType<double> { static const cublasComputeType_t value = CUBLAS_COMPUTE_64F; };
template<> struct CudaComputeType<__half> { static const cublasComputeType_t value = CUBLAS_COMPUTE_32F; };

template<typename TA, typename TB, typename TC, typename TCompute>
cublasStatus_t gemmex(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                  int m, int n, int k,
                  const TCompute* alpha,
                  const TA* A, int lda,
                  const TB* B, int ldb,
                  const TCompute* beta,
                  TC* C, int ldc)
{
    // 对 FP16/BF16 输入使用 FP32 计算精度，以利用 Tensor Cores 并保持准确性
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    if constexpr (std::is_same_v<TCompute, double>) {
        compute_type = CUBLAS_COMPUTE_64F;
    }

    return cublasGemmEx(h, opA, opB, m, n, k,
                        alpha,
                        A, CudaDataType<TA>::value, lda,
                        B, CudaDataType<TB>::value, ldb,
                        beta,
                        C, CudaDataType<TC>::value, ldc,
                        compute_type,
                        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

// cuBLASLt 封装：可选更优算法 + Tensor Core，失败则回退到 cublasGemmEx
template<typename TA, typename TB, typename TC, typename TCompute>
cublasStatus_t gemmex_lt(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
               int m, int n, int k,
               const TCompute* alpha,
               const TA* A, int lda,
               const TB* B, int ldb,
               const TCompute* beta,
               TC* C, int ldc,
               cudaStream_t stream = 0)
{
    // 默认回退到 GemmEx；仅当 HQB_USE_LT=1 时启用 cuBLASLt 路径
    if (const char* env = std::getenv("HQB_USE_LT"); env == nullptr || env[0] == '0') {
        return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    // 默认使用 GemmEx；仅当 HQB_USE_LT=1 时启用 cuBLASLt
    if (const char* env = std::getenv("HQB_USE_LT"); env == nullptr || env[0] == '0') {
        // 调试打印所选路径
        if (const char* dbg = std::getenv("HQB_DEBUG"); dbg && dbg[0] != '0') {
            printf("[hqbtune] gemm: using cublasGemmEx (Lt off) m=%d n=%d k=%d\n", m, n, k);
        }
        return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    static cublasLtHandle_t lt = nullptr;
    if (lt == nullptr && cublasLtCreate(&lt) != CUBLAS_STATUS_SUCCESS) {
            return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    cudaDataType_t scale_type = CUDA_R_32F;
    if constexpr (std::is_same_v<TCompute, double>) {
        compute_type = CUBLAS_COMPUTE_64F;
        scale_type = CUDA_R_64F;
    }

    // desc/layout 缓存，减少重复创建/销毁开销
    struct LayoutKey {
        int m,n,k,lda,ldb,ldc; int ta,tb; int da,db,dc; int comp;
        bool operator==(const LayoutKey& o) const {
            return m==o.m&&n==o.n&&k==o.k&&lda==o.lda&&ldb==o.ldb&&ldc==o.ldc&&ta==o.ta&&tb==o.tb&&da==o.da&&db==o.db&&dc==o.dc&&comp==o.comp;
        }
    };
    struct LKHash { size_t operator()(const LayoutKey& x) const {
        size_t h=1469598103934665603ull; auto mix=[&](uint64_t v){ h^=v; h*=1099511628211ull;};
        mix(x.m);mix(x.n);mix(x.k);mix(x.lda);mix(x.ldb);mix(x.ldc);mix(x.ta);mix(x.tb);mix(x.da);mix(x.db);mix(x.dc);mix(x.comp);return h; } };
    struct DescLayouts { cublasLtMatmulDesc_t desc; cublasLtMatrixLayout_t A,B,C; };
    static std::unordered_map<LayoutKey, DescLayouts, LKHash> dl_cache; static std::mutex dl_mu;
    LayoutKey lkey{m,n,k,lda,ldb,ldc,(int)opA,(int)opB,(int)CudaDataType<TA>::value,(int)CudaDataType<TB>::value,(int)CudaDataType<TC>::value,(int)compute_type};
    DescLayouts dls{}; bool have_layout=false;
    {
        std::lock_guard<std::mutex> lk(dl_mu); dl_cache.find(lkey);
        auto it=dl_cache.find(lkey);
        if (it != dl_cache.end()) { dls = it->second; have_layout = true; }
    }
    if (!have_layout) {
        cublasLtMatmulDesc_t matmul_desc;
        if (cublasLtMatmulDescCreate(&matmul_desc, compute_type, scale_type) != CUBLAS_STATUS_SUCCESS) {
            cublasLtDestroy(lt);
            return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
        // 设置转置
        cublasOperation_t transa = opA, transb = opB;
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

        // A 布局（列主序）
        int a_rows = (opA == CUBLAS_OP_N) ? m : k;
        int a_cols = (opA == CUBLAS_OP_N) ? k : m;
        cublasLtMatrixLayout_t layoutA;
        if (cublasLtMatrixLayoutCreate(&layoutA, CudaDataType<TA>::value, a_rows, a_cols, lda) != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatmulDescDestroy(matmul_desc); cublasLtDestroy(lt);
            return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
        cublasLtOrder_t col = CUBLASLT_ORDER_COL;
        cublasLtMatrixLayoutSetAttribute(layoutA, CUBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col));

        // B 布局（列主序）
        int b_rows = (opB == CUBLAS_OP_N) ? k : n;
        int b_cols = (opB == CUBLAS_OP_N) ? n : k;
        cublasLtMatrixLayout_t layoutB;
        if (cublasLtMatrixLayoutCreate(&layoutB, CudaDataType<TB>::value, b_rows, b_cols, ldb) != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutDestroy(layoutA); cublasLtMatmulDescDestroy(matmul_desc); cublasLtDestroy(lt);
            return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
        cublasLtMatrixLayoutSetAttribute(layoutB, CUBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col));

        // C 布局（列主序）
        cublasLtMatrixLayout_t layoutC;
        if (cublasLtMatrixLayoutCreate(&layoutC, CudaDataType<TC>::value, m, n, ldc) != CUBLAS_STATUS_SUCCESS) {
            cublasLtMatrixLayoutDestroy(layoutB); cublasLtMatrixLayoutDestroy(layoutA); cublasLtMatmulDescDestroy(matmul_desc); cublasLtDestroy(lt);
            return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
        cublasLtMatrixLayoutSetAttribute(layoutC, CUBLASLT_MATRIX_LAYOUT_ORDER, &col, sizeof(col));
        dls = {matmul_desc, layoutA, layoutB, layoutC};
        std::lock_guard<std::mutex> lk(dl_mu); dl_cache.emplace(lkey, dls);
    }
    cublasLtMatmulDesc_t matmul_desc = dls.desc;
    cublasLtMatrixLayout_t layoutA = dls.A;
    cublasLtMatrixLayout_t layoutB = dls.B;
    cublasLtMatrixLayout_t layoutC = dls.C;

    // 工作区与算法缓存
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    static void* lt_ws = nullptr; static size_t lt_ws_bytes = 0; static std::once_flag ws_once;
    std::call_once(ws_once, [](){
        size_t want = 256ull << 20; // 默认 256MB，解锁更优 Lt 算法
        if (const char* e = std::getenv("HQB_LT_WS")) {
            long v = std::atol(e); if (v > 0) want = (size_t)v;
        }
        cudaMalloc(&lt_ws, want); lt_ws_bytes = want;
    });
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt_ws_bytes, sizeof(lt_ws_bytes));

    struct GemmKey {
        int m,n,k,lda,ldb,ldc; int ta,tb; int da,db,dc; int comp;
        bool operator==(const GemmKey& o) const {
            return m==o.m&&n==o.n&&k==o.k&&lda==o.lda&&ldb==o.ldb&&ldc==o.ldc&&ta==o.ta&&tb==o.tb&&da==o.da&&db==o.db&&dc==o.dc&&comp==o.comp;
        }
    };
    struct KeyHash { size_t operator()(const GemmKey& x) const {
        size_t h = 1469598103934665603ull; auto mix=[&](uint64_t v){ h^=v; h*=1099511628211ull; };
        mix(x.m); mix(x.n); mix(x.k); mix(x.lda); mix(x.ldb); mix(x.ldc); mix(x.ta); mix(x.tb); mix(x.da); mix(x.db); mix(x.dc); mix(x.comp); return h; } };
    static std::unordered_map<GemmKey, cublasLtMatmulAlgo_t, KeyHash> algo_cache;
    static std::mutex cache_mu;
    GemmKey key{m,n,k,lda,ldb,ldc,(int)opA,(int)opB,(int)CudaDataType<TA>::value,(int)CudaDataType<TB>::value,(int)CudaDataType<TC>::value,(int)compute_type};

    cublasLtMatmulAlgo_t algo{}; bool have_algo=false;
    {
        std::lock_guard<std::mutex> lk(cache_mu);
        auto it = algo_cache.find(key);
        if (it != algo_cache.end()) { algo = it->second; have_algo=true; }
    }
    if (!have_algo) {
        cublasLtMatmulHeuristicResult_t heur[16]; int found=0;
        if (cublasLtMatmulAlgoGetHeuristic(lt, matmul_desc, layoutA, layoutB, layoutC, layoutC, pref, 16, heur, &found) != CUBLAS_STATUS_SUCCESS || found==0) {
            // 回退
            cublasLtMatmulPreferenceDestroy(pref);
            cublasLtMatrixLayoutDestroy(layoutC); cublasLtMatrixLayoutDestroy(layoutB); cublasLtMatrixLayoutDestroy(layoutA); cublasLtMatmulDescDestroy(matmul_desc);
            return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        }
        algo = heur[0].algo;
        std::lock_guard<std::mutex> lk(cache_mu);
        algo_cache.emplace(key, algo);
    }

    cublasStatus_t st = cublasLtMatmul(lt, matmul_desc,
                        alpha,
                        A, layoutA,
                        B, layoutB,
                        beta,
                        C, layoutC,
                        C, layoutC,
                        &algo,
                        lt_ws, lt_ws_bytes,
                        stream);

    // 资源清理（desc/layout 已缓存，保留至进程结束）
    cublasLtMatmulPreferenceDestroy(pref);
    if (st != CUBLAS_STATUS_SUCCESS) {
        // 回退到 GemmEx
        return gemmex<TA,TB,TC,TCompute>(h, opA, opB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }
    return st;
}

template<typename T>
struct AbsFunctor {
    __host__ __device__ T operator()(T x) const { return std::abs(x); }
};

template<typename T>
struct GeThreshold {
    T t;
    __host__ __device__ bool operator()(T x) const { return std::abs(x) >= t; }
};
template<typename T> struct CurandCaller;

template<> struct CurandCaller<float> {
    static void generate(curandGenerator_t g, float* p, size_t n) {
        curandGenerateNormal(g, p, n, 0.0f, 1.0f);
    }
};

template<> struct CurandCaller<double> {
    static void generate(curandGenerator_t g, double* p, size_t n) {
        curandGenerateNormalDouble(g, p, n, 0.0, 1.0);
    }
};

template<typename T> struct BlasCaller;

template<> struct BlasCaller<float> {
    static void gemm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                     int m, int n, int k,
                     const float* alpha,
                     const float* A, int lda,
                     const float* B, int ldb,
                     const float* beta,
                     float* C, int ldc) {
        // 控制是否使用 TF32/TensorCore：默认开启以提速；HQB_TF32=0 可强制关闭（如 --debug）
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        if (const char* e = std::getenv("HQB_TF32")) {
            if (e[0] == '0') algo = CUBLAS_GEMM_DEFAULT;
        }
        cublasGemmEx(h, opA, opB, m, n, k,
                     alpha,
                     A, CUDA_R_32F, lda,
                     B, CUDA_R_32F, ldb,
                     beta,
                     C, CUDA_R_32F, ldc,
                     CUBLAS_COMPUTE_32F,
                     algo);
    }

    // Strided-batched GEMM (column-major)
    static cublasStatus_t gemm_strided_batched(cublasHandle_t h,
                                               cublasOperation_t opA, cublasOperation_t opB,
                                               int m, int n, int k,
                                               const float* alpha,
                                               const float* A, int lda, long long int strideA,
                                               const float* B, int ldb, long long int strideB,
                                               const float* beta,
                                               float* C, int ldc, long long int strideC,
                                               int batchCount) {
        // honor TF32 via HQB_TF32 env
        cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
        if (const char* e = std::getenv("HQB_TF32")) { if (e[0] == '0') algo = CUBLAS_GEMM_DEFAULT; }
        return cublasSgemmStridedBatched(h, opA, opB, m, n, k,
                                         alpha,
                                         A, lda, strideA,
                                         B, ldb, strideB,
                                         beta,
                                         C, ldc, strideC,
                                         batchCount);
    }
     static cublasStatus_t geam(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                              int m, int n, const float* alpha, const float* A, int lda,
                              const float* beta, const float* B, int ldb, float* C, int ldc) {
        return cublasSgeam(h, opA, opB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
    
    static cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                              const float* alpha, const float* A, int lda, float* B, int ldb) {
        return cublasStrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const float* x, int incx, float* result) {
    return cublasSnrm2(h, n, x, incx, result);
}
};

template<> struct BlasCaller<double> {
    static void gemm(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                     int m, int n, int k,
                     const double* alpha,
                     const double* A, int lda,
                     const double* B, int ldb,
                     const double* beta,
                     double* C, int ldc) {
         cublasGemmEx(h, opA, opB, m, n, k,
                     alpha,                        
                     A, CUDA_R_64F, lda,
                     B, CUDA_R_64F, ldb,
                     beta,                        
                     C, CUDA_R_64F, ldc,
                     CUBLAS_COMPUTE_64F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }

    static cublasStatus_t gemm_strided_batched(cublasHandle_t h,
                                               cublasOperation_t opA, cublasOperation_t opB,
                                               int m, int n, int k,
                                               const double* alpha,
                                               const double* A, int lda, long long int strideA,
                                               const double* B, int ldb, long long int strideB,
                                               const double* beta,
                                               double* C, int ldc, long long int strideC,
                                               int batchCount) {
        return cublasDgemmStridedBatched(h, opA, opB, m, n, k,
                                         alpha,
                                         A, lda, strideA,
                                         B, ldb, strideB,
                                         beta,
                                         C, ldc, strideC,
                                         batchCount);
    }
    static cublasStatus_t geam(cublasHandle_t h, cublasOperation_t opA, cublasOperation_t opB,
                              int m, int n, const double* alpha, const double* A, int lda,
                              const double* beta, const double* B, int ldb, double* C, int ldc) {
        return cublasDgeam(h, opA, opB, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
    }
                              
    static cublasStatus_t trsm(cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
                              cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
                              const double* alpha, const double* A, int lda, double* B, int ldb) {
        return cublasDtrsm(h, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
    }
    static cublasStatus_t nrm2(cublasHandle_t h, int n, const double* x, int incx, double* result) {
    return cublasDnrm2(h, n, x, incx, result);
}
};

// 列缩放
#ifndef HQB_NO_TEMPLATE_KERNEL_STUBS
template<typename T>
__global__ void scale_columns_kernel(T* M_out, const T* M_in, const T* vec, size_t rows, size_t cols, size_t ld_out, size_t ld_in)
{
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        T scale_factor = vec[col];
        M_out[col * ld_out + row] = M_in[col * ld_in + row] * scale_factor;
    }
}
#endif

// 非模板封装版本，避免某些编译环境下模板 __global__ 的 host stub 解析问题
__global__ void scale_columns_kernel_f(float* M_out, const float* M_in, const float* vec, int rows, int cols, int ld_out, int ld_in)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        float scale_factor = vec[col];
        M_out[col * ld_out + row] = M_in[col * ld_in + row] * scale_factor;
    }
}
__global__ void scale_columns_kernel_d(double* M_out, const double* M_in, const double* vec, int rows, int cols, int ld_out, int ld_in)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        double scale_factor = vec[col];
        M_out[col * ld_out + row] = M_in[col * ld_in + row] * scale_factor;
    }
}

// 生成正态分布矩阵
template <typename T>
void generateNormalMatrix(curandGenerator_t& gen, thrust::device_vector<T>& d_vec, size_t m, size_t n)
{
    T* raw_ptr = thrust::raw_pointer_cast(d_vec.data());
    CurandCaller<T>::generate(gen, raw_ptr, m * n);
}

// 提取对角线
template<typename T>
__global__ void extractDiagonal(T* diag_vector, const T* R_block, size_t ld_R, size_t ost, size_t block_width) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < block_width) {
        diag_vector[i] = R_block[(i + ost) * ld_R + (i + ost)];
    }
}

// 计算上三角矩阵每行从对角线开始的 2-范数，并按尾部长度归一化（第0行除以width，第1行除以width-1，...）
template<typename T>
__global__ void compute_row_tail_norms_kernel(T* norms,
                                              const T* R_block,
                                              int ld,
                                              int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= width) return;

    double sum = 0.0;
    for (int col = row; col < width; ++col) {
        double v = static_cast<double>(R_block[col * ld + row]);
        sum += v * v;
    }
    int tail_len = width - row;
    double norm = sqrt(sum);
    norms[row] = static_cast<T>(norm / static_cast<double>(tail_len));
}

// 在对角线上加 alpha
template<typename T>
__global__ void add_identity_diagonal_kernel(
    T* matrix, 
    long int rows, 
    long int cols, 
    long int lda,
    T alpha = T(1) 
) {
    long int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && i < cols) {
        matrix[i * lda + i] += alpha;
    }
}

// LU分解提取L
template<typename T>
__global__ void extract_L_factor_kernel(
    T*       L_out, 
    long int      lda_out,
    const T* LU_in, 
    long int      lda_in,
    long int      rows, 
    long int      cols
) {
    long int c = blockIdx.x * blockDim.x + threadIdx.x;
    long int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        long int out_idx = c * lda_out + r;
        if (r < c) {
            L_out[out_idx] = T(0);
        } else if (r == c) {
            L_out[out_idx] = T(1);
        } else {
            long int in_idx = c * lda_in + r;
            L_out[out_idx] = LU_in[in_idx];
        }
    }
}

// 选择列
template<typename T>
__global__ void select_columns_kernel(
    T*       dest,        
    const int     ld_dest,
    const T* src,         
    const int     ld_src,
    const int*    indices,
    const int     rows,
    const int     num_new_cols
) {
    int dest_col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && dest_col < num_new_cols) {
        int src_col = indices[dest_col];
        size_t src_idx = (size_t)row + (size_t)src_col * ld_src;
        size_t dest_idx = (size_t)row + (size_t)dest_col * ld_dest;
        dest[dest_idx] = src[src_idx];
    }
}

// （已回滚）行向量化列选择实现：在性能评估后移除，恢复原核

// 递归计算 W
template <typename T>
void recursive_compute_w(cublasHandle_t handle,
                         size_t m,
                         size_t total_cols,
                         size_t block_size,
                         T* d_W, size_t ldw,
                         const T* d_Y, size_t ldy,
                         T* d_work_C)
{
    const T one = T(1);
    const T zero = T(0);
    const T minus_one = T(-1);

    // 取出与句柄绑定的流，传给 Lt 封装以避免落到默认流
    cudaStream_t hstream = nullptr;
    cublasGetStream(handle, &hstream);

    for (size_t i = block_size; i < total_cols; i += block_size) {
        size_t left_width = i;
        size_t right_width = std::min(block_size, total_cols - i);
        if (right_width == 0) continue;

        const T* Y_left = d_Y;
        const T* W_left = d_W;
        T*       W_right = d_W + left_width * ldw;

        // C = Y_left^T * W_right
        // 使用 gemmex_lt（未启用 Lt 时自动回退到 GemmEx），可选启用 TF32/TensorCore
        gemmex_lt<T,T, T, T>(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                             (int)left_width, (int)right_width, (int)m,
                             &one,
                             Y_left, (int)ldy,
                             W_right, (int)ldw,
                             &zero,
                             d_work_C, (int)left_width,
                             hstream);

        // W_right = W_right - W_left * C
        gemmex_lt<T,T, T, T>(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)m, (int)right_width, (int)left_width,
                             &minus_one,
                             W_left, (int)ldw,
                             d_work_C, (int)left_width,
                             &one,
                             W_right, (int)ldw,
                             hstream);
    }
}

// 验证正交性
template<typename T>
T verify_orthogonality(cublasHandle_t handle, int rows, int cols, T* d_X_ptr, int ldx) {
    thrust::device_vector<T> d_XTX((size_t)cols * cols);
    T* d_XTX_ptr = thrust::raw_pointer_cast(d_XTX.data());
    const T one = T(1), zero = T(0), minus_one = T(-1);
    // XTX = X^T * X
    BlasCaller<T>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        cols, cols, rows,
                        &one, d_X_ptr, ldx,
                        d_X_ptr, ldx,
                        &zero, d_XTX_ptr, cols);
    // XTX -= I (in place) on device
    cudaStream_t s = nullptr; cublasGetStream(handle, &s);
    add_identity_diagonal_kernel<T><<< (cols + 255)/256, 256, 0, s >>>(d_XTX_ptr, cols, cols, cols, minus_one);
    // ||XTX||_F
    T fn = 0;
    BlasCaller<T>::nrm2(handle, (int)((size_t)cols*cols), d_XTX_ptr, 1, &fn);
    return fn;
}

// 提取上三角矩阵
template<typename T>
__global__ void extract_upper_triangle_kernel(
    T*       dest,
    const int     ld_dest,
    const T* src,
    const int     ld_src,
    const int     rows,
    const int     cols
) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < rows && j < cols) {
        size_t src_idx = (size_t)i + (size_t)j * ld_src;
        size_t dest_idx = (size_t)i + (size_t)j * ld_dest;
        if (i <= j) {
            dest[dest_idx] = src[src_idx];
        } else {
            dest[dest_idx] = T(0);
        }
    }
}
// 计算奇异值
template<typename T>
__global__ void computeSingularValues_kernel(const T* eigenvals, T* singularvals, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        singularvals[idx] = std::sqrt(std::abs(eigenvals[idx]));
    }
}
// 计算奇异值的倒数
template<typename T>
__global__ void computeInverseSingularValues_kernel(const T* singularvals, T* inv_singularvals,
                                                    int n, T threshold = T(1e-12)) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (singularvals[idx] > threshold) {
            inv_singularvals[idx] = T(1.0) / singularvals[idx];
        } else {
            inv_singularvals[idx] = T(0);
        }
    }
}
// float <-> double 转换
__global__ void convertFloatToDouble(const float* src, double* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<double>(src[idx]);
}
__global__ void convertDoubleToFloat(const double* src, float* dst, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<float>(src[idx]);
}
// 拷贝矩阵
template<typename T>
__global__ void copyMatrix_kernel(const T* src, T* dst, int rows, int cols, int ld_src, int ld_dst) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < rows && j < cols) {
        dst[j * ld_dst + i] = src[j * ld_src + i];
    }
}

// 将 dst 的 [row_start, rows_total) 行在每一列清零（用于对齐填充区域置零）
template<typename T>
__global__ void zero_rows_kernel(T* dst, int ld_dst, int row_start, int rows_total, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < (rows_total - row_start)) {
        dst[col * ld_dst + (row_start + row)] = T(0);
    }
}
// 列缩放 inplace
template<typename T>
__global__ void scale_columns_by_vector_inplace_kernel(T* A, const T* s, int m, int n, int ld) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        A[col * ld + row] *= s[col];
    }
}
// 每列点积
template<typename T>
__global__ void column_dot_kernel(const T* __restrict__ A, const T* __restrict__ B,
                                  T* __restrict__ dots, int m, int n, int ld) {
    int col = blockIdx.x;
    if (col >= n) return;
    T sum = T(0);
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        sum += A[col * ld + i] * B[col * ld + i];
    }
    __shared__ T sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[col] = sh[0];
}

// 构建符号向量
template<typename T>
__global__ void build_sign_vector_kernel(const T* __restrict__ dots, T* __restrict__ sgn, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sgn[i] = (dots[i] >= T(0) ? T(1) : T(-1));
}

// 列除以对角向量
#ifndef HQB_NO_TEMPLATE_KERNEL_STUBS
template<typename T>
__global__ void divide_columns_by_diag_kernel(T* A, const T* S, int m, int n, int ld, T eps) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        T s = S[col];
        T val = (std::abs(s) > eps) ? (A[col * ld + row] / s) : T(0);
        A[col * ld + row] = val;
    }
}
#endif

// 非模板封装：float 版本，避免模板核在某些环境中的 stub 解析问题
__global__ void divide_columns_by_diag_kernel_f(float* A, const float* S, int m, int n, int ld, float eps) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        float s = S[col];
        float val = (fabsf(s) > eps) ? (A[col * ld + row] / s) : 0.0f;
        A[col * ld + row] = val;
    }
}

// 随机采样正交性检验
template<typename T>
static bool sample_orthogonality(cublasHandle_t handle, int rows, int cols,
                                 const T* d_mat, int ld,
                                 int num_samples, T tol,
                                 const char* name) {
    if (rows <= 0 || cols <= 0) return true;
    std::mt19937 rng(1234567);
    std::uniform_int_distribution<int> coldist(0, cols - 1);

    T max_diag_dev = T(0);
    T max_offdiag = T(0);

    for (int s = 0; s < num_samples; ++s) {
        int i = coldist(rng);
        const T* col = d_mat + (size_t)i * ld;
        T dot = T(0);
        BlasCaller<T>::dot(handle, rows, col, 1, col, 1, &dot);
        T dev = std::abs(dot - T(1));
        if (dev > max_diag_dev) max_diag_dev = dev;
    }

    for (int s = 0; s < num_samples; ++s) {
        int i = coldist(rng), j = coldist(rng);
        if (cols > 1) while (j == i) j = coldist(rng);
        const T* coli = d_mat + (size_t)i * ld;
        const T* colj = d_mat + (size_t)j * ld;
        T dot = T(0);
        BlasCaller<T>::dot(handle, rows, coli, 1, colj, 1, &dot);
        T val = std::abs(dot);
        if (val > max_offdiag) max_offdiag = val;
    }

    bool pass = (max_diag_dev <= tol) && (max_offdiag <= tol);
    printf("[Sample Ortho] %s: max |x_i^T x_i - 1| = %e, max |x_i^T x_j| = %e, tol=%e => %s\n",
           name, (double)max_diag_dev, (double)max_offdiag, (double)tol,
           pass ? "PASS" : "FAIL");
    return pass;
}

// 全量正交性检验
template<typename T>
static T verify_orthogonality(cublasHandle_t handle, int rows, int cols,
                              const T* d_X_ptr, int ldx) {
    thrust::device_vector<T> d_XTX((size_t)cols * cols);
    T* d_XTX_ptr = thrust::raw_pointer_cast(d_XTX.data());
    const T one = T(1), zero = T(0), minus_one = T(-1);
    BlasCaller<T>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                        cols, cols, rows,
                        &one, d_X_ptr, ldx,
                        d_X_ptr, ldx,
                        &zero, d_XTX_ptr, cols);
    cudaStream_t s = nullptr; cublasGetStream(handle, &s);
    add_identity_diagonal_kernel<T><<< (cols + 255)/256, 256, 0, s >>>(d_XTX_ptr, cols, cols, cols, minus_one);
    T fn = 0; BlasCaller<T>::nrm2(handle, (int)((size_t)cols*cols), d_XTX_ptr, 1, &fn);
    return fn;
}

cudaEvent_t begin, end;
// 全局计时绑定的 stream（在 main 中赋值）
static cudaStream_t g_stream = 0;
void startTimer()
{
    cudaEventCreate(&begin);
    cudaEventRecord(begin, g_stream);
    cudaEventCreate(&end);
}

float stopTimer()
{
    cudaEventRecord(end, g_stream);
    cudaEventSynchronize(end);
    float milliseconds;
    cudaEventElapsedTime(&milliseconds, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return milliseconds;
}
template<typename T>
void generate_singular_values(
    const std::string& type,
    T max_val,
    T min_val,
    std::vector<T>& s_values,
    size_t rank)
{
    s_values.resize(rank);
    if (rank == 0) return;

    if (type == "geometric") {
        if (rank <= 1) {
            if (rank == 1) s_values[0] = max_val;
            return;
        }
        size_t split_point = rank / 2;
        T log_max = std::log(max_val);
        T log_min = std::log(min_val);
        T log_mid = log_max + (log_min - log_max) * T(0.75);
        T mid_val = std::exp(log_mid);

        if (split_point > 0) {
            T step1 = (log_mid - log_max) / T(split_point);
            for (size_t i = 0; i < split_point; ++i) {
                s_values[i] = std::exp(log_max + T(i) * step1);
            }
        }
        size_t remaining_rank = rank - split_point;
        if (remaining_rank > 0) {
            T step2 = (log_min - log_mid) / T(remaining_rank - 1);
            for (size_t i = 0; i < remaining_rank; ++i) {
                s_values[split_point + i] = std::exp(log_mid + T(i) * step2);
            }
        }
    }
    else if (type == "uniform" || type == "arithmetic") {
        printf("Generating UNIFORMLY (linearly) distributed singular values...\n");
        if (rank == 1) {
            s_values[0] = max_val;
        } else {
            T step = (max_val - min_val) / T(rank - 1);
            for (size_t i = 0; i < rank; ++i) {
                s_values[i] = max_val - T(i) * step;
            }
        }
    }
    else if (type == "cluster0") {
        printf("Generating 'Cluster0' singular values (fixed 128 decay)...\n");
        const size_t cutoff_rank = 128; 
        size_t actual_cutoff = std::min(rank, cutoff_rank);
        T high_end_val = max_val * T(0.9);
        T step = (actual_cutoff > 1) ? (max_val - high_end_val) / T(actual_cutoff - 1) : T(0.0);
        
        for (size_t i = 0; i < rank; ++i) {
            if (i < actual_cutoff) {
                s_values[i] = max_val - T(i) * step;
            } else {
                s_values[i] = min_val;
            }
        }
    }
    else if (type == "cluster1") {
        printf("Generating 'Cluster1' singular values (staircase)...\n");
        size_t cutoff1 = rank / 3;
        size_t cutoff2 = 2 * rank / 3;
        T mid_val = (max_val + min_val) / T(2.0);

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
    else if (type == "normal") {
        printf("Generating NORMAL (Gaussian-like) distributed singular values...\n");
        T mean = T(rank) / T(2.0);
        T sigma = T(rank) / T(6.0); 
        
        for (size_t i = 0; i < rank; ++i) {
            T x = T(i);
            T gaussian_weight = std::exp(T(-0.5) * std::pow((x - mean) / sigma, T(2.0)));
            s_values[i] = min_val + (max_val - min_val) * gaussian_weight;
        }
    }
}
// 为 cuSOLVER 创建模板包装器 
template<typename T> struct SolverCaller;

template<> struct SolverCaller<float> {
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t h, int m, int n, float* A, int lda, int* lwork) {
        return cusolverDnSgeqrf_bufferSize(h, m, n, A, lda, lwork);
    }
    static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, float* A, int lda, float* tau, float* work, int lwork, int* devInfo) {
        return cusolverDnSgeqrf(h, m, n, A, lda, tau, work, lwork, devInfo);
    }
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t h, int m, int n, int k, float* A, int lda, const float* tau, int* lwork) {
        return cusolverDnSorgqr_bufferSize(h, m, n, k, A, lda, tau, lwork);
    }
    static cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k, float* A, int lda, const float* tau, float* work, int lwork, int* devInfo) {
        return cusolverDnSorgqr(h, m, n, k, A, lda, tau, work, lwork, devInfo);
    }
    static cusolverStatus_t ormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans,
                                             int m, int n, int k, const float* A, int lda, const float* tau,
                                             const float* C, int ldc, int* lwork) {
        return cusolverDnSormqr_bufferSize(h, (cublasSideMode_t)side, (cublasOperation_t)trans, m, n, k, A, lda, tau, C, ldc, lwork);
    }
    static cusolverStatus_t ormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans,
                                  int m, int n, int k, const float* A, int lda, const float* tau,
                                  float* C, int ldc, float* work, int lwork, int* devInfo) {
        return cusolverDnSormqr(h, (cublasSideMode_t)side, (cublasOperation_t)trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
    }
     static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t h, int m, int n, float* A, int lda, int* lwork) {
        return cusolverDnSgetrf_bufferSize(h, m, n, A, lda, lwork);
    }
    static cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, float* A, int lda, float* work, int* p, int* info) {
        return cusolverDnSgetrf(h, m, n, A, lda, work, p, info);
    }
};

template<> struct SolverCaller<double> {
    static cusolverStatus_t geqrf_bufferSize(cusolverDnHandle_t h, int m, int n, double* A, int lda, int* lwork) {
        return cusolverDnDgeqrf_bufferSize(h, m, n, A, lda, lwork);
    }
    static cusolverStatus_t geqrf(cusolverDnHandle_t h, int m, int n, double* A, int lda, double* tau, double* work, int lwork, int* devInfo) {
        return cusolverDnDgeqrf(h, m, n, A, lda, tau, work, lwork, devInfo);
    }
    static cusolverStatus_t orgqr_bufferSize(cusolverDnHandle_t h, int m, int n, int k, double* A, int lda, const double* tau, int* lwork) {
        return cusolverDnDorgqr_bufferSize(h, m, n, k, A, lda, tau, lwork);
    }
    static cusolverStatus_t orgqr(cusolverDnHandle_t h, int m, int n, int k, double* A, int lda, const double* tau, double* work, int lwork, int* devInfo) {
        return cusolverDnDorgqr(h, m, n, k, A, lda, tau, work, lwork, devInfo);
    }
    static cusolverStatus_t ormqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans,
                                             int m, int n, int k, const double* A, int lda, const double* tau,
                                             const double* C, int ldc, int* lwork) {
        return cusolverDnDormqr_bufferSize(h, (cublasSideMode_t)side, (cublasOperation_t)trans, m, n, k, A, lda, tau, C, ldc, lwork);
    }
    static cusolverStatus_t ormqr(cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans,
                                  int m, int n, int k, const double* A, int lda, const double* tau,
                                  double* C, int ldc, double* work, int lwork, int* devInfo) {
        return cusolverDnDormqr(h, (cublasSideMode_t)side, (cublasOperation_t)trans, m, n, k, A, lda, tau, C, ldc, work, lwork, devInfo);
    }
    static cusolverStatus_t getrf_bufferSize(cusolverDnHandle_t h, int m, int n, double* A, int lda, int* lwork) {
        return cusolverDnDgetrf_bufferSize(h, m, n, A, lda, lwork);
    }
    static cusolverStatus_t getrf(cusolverDnHandle_t h, int m, int n, double* A, int lda, double* work, int* p, int* info) {
        return cusolverDnDgetrf(h, m, n, A, lda, work, p, info);
    }
};
#ifndef HQB_NO_TEMPLATE_KERNEL_STUBS
template<typename Tin, typename Tout>
__global__ void convert_kernel_strided(const Tin* src, size_t ld_src, 
                                        Tout* dst, size_t ld_dst, 
                                        size_t rows, size_t cols) 
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < cols && row < rows) {
        size_t src_idx = col * ld_src + row;
        size_t dst_idx = col * ld_dst + row;
        dst[dst_idx] = static_cast<Tout>(src[src_idx]);
    }
}
#endif

// 显式非模板封装：float -> double 与 double -> float
// 目的：在某些主机编译器/环境下，避免对模板 __global__ 内核的显式实参形式产生的 stub 解析问题
__global__ void convert_kernel_strided_f2d(const float* src, size_t ld_src,
                                           double* dst, size_t ld_dst,
                                           size_t rows, size_t cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        size_t src_idx = col * ld_src + row;
        size_t dst_idx = col * ld_dst + row;
        dst[dst_idx] = static_cast<double>(src[src_idx]);
    }
}

__global__ void convert_kernel_strided_d2f(const double* src, size_t ld_src,
                                           float* dst, size_t ld_dst,
                                           size_t rows, size_t cols)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        size_t src_idx = col * ld_src + row;
        size_t dst_idx = col * ld_dst + row;
        dst[dst_idx] = static_cast<float>(src[src_idx]);
    }
}

// 半精度/bfloat16 → float 的非模板标量转换核
__global__ void convert_kernel_strided_h2f(const __half* src, int ld_src,
                                           float* dst, int ld_dst,
                                           int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        int src_idx = col * ld_src + row;
        int dst_idx = col * ld_dst + row;
        dst[dst_idx] = __half2float(src[src_idx]);
    }
}

__global__ void convert_kernel_strided_bf162f(const __nv_bfloat16* src, int ld_src,
                                              float* dst, int ld_dst,
                                              int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        int src_idx = col * ld_src + row;
        int dst_idx = col * ld_dst + row;
        dst[dst_idx] = __bfloat162float(src[src_idx]);
    }
}

// float -> half 标量步长转换
__global__ void convert_kernel_strided_f2h(const float* src, int ld_src,
                                           __half* dst, int ld_dst,
                                           int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        int src_idx = col * ld_src + row;
        int dst_idx = col * ld_dst + row;
        dst[dst_idx] = __float2half(src[src_idx]);
    }
}

// float -> bfloat16 标量步长转换
__global__ void convert_kernel_strided_f2bf16(const float* src, int ld_src,
                                              __nv_bfloat16* dst, int ld_dst,
                                              int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        int src_idx = col * ld_src + row;
        int dst_idx = col * ld_dst + row;
        dst[dst_idx] = __float2bfloat16(src[src_idx]);
    }
}

// 带缩放的步长转换（逐元素）：dst[col*ld_dst+row] = (Tin)src[...] * scale 后再转换为 Tout
template<typename Tin, typename Tout>
__global__ void convert_kernel_strided_scaled(const Tin* src, size_t ld_src,
                                              Tout* dst, size_t ld_dst,
                                              size_t rows, size_t cols,
                                              float scale)
{
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        size_t src_idx = col * ld_src + row;
        size_t dst_idx = col * ld_dst + row;
        float v = static_cast<float>(src[src_idx]) * scale;
        if constexpr (std::is_same_v<Tout, __half>) {
            dst[dst_idx] = __float2half(v);
        } else if constexpr (std::is_same_v<Tout, __nv_bfloat16>) {
            dst[dst_idx] = __float2bfloat16(v);
        } else {
            dst[dst_idx] = static_cast<Tout>(v);
        }
    }
}
template<typename Tin, typename Tout, int VECLEN>
struct VecType;

template<> struct VecType<float, float, 4> { using type = float4; };
template<> struct VecType<__half, float, 4> { using type_in = int2; using type_out = float4; }; // 4个half是int2
template<> struct VecType<float, __half, 4> { using type_in = float4; using type_out = int2; }; // 4个half是int2

// template<typename Tin, typename Tout>
// __global__ void convert_kernel_strided_vec4(const Tin* src, size_t ld_src, 
//                                             Tout* dst, size_t ld_dst, 
//                                             size_t rows, size_t cols) 
// {
//     const int VECLEN = 4;
//     size_t row_start = (blockIdx.y * blockDim.y + threadIdx.y) * VECLEN;
//     size_t col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (col < cols && row_start < rows) {
//         if constexpr (std::is_same_v<Tin, float> && std::is_same_v<Tout, __half>) {
//             // float -> half
//             const float4* src_v4 = reinterpret_cast<const float4*>(&src[col * ld_src + row_start]);
//             __half2* dst_v2 = reinterpret_cast<__half2*>(&dst[col * ld_dst + row_start]);

//             float4 temp_f4 = *src_v4;
            
//             // 使用 __floats2half2_rn 将两个float转换为一个__half2
//             // 这个函数接收两个独立的 float 参数
//             dst_v2[0] = __floats2half2_rn(temp_f4.x, temp_f4.y);
//             dst_v2[1] = __floats2half2_rn(temp_f4.z, temp_f4.w);

//         } else if constexpr (std::is_same_v<Tin, __half> && std::is_same_v<Tout, float>) {
//             // Case: half -> float
//             const __half2* src_v2 = reinterpret_cast<const __half2*>(&src[col * ld_src + row_start]);
//             float4* dst_v4 = reinterpret_cast<float4*>(&dst[col * ld_dst + row_start]);
            
//             __half2 temp_h2_0 = src_v2[0];
//             __half2 temp_h2_1 = src_v2[1];

//             // __half22float2 将一个__half2转换为一个float2
//             float2 temp_f2_0 = __half22float2(temp_h2_0);
//             float2 temp_f2_1 = __half22float2(temp_h2_1);

//             *dst_v4 = make_float4(temp_f2_0.x, temp_f2_0.y, temp_f2_1.x, temp_f2_1.y);
            
//         } else if constexpr (std::is_same_v<Tin, float> && std::is_same_v<Tout, float>) {
//             // float -> float
//             const float4* src_v4 = reinterpret_cast<const float4*>(&src[col * ld_src + row_start]);
//             float4* dst_v4 = reinterpret_cast<float4*>(&dst[col * ld_dst + row_start]);
//             *dst_v4 = *src_v4;
            
//         } else {
//             #pragma unroll
//             for (int i = 0; i < VECLEN; ++i) {
//                 if (row_start + i < rows) {
//                     dst[col * ld_dst + row_start + i] = static_cast<Tout>(src[col * ld_src + row_start + i]);
//                 }
//             }
//         }
//     }
// }

// // 带缩放的 vec4 步长转换（主要覆盖 float->half 的高效路径）
// template<typename Tin, typename Tout>
// __global__ void convert_kernel_strided_vec4_scaled(const Tin* src, size_t ld_src,
//                                                   Tout* dst, size_t ld_dst,
//                                                   size_t rows, size_t cols,
//                                                   float scale)
// {
//     const int VECLEN = 4;
//     size_t row_start = (blockIdx.y * blockDim.y + threadIdx.y) * VECLEN;
//     size_t col = blockIdx.x * blockDim.x + threadIdx.x;
//     if (col < cols && row_start < rows) {
//         if constexpr (std::is_same_v<Tin, float> && std::is_same_v<Tout, __half>) {
//             const float4* src_v4 = reinterpret_cast<const float4*>(&src[col * ld_src + row_start]);
//             __half2* dst_v2 = reinterpret_cast<__half2*>(&dst[col * ld_dst + row_start]);
//             float4 f4 = *src_v4;
//             f4.x *= scale; f4.y *= scale; f4.z *= scale; f4.w *= scale;
//             dst_v2[0] = __floats2half2_rn(f4.x, f4.y);
//             dst_v2[1] = __floats2half2_rn(f4.z, f4.w);
//         } else {
//             // 其他类型回退到逐元素路径
//             #pragma unroll
//             for (int i = 0; i < VECLEN; ++i) {
//                 if (row_start + i < rows) {
//                     float v = static_cast<float>(src[col * ld_src + row_start + i]) * scale;
//                     if constexpr (std::is_same_v<Tout, __half>) {
//                         dst[col * ld_dst + row_start + i] = __float2half(v);
//                     } else if constexpr (std::is_same_v<Tout, __nv_bfloat16>) {
//                         dst[col * ld_dst + row_start + i] = __float2bfloat16(v);
//                     } else {
//                         dst[col * ld_dst + row_start + i] = static_cast<Tout>(v);
//                     }
//                 }
//             }
//         }
//     }
// }
template<typename Tin, typename Tout>
__global__ void convert_kernel_strided_vec4_scaled(const Tin* src, size_t ld_src,
                                                  Tout* dst, size_t ld_dst,
                                                  size_t rows, size_t cols,
                                                  float scale)
{
    const int VECLEN = 4;
    size_t row_start = (blockIdx.y * blockDim.y + threadIdx.y) * VECLEN;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols && row_start < rows) {
        // --- 优化的 float -> half 路径 ---
        if constexpr (std::is_same_v<Tin, float> && std::is_same_v<Tout, __half>) {
            
            // 检查 ld_src 是否是4的倍数，这是对齐的必要条件
            if ((ld_src & 3) == 0) {
                // *** 快速路径 (Fast Path) ***
                // 当 ld_src 对齐时，使用最高效的矢量化加载
                const float4* src_v4 = reinterpret_cast<const float4*>(&src[col * ld_src + row_start]);
                __half2* dst_v2 = reinterpret_cast<__half2*>(&dst[col * ld_dst + row_start]);
                float4 f4 = *src_v4;
                f4.x *= scale; f4.y *= scale; f4.z *= scale; f4.w *= scale;
                dst_v2[0] = __floats2half2_rn(f4.x, f4.y);
                dst_v2[1] = __floats2half2_rn(f4.z, f4.w);
            } else {
                // *** 安全路径 (Safe Path) ***
                // 当 ld_src 未对齐时，必须逐元素读取
                const Tin* src_col_ptr = src + col * ld_src;
                Tout*      dst_col_ptr = dst + col * ld_dst;
                
                // 1. 安全地、逐个元素地读取
                float val0 = src_col_ptr[row_start + 0];
                float val1 = src_col_ptr[row_start + 1];
                float val2 = src_col_ptr[row_start + 2];
                float val3 = src_col_ptr[row_start + 3];
                
                // 2. 缩放
                val0 *= scale; val1 *= scale; val2 *= scale; val3 *= scale;

                // 3. 转换并矢量化写入
                __half2 p1 = __floats2half2_rn(val0, val1);
                __half2 p2 = __floats2half2_rn(val2, val3);
                *(reinterpret_cast<__half2*>(&dst_col_ptr[row_start + 0])) = p1;
                *(reinterpret_cast<__half2*>(&dst_col_ptr[row_start + 2])) = p2;
            }
        } else {
            // --- 其他类型的回退路径 ---
            // 你的原始回退路径已经是安全的逐元素操作，可以直接保留
            #pragma unroll
            for (int i = 0; i < VECLEN; ++i) {
                if (row_start + i < rows) {
                    float v = static_cast<float>(src[col * ld_src + row_start + i]) * scale;
                    if constexpr (std::is_same_v<Tout, __half>) {
                        dst[col * ld_dst + row_start + i] = __float2half(v);
                    } else if constexpr (std::is_same_v<Tout, __nv_bfloat16>) {
                        dst[col * ld_dst + row_start + i] = __float2bfloat16(v);
                    } else {
                        dst[col * ld_dst + row_start + i] = static_cast<Tout>(v);
                    }
                }
            }
        }
    }
}
template<typename Tin, typename Tout>
__global__ void convert_kernel_strided_vec4(const Tin* src, size_t ld_src, 
                                            Tout* dst, size_t ld_dst, 
                                            size_t rows, size_t cols) 
{
    const int VECLEN = 4;
    size_t row_start = (blockIdx.y * blockDim.y + threadIdx.y) * VECLEN;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    if (col < cols && row_start < rows) {
        // --- float -> half 分支 ---
        if constexpr (std::is_same_v<Tin, float> && std::is_same_v<Tout, __half>) {
            
            // 检查 ld_src 是否是偶数 (更严格可以检查是否是4的倍数)
            // 地址对齐要求 (float4) 是16字节，所以地址必须是16的倍数。
            // 一个简单的代理检查是 ld_src 是4的倍数。
            if ((ld_src & 3) == 0) { 
                // *** 快速路径 (Fast Path) ***
                const float4* src_v4 = reinterpret_cast<const float4*>(&src[col * ld_src + row_start]);
                __half2* dst_v2 = reinterpret_cast<__half2*>(&dst[col * ld_dst + row_start]);
                float4 temp_f4 = *src_v4;
                dst_v2[0] = __floats2half2_rn(temp_f4.x, temp_f4.y);
                dst_v2[1] = __floats2half2_rn(temp_f4.z, temp_f4.w);
            } else {
                // *** 安全路径 (Safe Path) ***
                const Tin* src_col_ptr = src + col * ld_src;
                Tout*      dst_col_ptr = dst + col * ld_dst;
                float val0 = src_col_ptr[row_start + 0];
                float val1 = src_col_ptr[row_start + 1];
                float val2 = src_col_ptr[row_start + 2];
                float val3 = src_col_ptr[row_start + 3];
                __half2 p1 = __floats2half2_rn(val0, val1);
                __half2 p2 = __floats2half2_rn(val2, val3);
                *(reinterpret_cast<__half2*>(&dst_col_ptr[row_start + 0])) = p1;
                *(reinterpret_cast<__half2*>(&dst_col_ptr[row_start + 2])) = p2;
            }

        } 
        // --- half -> float 分支 ---
        else if constexpr (std::is_same_v<Tin, __half> && std::is_same_v<Tout, float>) {
            // __half2 的对齐要求是4字节，ld_src 是偶数即可
            if ((ld_src & 1) == 0) {
                // *** 快速路径 ***
                const __half2* src_v2 = reinterpret_cast<const __half2*>(&src[col * ld_src + row_start]);
                float4* dst_v4 = reinterpret_cast<float4*>(&dst[col * ld_dst + row_start]);
                __half2 temp_h2_0 = src_v2[0];
                __half2 temp_h2_1 = src_v2[1];
                float2 temp_f2_0 = __half22float2(temp_h2_0);
                float2 temp_f2_1 = __half22float2(temp_h2_1);
                *dst_v4 = make_float4(temp_f2_0.x, temp_f2_0.y, temp_f2_1.x, temp_f2_1.y);
            } else {
                // *** 安全路径 ***
                const Tin* src_col_ptr = src + col * ld_src;
                Tout*      dst_col_ptr = dst + col * ld_dst;
                // 安全地、逐个元素地读取
                __half h0 = src_col_ptr[row_start + 0];
                __half h1 = src_col_ptr[row_start + 1];
                __half h2 = src_col_ptr[row_start + 2];
                __half h3 = src_col_ptr[row_start + 3];
                // 转换
                float2 f_p1 = __half22float2(make_half2(h0, h1));
                float2 f_p2 = __half22float2(make_half2(h2, h3));
                // 写入
                *(reinterpret_cast<float2*>(&dst_col_ptr[row_start + 0])) = f_p1;
                *(reinterpret_cast<float2*>(&dst_col_ptr[row_start + 2])) = f_p2;
            }
        } 
        // --- 其他分支，比如 float -> float ---
        else {
            // 这里也可以添加类似的 if-else 逻辑
            const Tin* src_col_ptr = src + col * ld_src;
            Tout*      dst_col_ptr = dst + col * ld_dst;
            #pragma unroll
            for (int i = 0; i < VECLEN; ++i) {
                dst_col_ptr[row_start + i] = static_cast<Tout>(src_col_ptr[row_start + i]);
            }
        }
    }
}

// 将两次 GEMM（Temp1 = W^T*A, A = A - Y*Temp1）合并到一个可选的 cuda graph 捕获中
template<typename T>
inline void gemm_pair_graph_or_plain(
    cublasHandle_t h,
    cudaStream_t stream,
    // sizes
    int m_h, int n_t, int m_r,
    // Temp1 = W^T * A_trailing
    const T* d_W_ptr, int ldw,
    const T* d_A_trailing_ptr, int lda,
    T* d_Temp1_ptr, int ldtemp,
    // A_trailing = A_trailing - Y * Temp1
    const T* d_Y_ptr, int ldy,
    T* d_A_trailing_out_ptr, int lda2)
{
    bool use_cg = false;
    if (const char* e = std::getenv("HQB_USE_CG")) { use_cg = (e[0] != '0'); }
    if (!use_cg) {
        const T one = T(1), zero = T(0), minus_one = T(-1);
        BlasCaller<T>::gemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                            m_r, n_t, m_h,
                            &one, d_W_ptr, ldw, d_A_trailing_ptr, lda,
                            &zero, d_Temp1_ptr, ldtemp);
        BlasCaller<T>::gemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                            m_h, n_t, m_r,
                            &minus_one, d_Y_ptr, ldy, d_Temp1_ptr, ldtemp,
                            &one, d_A_trailing_out_ptr, lda2);
        return;
    }
    cudaGraph_t graph = nullptr; cudaGraphExec_t gexec = nullptr;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed);
    {
        const T one = T(1), zero = T(0), minus_one = T(-1);
        BlasCaller<T>::gemm(h, CUBLAS_OP_T, CUBLAS_OP_N,
                            m_r, n_t, m_h,
                            &one, d_W_ptr, ldw, d_A_trailing_ptr, lda,
                            &zero, d_Temp1_ptr, ldtemp);
        BlasCaller<T>::gemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                            m_h, n_t, m_r,
                            &minus_one, d_Y_ptr, ldy, d_Temp1_ptr, ldtemp,
                            &one, d_A_trailing_out_ptr, lda2);
    }
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(gexec, stream);
    cudaGraphExecDestroy(gexec);
    cudaGraphDestroy(graph);
}

// 计算 B = I - Q（列主序，支持不同的 lda/ldb）
template<typename T>
__global__ void i_minus_matrix_kernel(T* __restrict__ B, const T* __restrict__ Q,
                                      int rows, int cols, int ldb, int ldq)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < cols && row < rows) {
        T q = Q[col * ldq + row];
        T val = -q;
        if (row == col) val = T(1) - q;
        B[col * ldb + row] = val;
    }
}

// 向量化版本（float4 / double2），按列主序处理，每线程一次性处理多个行元素
// 仅当 rows 按 4(对 float) 或 2(对 double) 对齐时启用
#ifndef HQB_DISABLE_VEC_I_MINUS
__global__ void i_minus_matrix_kernel_f4(float* __restrict__ B, const float* __restrict__ Q,
                                         int rows, int cols, int ldb, int ldq)
{
    const int VECLEN = 4;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_row = blockIdx.y * blockDim.y + threadIdx.y; // 向量块索引
    int base_row = vec_row * VECLEN;
    if (col >= cols || base_row >= rows) return;

    // 列起始地址（列主序）
    const float4* __restrict__ qv = reinterpret_cast<const float4*>(Q + col * ldq + base_row);
    float4* __restrict__ bv = reinterpret_cast<float4*>(B + col * ldb + base_row);

    float4 q = *qv;
    // 处理对角线：若当前向量块包含对角元素（行==列）
    // 行索引分别为 base_row + {0,1,2,3}
    float r0 = (base_row + 0 == col) ? (1.f - q.x) : (-q.x);
    float r1 = (base_row + 1 == col) ? (1.f - q.y) : (-q.y);
    float r2 = (base_row + 2 == col) ? (1.f - q.z) : (-q.z);
    float r3 = (base_row + 3 == col) ? (1.f - q.w) : (-q.w);
    *bv = make_float4(r0, r1, r2, r3);
}

__global__ void i_minus_matrix_kernel_d2(double* __restrict__ B, const double* __restrict__ Q,
                                         int rows, int cols, int ldb, int ldq)
{
    const int VECLEN = 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int vec_row = blockIdx.y * blockDim.y + threadIdx.y; // 向量块索引
    int base_row = vec_row * VECLEN;
    if (col >= cols || base_row >= rows) return;

    const double2* __restrict__ qv = reinterpret_cast<const double2*>(Q + col * ldq + base_row);
    double2* __restrict__ bv = reinterpret_cast<double2*>(B + col * ldb + base_row);

    double2 q = *qv;
    double r0 = (base_row + 0 == col) ? (1.0 - q.x) : (-q.x);
    double r1 = (base_row + 1 == col) ? (1.0 - q.y) : (-q.y);
    *bv = make_double2(r0, r1);
}
#endif // HQB_DISABLE_VEC_I_MINUS

// ====== Common utility kernels (moved from AP_HQB-SVD-TSQR.cu) ======
// 下三角清零（不含对角）
template<typename T>
__global__ void zero_lower_triangular_kernel(T* matrix, int rows, int cols, int lda) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < rows && col < cols && row > col) {
        matrix[col * lda + row] = T(0);
    }
}

// 清零指定行（列主序）
template<typename T>
__global__ void zero_row_kernel(T* A, int m, int n, int row_idx) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        A[col * m + row_idx] = T(0);
    }
}

template <typename T>
__global__ void compute_sub_row_norms_kernel(
    T* d_norms,
    const T* d_R,
    int n,
    int ld)
{

    extern __shared__ unsigned char s_raw_shmem[];

    T* s_shmem = reinterpret_cast<T*>(s_raw_shmem);
    
    
    int row_idx = blockIdx.x;
    if (row_idx >= n) return;

    T local_sum_sq = 0.0;
    for (int j = row_idx + threadIdx.x; j < n; j += blockDim.x) {
        size_t index = (size_t)row_idx + (size_t)j * ld;
        T val = d_R[index];
        local_sum_sq += val * val;
    }

    s_shmem[threadIdx.x] = local_sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_shmem[threadIdx.x] += s_shmem[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_norms[row_idx] = sqrt(s_shmem[0]);
    }
}

// Double-specialized helper kernels (definitions in my_cuda_kernels_extras.cu)
__global__ void computeSingularValues_double(const double* __restrict__ eigenvals,
                                             double* __restrict__ singularvals,
                                             int n);
__global__ void divide_columns_by_diag_double(double* A, const double* S,
                                              int m, int n, int ld, double eps);
__global__ void scale_columns_by_vector_inplace_double(double* A, const double* s,
                                                       int m, int n, int ld);
__global__ void column_dot_double(const double* __restrict__ A,
                                  const double* __restrict__ B,
                                  double* __restrict__ dots,
                                  int m, int n, int ld);
__global__ void build_sign_vector_double(const double* __restrict__ dots,
                                         double* __restrict__ sgn,
                                         int n);

#endif
