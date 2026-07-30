
#include "my_cuda_kernels.h"
#include <math.h>


__global__ void computeSingularValues_double(const double* __restrict__ eigenvals,
                                             double* __restrict__ singularvals,
                                             int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) singularvals[idx] = sqrt(fabs(eigenvals[idx]));
}

__global__ void divide_columns_by_diag_double(double* A, const double* S,
                                              int m, int n, int ld, double eps)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        double s = S[col];
        double v = (fabs(s) > eps) ? (A[col * ld + row] / s) : 0.0;
        A[col * ld + row] = v;
    }
}

__global__ void scale_columns_by_vector_inplace_double(double* A, const double* s,
                                                       int m, int n, int ld)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < n && row < m) {
        A[col * ld + row] *= s[col];
    }
}

__global__ void column_dot_double(const double* __restrict__ A,
                                  const double* __restrict__ B,
                                  double* __restrict__ dots,
                                  int m, int n, int ld)
{
    int col = blockIdx.x;
    if (col >= n) return;
    double sum = 0.0;
    for (int i = threadIdx.x; i < m; i += blockDim.x) {
        sum += A[col * ld + i] * B[col * ld + i];
    }
    __shared__ double sh[256];
    sh[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) dots[col] = sh[0];
}

__global__ void build_sign_vector_double(const double* __restrict__ dots,
                                         double* __restrict__ sgn,
                                         int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) sgn[i] = (dots[i] >= 0.0 ? 1.0 : -1.0);
}


