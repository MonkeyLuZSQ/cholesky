#pragma once

#include "Common.cuh"

template <int DIM_X, int DIM_Y, typename T_Index, typename Ti, typename Tex,
          typename To>
__device__ void rocblas_gemvn_kernel_calc(int m, int n, Tex alpha, const Ti *A,
                                          T_Index lda, const Ti *x,
                                          T_Index incx, Tex beta, To *y,
                                          T_Index incy) {
  int thread_id = threadIdx.x + threadIdx.y * DIM_X;

  if (!alpha) {
    if (thread_id < DIM_X * 4) {
      int64_t ind = blockIdx.x * DIM_X * 4 + thread_id;
      if (ind < m)
        y[ind * T_Index(incy)] =
            beta ? (To)(beta * y[ind * T_Index(incy)]) : (To)0;
    }
    return;
  }

  // threads are all configurated locally
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int ind;

  __shared__ Tex sdata[DIM_X * 4 * DIM_Y];

  Tex res_A[4];
  Tex res_x[4];

  res_A[0] = res_A[1] = res_A[2] = res_A[3] = Tex{0};

  ind = blockIdx.x * DIM_X * 4 + tx;

  int n_tail = n % (4 * DIM_Y);
  int col;

  for (col = ty * 4; col < (n - n_tail); col += 4 * DIM_Y) {
    res_x[0] = x[(col + 0) * T_Index(incx)];
    res_x[1] = x[(col + 1) * T_Index(incx)];
    res_x[2] = x[(col + 2) * T_Index(incx)];
    res_x[3] = x[(col + 3) * T_Index(incx)];

    if (ind < m) {
      res_A[0] += A[ind + (col + 0) * T_Index(lda)] * res_x[0];
      res_A[0] += A[ind + (col + 1) * T_Index(lda)] * res_x[1];
      res_A[0] += A[ind + (col + 2) * T_Index(lda)] * res_x[2];
      res_A[0] += A[ind + (col + 3) * T_Index(lda)] * res_x[3];

      if (ind + DIM_X < m) {
        res_A[1] += A[ind + DIM_X + (col + 0) * T_Index(lda)] * res_x[0];
        res_A[1] += A[ind + DIM_X + (col + 1) * T_Index(lda)] * res_x[1];
        res_A[1] += A[ind + DIM_X + (col + 2) * T_Index(lda)] * res_x[2];
        res_A[1] += A[ind + DIM_X + (col + 3) * T_Index(lda)] * res_x[3];

        if (ind + 2 * DIM_X < m) {
          res_A[2] += A[ind + 2 * DIM_X + (col + 0) * T_Index(lda)] * res_x[0];
          res_A[2] += A[ind + 2 * DIM_X + (col + 1) * T_Index(lda)] * res_x[1];
          res_A[2] += A[ind + 2 * DIM_X + (col + 2) * T_Index(lda)] * res_x[2];
          res_A[2] += A[ind + 2 * DIM_X + (col + 3) * T_Index(lda)] * res_x[3];

          if (ind + 3 * DIM_X < m) {
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 0) * T_Index(lda)] * res_x[0];
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 1) * T_Index(lda)] * res_x[1];
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 2) * T_Index(lda)] * res_x[2];
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 3) * T_Index(lda)] * res_x[3];
          }
        }
      }
    }
  }

  // if n is not multiple of (DIM_Y * 4)
  if (n_tail > 0) {
    res_x[0] = res_x[1] = res_x[2] = res_x[3] = Tex{0};

    if (col + 0 < n) {
      res_x[0] = x[(col + 0) * T_Index(incx)];

      if (col + 1 < n) {
        res_x[1] = x[(col + 1) * T_Index(incx)];

        if (col + 2 < n) {
          res_x[2] = x[(col + 2) * T_Index(incx)];

          if (col + 3 < n)
            res_x[3] = x[(col + 3) * T_Index(incx)];
        }
      }
    }

    if (ind < m) {
      res_A[0] += A[ind + (col + 0) * T_Index(lda) * (col + 0 < n)] * res_x[0];
      res_A[0] += A[ind + (col + 1) * T_Index(lda) * (col + 1 < n)] * res_x[1];
      res_A[0] += A[ind + (col + 2) * T_Index(lda) * (col + 2 < n)] * res_x[2];
      res_A[0] += A[ind + (col + 3) * T_Index(lda) * (col + 3 < n)] * res_x[3];

      if (ind + DIM_X < m) {
        res_A[1] += A[ind + DIM_X + (col + 0) * T_Index(lda) * (col + 0 < n)] *
                    res_x[0];
        res_A[1] += A[ind + DIM_X + (col + 1) * T_Index(lda) * (col + 1 < n)] *
                    res_x[1];
        res_A[1] += A[ind + DIM_X + (col + 2) * T_Index(lda) * (col + 2 < n)] *
                    res_x[2];
        res_A[1] += A[ind + DIM_X + (col + 3) * T_Index(lda) * (col + 3 < n)] *
                    res_x[3];

        if (ind + 2 * DIM_X < m) {
          res_A[2] +=
              A[ind + 2 * DIM_X + (col + 0) * T_Index(lda) * (col + 0 < n)] *
              res_x[0];
          res_A[2] +=
              A[ind + 2 * DIM_X + (col + 1) * T_Index(lda) * (col + 1 < n)] *
              res_x[1];
          res_A[2] +=
              A[ind + 2 * DIM_X + (col + 2) * T_Index(lda) * (col + 2 < n)] *
              res_x[2];
          res_A[2] +=
              A[ind + 2 * DIM_X + (col + 3) * T_Index(lda) * (col + 3 < n)] *
              res_x[3];

          if (ind + 3 * DIM_X < m) {
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 0) * T_Index(lda) * (col + 0 < n)] *
                res_x[0];
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 1) * T_Index(lda) * (col + 1 < n)] *
                res_x[1];
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 2) * T_Index(lda) * (col + 2 < n)] *
                res_x[2];
            res_A[3] +=
                A[ind + 3 * DIM_X + (col + 3) * T_Index(lda) * (col + 3 < n)] *
                res_x[3];
          }
        }
      }
    }
  }

  sdata[tx + ty * DIM_X * 4] = res_A[0];
  sdata[tx + DIM_X + ty * DIM_X * 4] = res_A[1];
  sdata[tx + 2 * DIM_X + ty * DIM_X * 4] = res_A[2];
  sdata[tx + 3 * DIM_X + ty * DIM_X * 4] = res_A[3];

  __syncthreads();

  if (thread_id < DIM_X * 4) {
    for (int i = 1; i < DIM_Y; i++)
      sdata[thread_id] += sdata[thread_id + DIM_X * 4 * i];

    ind = blockIdx.x * DIM_X * 4 + thread_id;

    if (ind < m)
      y[ind * T_Index(incy)] =
          beta ? (To)(alpha * sdata[thread_id] + beta * y[ind * T_Index(incy)])
               : (To)(alpha * sdata[thread_id]);
  }
}

template <int DIM_X, int DIM_Y, typename T_Index, typename Ti, typename Tex,
          typename To>
ROCBLAS_KERNEL(DIM_X *DIM_Y)
rocblas_gemvn_kernel(int m, int n, Tex alpha_device_host, int64_t stride_alpha,
                     const Ti *Aa, int64_t shifta, T_Index lda, int64_t strideA,
                     const Ti *xa, int64_t shiftx, T_Index incx,
                     int64_t stridex, Tex beta_device_host, int64_t stride_beta,
                     To *ya, int64_t shifty, T_Index incy, int64_t stridey) {
  int num_threads = blockDim.x * blockDim.y * blockDim.z;
  if (DIM_X * DIM_Y != num_threads)
    return; // need to launch exactly the same number of threads as template
            // parameters indicate

  auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);
  auto beta = load_scalar(beta_device_host, blockIdx.y, stride_beta);

  if (!alpha && beta == half(1))
    return;

  const auto *A = cond_load_ptr_batch(alpha, Aa, blockIdx.y, shifta, strideA);
  const auto *x = cond_load_ptr_batch(alpha, xa, blockIdx.y, shiftx, stridex);

  auto *y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

  rocblas_gemvn_kernel_calc<DIM_X, DIM_Y, T_Index>(m, n, alpha, A, lda, x, incx,
                                                   beta, y, incy);
}

template <typename Ti, typename Tex, typename To>
cublasStatus_t rocblas_internal_gemv_launcher(
    cublasHandle_t handle, int m, int n, const Tex *alpha, int64_t stride_alpha,
    const Ti *A, int64_t offseta, int64_t lda, int64_t strideA, const Ti *x,
    int64_t offsetx, int64_t incx, int64_t stridex, const Tex *beta,
    int64_t stride_beta, To *y, int64_t offsety, int64_t incy, int64_t stridey,
    int batch_count, Tex *workspace) {
  // quick return
  if (!m || !n || !batch_count)
    return CUBLAS_STATUS_SUCCESS;

  cudaStream_t rocblas_stream;
  cublasGetStream_v2(handle, &rocblas_stream);

  // in case of negative inc shift pointer to end of data for negative indexing
  // tid*inc
  auto shiftx = incx < 0 ? offsetx - int64_t(incx) * (n - 1) : offsetx;
  auto shifty = incy < 0 ? offsety - int64_t(incy) * (m - 1) : offsety;

  bool i64_incs = lda > c_i32_max || incx > c_i32_max || incx < c_i32_min ||
                  incy > c_i32_max || incy < c_i32_min;

  bool i64_indices // i64_incs implies i64_indices
      = i64_incs || size_t(n) * lda > c_i32_max ||
        size_t(n - 1) * std::abs(incx) >= c_i32_max ||
        size_t(m - 1) * std::abs(incy) >= c_i32_max;

  {
#define gemvn_KARGS(alpha_, beta_)                                             \
  <<<gemvn_grid, gemvn_threads, 0, rocblas_stream>>>(                          \
      m, n, alpha_, stride_alpha, A, offseta, lda, strideA, x, shiftx, incx,   \
      stridex, beta_, stride_beta, y, shifty, incy, stridey)

    {
      // GEMVN_DIM_Y must be at least 4, 8 * 8 is very slow only 40Gflop/s
      static constexpr int GEMVN_DIM_X = 64;
      static constexpr int GEMVN_DIM_Y = 16;
      int blocks = (m - 1) / (GEMVN_DIM_X * 4) + 1;
      dim3 gemvn_grid(blocks, batch_count);
      dim3 gemvn_threads(GEMVN_DIM_X, GEMVN_DIM_Y);
      cublasPointerMode_t pointer_mode;
      cublasGetPointerMode_v2(handle, &pointer_mode);
      if (pointer_mode == CUBLAS_POINTER_MODE_HOST) {
        if (!i64_indices)
          rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int> gemvn_KARGS(alpha,
                                                                          beta);
        else
          rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t> gemvn_KARGS(
              alpha, beta);
      } else {
        if (!*alpha && int(*beta) == 1)
          return CUBLAS_STATUS_SUCCESS;

        if (!i64_indices)
          rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int> gemvn_KARGS(
              *alpha, *beta);
        else
          rocblas_gemvn_kernel<GEMVN_DIM_X, GEMVN_DIM_Y, int64_t> gemvn_KARGS(
              *alpha, *beta);
      }
    }
#undef gemvn_KARGS
  }
  return CUBLAS_STATUS_SUCCESS;
}

template <typename T>
cublasStatus_t rocblas_internal_gemv_template(
    cublasHandle_t handle, int m, int n, const T *alpha, int64_t stride_alpha,
    const T *A, int64_t offseta, int lda, int64_t strideA, const T *x,
    int64_t offsetx, int incx, int64_t stridex, const T *beta,
    int64_t stride_beta, T *y, int64_t offsety, int incy, int64_t stridey,
    int batch_count, T *workspace) {
  return rocblas_internal_gemv_launcher<T, T, T>(
      handle, m, n, alpha, stride_alpha, A, offseta, (int64_t)lda, strideA, x,
      offsetx, (int64_t)incx, stridex, beta, stride_beta, y, offsety,
      (int64_t)incy, stridey, batch_count, workspace);
}

template <typename T>
cublasStatus_t rocblas_internal_gemv_batched_template(
    cublasHandle_t handle, int m, int n, const T *alpha, int64_t stride_alpha,
    const T *const *A, int64_t offseta, int lda, int64_t strideA,
    const T *const *x, int64_t offsetx, int incx, int64_t stridex,
    const T *beta, int64_t stride_beta, T *const *y, int64_t offsety, int incy,
    int64_t stridey, int batch_count, T *workspace = nullptr) {
  return rocblas_internal_gemv_launcher(
      handle, m, n, alpha, stride_alpha, A, offseta, (int64_t)lda, strideA, x,
      offsetx, (int64_t)incx, stridex, beta, stride_beta, y, offsety,
      (int64_t)incy, stridey, batch_count, workspace);
}

// gemv - batched
template <typename T>
cublasStatus_t
rocblasCall_gemv(cublasHandle_t handle, int m, int n, const T *alpha,
                 int64_t stride_alpha, const T *const *A, int64_t offseta,
                 int lda, int64_t strideA, const T *const *x, int64_t offsetx,
                 int incx, int64_t stridex, const T *beta, int64_t stride_beta,
                 T *const *y, int64_t offsety, int incy, int64_t stridey,
                 int batch_count, T **work) {
  return rocblas_internal_gemv_batched_template<T>(
      handle, m, n, alpha, stride_alpha, A, offseta, lda, strideA, x, offsetx,
      incx, stridex, beta, stride_beta, y, offsety, incy, stridey, batch_count);
}

template cublasStatus_t rocblasCall_gemv<half>(
    cublasHandle_t handle, int m, int n, const half *alpha,
    int64_t stride_alpha, const half *const *A, int64_t offseta, int lda,
    int64_t strideA, const half *const *x, int64_t offsetx, int incx,
    int64_t stridex, const half *beta, int64_t stride_beta, half *const *y,
    int64_t offsety, int incy, int64_t stridey, int batch_count, half **work);
