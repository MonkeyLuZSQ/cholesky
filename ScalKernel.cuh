#pragma once

#include "Common.cuh"

#define ROCBLAS_SCAL_NB 256

template <typename API_INT, int NB, typename T, typename Tex, typename Ta,
          typename Tx>
ROCBLAS_KERNEL(NB)
rocblas_scal_kernel(int n, Ta alpha_device_host, int64_t stride_alpha, Tx xa,
                    int64_t offset_x, API_INT incx, int64_t stride_x) {
  auto *x = load_ptr_batch(xa, blockIdx.y, offset_x, stride_x);
  auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

  if (alpha == T(1))
    return;

  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  // bound
  if (tid < n) {
    Tex res = (Tex)x[tid * int64_t(incx)] * alpha;
    x[tid * int64_t(incx)] = (T)res;
  }
}

// //!
// //! @brief Optimized kernel for the SCAL when the compute and alpha type is
// half precision.
// //! @remark Increments are required to be equal to one, that's why they are
// unspecified.
// //!
// template <int NB, typename Ta, typename Tx>
// ROCBLAS_KERNEL(NB)
// rocblas_hscal_mlt_4_kernel(int    n,
//                            int    n_mod_4,
//                            int    n_mlt_4,
//                            Ta             alpha_device_host,
//                            int64_t stride_alpha,
//                            Tx __restrict__ xa,
//                            int64_t offset_x,
//                            int64_t stride_x)
// {
//     auto alpha = load_scalar(alpha_device_host, blockIdx.y, stride_alpha);

//     if(alpha == 1)
//         return;

//     half2 x0, x1;
//     half2 z0, z1;

//     uint32_t tid = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

//     if(tid + 3 < n)
//     {
//         half4* x = (half4*)load_ptr_batch(xa, blockIdx.y, offset_x + tid,
//         stride_x);

//         x0[0] = (*x)[0];
//         x0[1] = (*x)[1];
//         x1[0] = (*x)[2];
//         x1[1] = (*x)[3];

//         z0[0] = alpha * x0[0];
//         z0[1] = alpha * x0[1];
//         z1[0] = alpha * x1[0];
//         z1[1] = alpha * x1[1];

//         (*x)[0] = z0[0];
//         (*x)[1] = z0[1];
//         (*x)[2] = z1[0];
//         (*x)[3] = z1[1];
//     }

//     // If `n_mod_4` is true then the computation of last few element in the
//     vector `x` is covered below. if(n_mod_4)
//     {
//         //The last ThreadID which is a multiple of 4 should complete the
//         computation of last few elements of vector `x` if(tid == n_mlt_4)
//         {
//             auto* x = load_ptr_batch(xa, blockIdx.y, offset_x, stride_x);
//             for(int32_t j = 0; j < n_mod_4; ++j)
//             {
//                 x[tid + j] = x[tid + j] * alpha;
//             }
//         }
//     }
// }

template <typename API_INT, int NB, typename T, typename Tex, typename Ta,
          typename Tx>
cublasStatus_t
rocblas_internal_scal_launcher(cublasHandle_t handle, API_INT n,
                               const Ta *alpha, int64_t stride_alpha, Tx x,
                               int64_t offset_x, API_INT incx, int64_t stride_x,
                               API_INT batch_count) {
  // Quick return if possible. Not Argument error
  if (n <= 0 || incx <= 0 || batch_count <= 0) {
    return CUBLAS_STATUS_SUCCESS;
  }
  cublasPointerMode_t pointer_mode;
  cublasGetPointerMode(handle, &pointer_mode);
  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  // static constexpr bool using_rocblas_float
  //     = std::is_same_v<Tx, rocblas_float*> || std::is_same_v<Tx,
  //     rocblas_float* const*>;

  // // Using rocblas_half ?
  // static constexpr bool using_rocblas_half
  //     = std::is_same_v<Ta, half> && std::is_same_v<Tex, half>;
#if 0
    if(using_rocblas_half && incx == 1)
    {
        // Kernel function for improving the performance of HSCAL when incx==1
        int32_t n_mod_4 = n & 3; // n mod 4
        int32_t n_mlt_4 = n & ~(rocblas_int)3; // multiple of 4
        int32_t blocks  = 1 + ((n - 1) / (NB * 4));
        dim3    grid(blocks, batch_count);
        dim3    threads(NB);

        if constexpr(using_rocblas_half)
        {
            if(rocblas_pointer_mode_device == handle->pointer_mode)
                ROCBLAS_LAUNCH_KERNEL((rocblas_hscal_mlt_4_kernel<NB>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      n_mod_4,
                                      n_mlt_4,
                                      (const rocblas_half*)alpha,
                                      stride_alpha,
                                      x,
                                      offset_x,
                                      stride_x);
            else // single alpha is on host
                ROCBLAS_LAUNCH_KERNEL((rocblas_hscal_mlt_4_kernel<NB>),
                                      grid,
                                      threads,
                                      0,
                                      handle->get_stream(),
                                      n,
                                      n_mod_4,
                                      n_mlt_4,
                                      load_scalar((const rocblas_half*)alpha),
                                      stride_alpha,
                                      x,
                                      offset_x,
                                      stride_x);
        }
    }
    else
#endif
  {
    int blocks = (n - 1) / NB + 1;
    dim3 grid(blocks, batch_count);
    dim3 threads(NB);

    if (CUBLAS_POINTER_MODE_DEVICE == pointer_mode)
      rocblas_scal_kernel<API_INT, NB, T, Tex><<<grid, threads, 0, stream>>>(
          n, alpha, stride_alpha, x, offset_x, incx, stride_x);
    // ROCBLAS_LAUNCH_KERNEL((rocblas_scal_kernel<API_INT, NB, T, Tex>),
    //                       grid,
    //                       threads,
    //                       0,
    //                       handle->get_stream(),
    //                       n,
    //                       alpha,
    //                       stride_alpha,
    //                       x,
    //                       offset_x,
    //                       incx,
    //                       stride_x);
    else // single alpha is on host
      rocblas_scal_kernel<API_INT, NB, T, Tex><<<grid, threads, 0, stream>>>(
          n, *alpha, stride_alpha, x, offset_x, incx, stride_x);
    // ROCBLAS_LAUNCH_KERNEL((rocblas_scal_kernel<API_INT, NB, T, Tex>),
    //                       grid,
    //                       threads,
    //                       0,
    //                       handle->get_stream(),
    //                       n,
    //                       *alpha,
    //                       stride_alpha,
    //                       x,
    //                       offset_x,
    //                       incx,
    //                       stride_x);
  }
  return CUBLAS_STATUS_SUCCESS;
}

template <typename T, typename Ta>
cublasStatus_t
rocblas_internal_scal_batched_template(cublasHandle_t handle, int n,
                                       const Ta *alpha, int64_t stride_alpha,
                                       T *const *x, int64_t offset_x, int incx,
                                       int64_t stride_x, int batch_count) {
  return rocblas_internal_scal_launcher<int, ROCBLAS_SCAL_NB, T, T>(
      handle, n, alpha, stride_alpha, x, offset_x, incx, stride_x, batch_count);
}

// batched scal
template <typename T, typename I, typename S>
cublasStatus_t rocblasCall_scal(cublasHandle_t handle, I n, const S *alpha,
                                int64_t stridea, T *const *x, int64_t offsetx,
                                I incx, int64_t stridex, I batch_count) {
  return rocblas_internal_scal_batched_template(
      handle, n, alpha, stridea, x, offsetx, incx, stridex, batch_count);
}

template cublasStatus_t
rocblasCall_scal<half, int, half>(cublasHandle_t handle, int n,
                                  const half *alpha, int64_t stridea,
                                  half *const *x, int64_t offsetx, int incx,
                                  int64_t stridex, int batch_count);
