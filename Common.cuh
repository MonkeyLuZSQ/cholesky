#pragma once

#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <thrust/complex.h>
#include <type_traits>

#define ROCBLAS_KERNEL(lb_) static __global__ __launch_bounds__((lb_)) void
#define ROCBLAS_KERNEL_NO_BOUNDS static __global__ void

#define ROCBLAS_DOT_NB 512
#define warpSize 32

constexpr int64_t c_i32_max = int64_t(std::numeric_limits<int32_t>::max());
constexpr int64_t c_i32_min = int64_t(std::numeric_limits<int32_t>::min());

template <typename T, typename I>
__global__ void get_array(T **out, T *in, int64_t stride, I batch) {
  I b = blockIdx.x * static_cast<I>(blockDim.x) + threadIdx.x;

  if (b < batch)
    out[b] = in + b * stride;
}

template <typename T> T const *cast2constType(T *array) { return array; }

template <typename T> T const *const *cast2constType(T *const *array) {
  return array;
}

/*
 * ===========================================================================
 *    common location for device functions and kernels that are used across
 *    several rocSOLVER routines, excepting those device functions and kernels
 *    that reproduce LAPACK functionality (see lapack_device_functions.hpp).
 * ===========================================================================
 */

#define BS1                                                                    \
  256 // generic 1 dimensional thread-block size used to call common kernels
#define BS2                                                                    \
  32 // generic 2 dimensional thread-block size used to call common kernels

/**
 * indexing for packed storage
 * for upper triangular
 *
 * ---------------------------
 * 0 1 3
 *   2 4
 *     5
 * ---------------------------
 *
 **/

template <typename I> __device__ static I idx_upper(I i, I j, I n) {
  assert((0 <= i) && (i <= (n - 1)));
  assert((0 <= j) && (j <= (n - 1)));
  assert(i <= j);

  return (i + (j * (j + 1)) / 2);
}

/**
 * indexing for packed storage
 * for lower triangular
 *
 * ---------------------------
 * 0
 * 1      n
 * *      (n+1)
 * *
 * (n-1)  ...        n*(n+1)/2
 * ---------------------------
 **/
template <typename I> __device__ static I idx_lower(I i, I j, I n) {
  assert((0 <= i) && (i <= (n - 1)));
  assert((0 <= j) && (j <= (n - 1)));
  assert(i >= j);

  return ((i - j) + (j * (2 * n + 1 - j)) / 2);
}

template <typename T, typename I, typename U>
__global__ void reset_info(T *info, const I n, U val, I incr = 0) {
  I idx = blockIdx.x * static_cast<I>(blockDim.x) + threadIdx.x;

  if (idx < n)
    info[idx] = T(val) + incr * idx;
}

inline int64_t idx2D(const int64_t i, const int64_t j, const int64_t lda) {
  return j * lda + i;
}

inline int64_t idx2D(const int64_t i, const int64_t j, const int64_t inca,
                     const int64_t lda) {
  return j * lda + i * inca;
}

// Load a scalar. If the argument is a pointer, dereference it; otherwise copy
// it. Allows the same kernels to be used for host and device scalars.

// For host scalars
template <typename T> __forceinline__ __device__ __host__ T load_scalar(T x) {
  return x;
}

// For device scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(const T *xp) {
  return *xp;
}

// Load a batched scalar. This only works on the device. Used for batched
// functions which may pass an array of scalars rather than a single scalar.

// For device side array of scalars
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(const T *x, uint32_t idx,
                                                  int64_t inc) {
  return x[idx * inc];
}

// Overload for single scalar value
template <typename T>
__forceinline__ __device__ __host__ T load_scalar(T x, uint32_t idx,
                                                  int64_t inc) {
  return x;
}

// Load a pointer from a batch. If the argument is a T**, use block to index it
// and add the offset, if the argument is a T*, add block * stride to pointer
// and add offset.

// For device array of device pointers

// For device pointers
template <typename T, typename I>
__forceinline__ __device__ __host__ T *
load_ptr_batch(T *p, I block, int64_t offset, int64_t stride) {
  return p + block * stride + offset;
}

// For device array of device pointers
template <typename T, typename I>
__forceinline__ __device__ __host__ T *
load_ptr_batch(T *const *p, I block, int64_t offset, int64_t stride) {
  return p[block] + offset;
}

template <typename T, typename I>
__forceinline__ __device__ __host__ T *
load_ptr_batch(T **p, I block, int64_t offset, int64_t stride) {
  return p[block] + offset;
}

// guarded by condition
template <typename C, typename T>
__forceinline__ __device__ __host__ T *
cond_load_ptr_batch(C cond, T *p, uint32_t block, int64_t offset,
                    int64_t stride) {
  // safe to offset pointer regardless of condition as not dereferenced
  return load_ptr_batch(p, block, offset, stride);
}

// For device array of device pointers array is dereferenced, e.g. alpha, if
// !alpha don't dereference pointer array as we allow it to be null
template <typename C, typename T>
__forceinline__ __device__ __host__ T *
cond_load_ptr_batch(C cond, T *const *p, uint32_t block, int64_t offset,
                    int64_t stride) {
  return cond ? load_ptr_batch(p, block, offset, stride) : nullptr;
}

template <typename C, typename T>
__forceinline__ __device__ __host__ T *
cond_load_ptr_batch(C cond, T **p, uint32_t block, int64_t offset,
                    int64_t stride) {
  return cond ? load_ptr_batch(p, block, offset, stride) : nullptr;
}

// template <typename T>
// cublasStatus_t rocblasCall_dot(cublasHandle_t handle, int n, T *x,
//                                int64_t offsetx, int incx, int64_t stridex,
//                                T *const y[], int64_t offsety, int incy,
//                                int64_t stridey, int batch_count, T *results,
//                                T *workspace, T **work);

// template <typename T>
// cublasStatus_t
// rocblasCall_gemv(cublasHandle_t handle, int m, int n, const T *alpha,
//                  int64_t stride_alpha, const T *const *A, int64_t offseta,
//                  int lda, int64_t strideA, const T *const *x, int64_t offsetx,
//                  int incx, int64_t stridex, const T *beta, int64_t stride_beta,
//                  T *const *y, int64_t offsety, int incy, int64_t stridey,
//                  int batch_count, T **work);

// template <typename T, typename I, typename S>
// cublasStatus_t rocblasCall_scal(cublasHandle_t handle, I n, const S *alpha,
//                                 int64_t stridea, T *const *x, int64_t offsetx,
//                                 I incx, int64_t stridex, I batch_count);

// template <typename T, typename U>
// cublasStatus_t rocsolver_potf2_template(cublasHandle_t handle,
//                                         const cublasFillMode_t uplo,
//                                         const int n,
//                                         U A,
//                                         const int shiftA,
//                                         const int lda,
//                                         const int64_t strideA,
//                                         int* info,
//                                         const int batch_count,
//                                         T* scalars,
//                                         T* work,
//                                         T* pivots);