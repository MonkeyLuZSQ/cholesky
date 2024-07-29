#pragma once

#include "Common.cuh"

static __device__ int rocblas_log2ui(int x) {
  unsigned int ax = (unsigned int)x;
  int v = 0;
  while (ax >>= 1) {
    v++;
  }
  return v;
}

template <int N> __inline__ __device__ half rocblas_wavefront_reduce(half val) {
  union {
    int i;
    half h;
  } tmp;
  int WFBITS = rocblas_log2ui(N);
  int offset = 1 << (WFBITS - 1);
  for (int i = 0; i < WFBITS; i++) {
    tmp.h = val;
    tmp.i = __shfl_down_sync(0xFFFFFFFF, tmp.i, offset, warpSize);
    val += tmp.h;
    offset >>= 1;
  }
  return val;
}

template <int NB, typename T>
__inline__ __device__ T rocblas_dot_block_reduce(T val) {
  __shared__ T psums[warpSize];

  int wavefront = threadIdx.x / warpSize;
  int wavelet = threadIdx.x % warpSize;

  if (wavefront == 0)
    psums[wavelet] = T(0);
  __syncthreads();

  val = rocblas_wavefront_reduce<warpSize>(val); // sum over wavefront
  if (wavelet == 0)
    psums[wavefront] = val; // store sum for wavefront

  __syncthreads(); // Wait for all wavefront reductions

  // ensure wavefront was run
  static constexpr int num_wavefronts = NB / warpSize;
  val = (threadIdx.x < num_wavefronts) ? psums[wavelet] : T(0);
  if (wavefront == 0)
    val = rocblas_wavefront_reduce<num_wavefronts>(val); // sum wavefront sums

  return val;
}

template <typename API_INT>
inline size_t rocblas_reduction_kernel_block_count(API_INT n, int NB) {
  if (n <= 0)
    n = 1; // avoid sign loss issues
  return size_t(n - 1) / NB + 1;
}

// work item number (WIN) of elements to process
template <typename T> constexpr int rocblas_dot_WIN() {
  size_t nb = sizeof(T);

  int n = 8;
  if (nb >= 8)
    n = 2;
  else if (nb >= 4)
    n = 4;

  return n;
}

constexpr int rocblas_dot_WIN(size_t nb) {
  int n = 8;
  if (nb >= 8)
    n = 2;
  else if (nb >= 4)
    n = 4;

  return n;
}

template <bool ONE_BLOCK, typename V, typename T>
__inline__ __device__ void
rocblas_dot_save_sum(V sum, V *__restrict__ workspace, T *__restrict__ out) {
  if (threadIdx.x == 0) {
    if (ONE_BLOCK || gridDim.x == 1) // small N avoid second kernel
      out[blockIdx.y] = T(sum);
    else
      workspace[blockIdx.x + size_t(blockIdx.y) * gridDim.x] = sum;
  }
}

template <bool ONE_BLOCK, int NB, int WIN, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1(int n, const U __restrict__ xa, int64_t shiftx,
                        int64_t stridex, const U __restrict__ ya,
                        int64_t shifty, int64_t stridey,
                        V *__restrict__ workspace, T *__restrict__ out) {
  const auto *x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
  const auto *y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

  int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

  V sum = 0;

  // sum WIN elements per thread
  int inc = !ONE_BLOCK ? blockDim.x * gridDim.x : blockDim.x;
  for (int j = 0; j < WIN && i < n; j++, i += inc) {
    sum += V(y[i]) * V(x[i]);
  }

  sum = rocblas_dot_block_reduce<NB>(sum);

  rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <bool ONE_BLOCK, int NB, int WIN, typename T, typename U, typename V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_inc1by2(int n, const U __restrict__ xa, int64_t shiftx,
                           int64_t stridex, const U __restrict__ ya,
                           int64_t shifty, int64_t stridey,
                           V *__restrict__ workspace, T *__restrict__ out) {
  const auto *x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
  const auto *y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

  V sum = 0;
  int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

  // sum WIN elements per thread
  int inc = !ONE_BLOCK ? blockDim.x * gridDim.x : blockDim.x;

  if constexpr (std::is_same_v<T, half> || std::is_same_v<T, nv_bfloat16> ||
                std::is_same_v<T, float>) {
    i *= 2;
    inc *= 2;
    for (int j = 0; j < WIN && i < n - 1; j++, i += inc) {
#pragma unroll
      for (int k = 0; k < 2; ++k) {
        sum += V(y[i + k]) * V(x[i + k]);
      }
    }
    // If `n` is odd then the computation of last element is covered below.
    if (n % 2 && i == n - 1) {
      sum += V(y[i]) * V(x[i]);
    }
  } else {
    for (int j = 0; j < WIN && i < n; j++, i += inc) {
      sum += V(y[i]) * V(x[i]);
    }
  }

  sum = rocblas_dot_block_reduce<NB>(sum);

  rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <typename API_INT, bool ONE_BLOCK, int NB, int WIN, typename T,
          typename U, typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel(int n, const U __restrict__ xa, int64_t shiftx, API_INT incx,
                   int64_t stridex, const U __restrict__ ya, int64_t shifty,
                   API_INT incy, int64_t stridey, V *__restrict__ workspace,
                   T *__restrict__ out) {
  const auto *x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);
  const auto *y = load_ptr_batch(ya, blockIdx.y, shifty, stridey);

  int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

  V sum = 0;

  // sum WIN elements per thread
  int inc = blockDim.x * gridDim.x;
  for (int j = 0; j < WIN && i < n; j++, i += inc) {
    sum += V(y[i * int64_t(incy)]) * V(x[i * int64_t(incx)]);
  }
  sum = rocblas_dot_block_reduce<NB>(sum);

  rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <typename API_INT, bool ONE_BLOCK, int NB, int WIN, typename T,
          typename U, typename V = T>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_magsq(int n, const U __restrict__ xa, int64_t shiftx,
                         API_INT incx, int64_t stridex,
                         V *__restrict__ workspace, T *__restrict__ out) {
  const auto *x = load_ptr_batch(xa, blockIdx.y, shiftx, stridex);

  int i = !ONE_BLOCK ? blockIdx.x * blockDim.x + threadIdx.x : threadIdx.x;

  V sum = 0;

  // sum WIN elements per thread
  int inc = blockDim.x * gridDim.x;
  for (int j = 0; j < WIN && i < n; j++, i += inc) {
    int64_t idx = i * int64_t(incx);
    sum += V(x[idx]) * V(x[idx]);
  }
  sum = rocblas_dot_block_reduce<NB>(sum);

  rocblas_dot_save_sum<ONE_BLOCK>(sum, workspace, out);
}

template <int NB, int WIN, typename V, typename T = V>
ROCBLAS_KERNEL(NB)
rocblas_dot_kernel_reduce(int n_sums, V *__restrict__ in, T *__restrict__ out) {
  V sum = 0;

  size_t offset = size_t(blockIdx.y) * n_sums;
  in += offset;

  int inc = blockDim.x * gridDim.x * WIN;

  int i = threadIdx.x * WIN;
  int remainder = n_sums % WIN;
  int end = n_sums - remainder;
  for (; i < end; i += inc) // cover all sums as 1 block
  {
    for (int j = 0; j < WIN; j++)
      sum += in[i + j];
  }
  if (threadIdx.x < remainder) {
    sum += in[n_sums - 1 - threadIdx.x];
  }

  sum = rocblas_dot_block_reduce<NB>(sum);
  if (threadIdx.x == 0)
    out[blockIdx.y] = T(sum);
}

template <typename API_INT, int NB_X, int NB_Y, bool CONJ, typename V,
          typename T, typename U>
ROCBLAS_KERNEL(NB_X *NB_Y)
rocblas_dot_batched_4_kernel(int n, const U __restrict__ xa, int64_t shiftx,
                             API_INT incx, int64_t stridex,
                             const U __restrict__ ya, int64_t shifty,
                             API_INT incy, int64_t stridey, int batch_count,
                             T *__restrict__ out) {
  // Thread Blocks more than or equal to the batch_count could be safely
  // returned
  if (blockIdx.x * NB_Y + threadIdx.y >= batch_count)
    return;

  const auto *x =
      load_ptr_batch(xa, blockIdx.x * NB_Y + threadIdx.y, shiftx, stridex);
  const auto *y =
      load_ptr_batch(ya, blockIdx.x * NB_Y + threadIdx.y, shifty, stridey);

  V reg_x = V(0), reg_y = V(0), sum = V(0);

  for (int tid = threadIdx.x; tid < n; tid += blockDim.x) {
    reg_x = V(CONJ ? conj(x[tid * int64_t(incx)]) : x[tid * int64_t(incx)]);
    reg_y = V(y[tid * int64_t(incy)]);
    sum += reg_x * reg_y;
  }
  __syncthreads();

  sum = rocblas_wavefront_reduce<NB_X>(sum); // sum over wavefront

  if (threadIdx.x == 0)
    out[blockIdx.x * NB_Y + threadIdx.y] = T(sum);
}

// assume workspace has already been allocated, recommended for repeated calling
// of dot_strided_batched product routine
template <typename API_INT, int NB, bool CONJ, typename T, typename U,
          typename V>
cublasStatus_t rocblas_internal_dot_launcher(
    cublasHandle_t __restrict__ handle, API_INT n, const U __restrict__ x,
    int64_t offsetx, API_INT incx, int64_t stridex, const U __restrict__ y,
    int64_t offsety, API_INT incy, int64_t stridey, API_INT batch_count,
    T *__restrict__ results, V *__restrict__ workspace) {

  // One or two kernels are used to finish the reduction
  // kernel 1 write partial results per thread block in workspace, number of
  // partial results is blocks kernel 2 if blocks > 1 the partial results in
  // workspace are reduced to output

  // Quick return if possible.
  if (n <= 0 || batch_count == 0) {
    // if(handle->is_device_memory_size_query())
    //     return cublasStatus_t_size_unchanged;
    // else if(rocblas_pointer_mode_device == handle->pointer_mode &&
    // batch_count > 0)
    // {
    //     RETURN_IF_HIP_ERROR(
    //         hipMemsetAsync(&results[0], 0, batch_count * sizeof(T),
    //         handle->get_stream()));
    // }
    // else
    // {
    for (int i = 0; i < batch_count; i++) {
      results[i] = T(0);
    }
    // }

    return CUBLAS_STATUS_SUCCESS;
  }

  static constexpr int WIN = rocblas_dot_WIN<T>();

  // in case of negative inc shift pointer to end of data for negative indexing
  // tid*inc
  int64_t shiftx = incx < 0 ? offsetx - int64_t(incx) * (n - 1) : offsetx;
  int64_t shifty = incy < 0 ? offsety - int64_t(incy) * (n - 1) : offsety;

  static constexpr bool ONE_BLOCK = false;

  int blocks = rocblas_reduction_kernel_block_count(n, NB * WIN);
  dim3 grid(blocks, batch_count);
  dim3 threads(NB);

  T *output = results;
  cublasPointerMode_t pointer_mode;
  cublasGetPointerMode(handle, &pointer_mode);
  if (pointer_mode == CUBLAS_POINTER_MODE_HOST) {
    size_t offset = size_t(batch_count) * blocks;
    output = (T *)(workspace + offset);
  }
  cudaStream_t stream;
  cublasGetStream(handle, &stream);
  if (x != y || incx != incy || offsetx != offsety || stridex != stridey) {
    if (incx == 1 && incy == 1) {
      rocblas_dot_kernel_inc1<ONE_BLOCK, NB, WIN, T>
          <<<grid, threads, 0, stream>>>(n, x, shiftx, stridex, y, shifty,
                                         stridey, workspace, output);
    } else {
      rocblas_dot_kernel<API_INT, ONE_BLOCK, NB, WIN, T>
          <<<grid, threads, 0, stream>>>(n, x, shiftx, incx, stridex, y, shifty,
                                         incy, stridey, workspace, output);
    }
  } else // x dot x
  {
    rocblas_dot_kernel_magsq<API_INT, ONE_BLOCK, NB, WIN, T>
        <<<grid, threads, 0, stream>>>(n, x, shiftx, incx, stridex, workspace,
                                       output);
  }

  if (blocks > 1) // if single block first kernel did all work
    rocblas_dot_kernel_reduce<NB, WIN>
        <<<dim3(1, batch_count), threads, 0, stream>>>(blocks, workspace,
                                                       output);

  if (pointer_mode == CUBLAS_POINTER_MODE_HOST) {
    cudaMemcpyAsync(&results[0], output, sizeof(T) * batch_count,
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }

  return CUBLAS_STATUS_SUCCESS;
}

template <typename T, typename Tex>
cublasStatus_t rocblas_internal_dot_template(
    cublasHandle_t __restrict__ handle, int n, const T *__restrict__ x,
    int64_t offsetx, int incx, int64_t stridex, const T *__restrict__ y,
    int64_t offsety, int incy, int64_t stridey, int batch_count,
    T *__restrict__ results, Tex *__restrict__ workspace) {
  return rocblas_internal_dot_launcher<int, ROCBLAS_DOT_NB, false>(
      handle, n, x, offsetx, incx, stridex, y, offsety, incy, stridey,
      batch_count, results, workspace);
}

template <typename T, typename Tex>
cublasStatus_t rocblas_internal_dot_batched_template(
    cublasHandle_t __restrict__ handle, int n, const T *const *__restrict__ x,
    int64_t offsetx, int incx, int64_t stridex, const T *const *__restrict__ y,
    int64_t offsety, int incy, int64_t stridey, int batch_count,
    T *__restrict__ results, Tex *__restrict__ workspace) {
  return rocblas_internal_dot_launcher<int, ROCBLAS_DOT_NB, false>(
      handle, n, x, offsetx, incx, stridex, y, offsety, incy, stridey,
      batch_count, results, workspace);
}

// // dot
// template <bool CONJ, typename T, typename Tex>
// cublasStatus_t rocblasCall_dot(cublasHandle_t handle, int n, const T *x,
//                                int64_t offsetx, int incx, int64_t stridex,
//                                const T *y, int64_t offsety, int incy,
//                                int64_t stridey, int batch_count, T *results,
//                                Tex *workspace, T **work = nullptr) {
//   // ROCBLAS_ENTER("dot", "n:", n, "shiftX:", offsetx, "incx:", incx,
//   "shiftY:",
//   // offsety,
//   //               "incy:", incy, "bc:", batch_count);

//   // if constexpr(CONJ)
//   //     return rocblas_internal_dotc_template(handle, n, x, offsetx, incx,
//   //     stridex, y, offsety,
//   //                                           incy, stridey, batch_count,
//   //                                           results, workspace);
//   // else
//   return rocblas_internal_dot_template(handle, n, x, offsetx, incx, stridex,
//   y,
//                                        offsety, incy, stridey, batch_count,
//                                        results, workspace);
// }

template <typename T>
cublasStatus_t rocblasCall_dot(cublasHandle_t handle, int n, T *x,
                               int64_t offsetx, int incx, int64_t stridex,
                               T *const y[], int64_t offsety, int incy,
                               int64_t stridey, int batch_count, T *results,
                               T *workspace, T **work) {
  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  int blocks = (batch_count - 1) / 256 + 1;
  get_array<<<dim3(blocks), dim3(256), 0, stream>>>(work, x, stridex,
                                                    batch_count);

  return rocblas_internal_dot_batched_template(
      handle, n, cast2constType<T>(work), offsetx, incx, stridex, y, offsety,
      incy, stridey, batch_count, results, workspace);
}

template cublasStatus_t rocblasCall_dot<half>(cublasHandle_t handle, int n,
                                              half *x, int64_t offsetx,
                                              int incx, int64_t stridex,
                                              half *const y[], int64_t offsety,
                                              int incy, int64_t stridey,
                                              int batch_count, half *results,
                                              half *workspace, half **work);