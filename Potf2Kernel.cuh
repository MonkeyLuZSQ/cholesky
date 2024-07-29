#pragma once

#include "Common.cuh"
#include "DotKernel.cuh"
#include "GemvKernel.cuh"
#include "ScalKernel.cuh"

/************************** potf2/potrf ***************************************
*******************************************************************************/
/*! \brief Determines the size of the leading block that is factorized at each
   step when using the blocked algorithm (POTRF). It also applies to the
    corresponding batched and strided-batched routines.*/
#ifndef POTRF_BLOCKSIZE
#define POTRF_BLOCKSIZE(T)                                                     \
  ((sizeof(T) == 4) ? 180 : (sizeof(T) == 8) ? 127 : 90)
#endif

/*! \brief Determines the size at which rocSOLVER switches from
    the unblocked to the blocked algorithm when executing POTRF. It also applies
   to the corresponding batched and strided-batched routines.

    \details POTRF will factorize blocks of POTRF_BLOCKSIZE columns at a time
   until the rest of the matrix has no more than POTRF_POTF2_SWITCHSIZE columns;
   at this point the last block, if any, will be factorized with the unblocked
   algorithm (POTF2).*/
#ifndef POTRF_POTF2_SWITCHSIZE
#define POTRF_POTF2_SWITCHSIZE(T) POTRF_BLOCKSIZE(T)
#endif

/*! \brief Determines the maximum size at which rocSOLVER can use POTF2
    \details
    POTF2 will attempt to factorize a small symmetric matrix that can fit
   entirely within the LDS share memory using compact storage. The amount of LDS
   shared memory is assumed to be at least (64 * 1024) bytes. */
#ifndef POTF2_MAX_SMALL_SIZE
#define POTF2_MAX_SMALL_SIZE(T)                                                \
  ((sizeof(T) == 4) ? 180 : (sizeof(T) == 8) ? 127 : 90)
#endif

template <typename T, typename U>
__global__ void sqrtDiagOnward(U A, const int shiftA, const int strideA,
                               const size_t loc, const int j, T *res,
                               int *info) {
  int id = blockIdx.x;

  T *M = load_ptr_batch<T>(A, id, shiftA, strideA);
  T t = M[loc] - res[id];

  if (t <= 0.0) {
    // error for non-positive definiteness
    if (info[id] == 0)
      info[id] = j + 1; // use fortran 1-based index
    M[loc] = t;
    res[id] = 0;
  }

  else {
    // minor is positive definite
    M[loc] = sqrt(t);
    res[id] = 1 / M[loc];
  }
}

template <typename T>
void rocsolver_potf2_getMemorySize(const int n, const int batch_count,
                                   size_t *size_scalars, size_t *size_work,
                                   size_t *size_pivots) {
  // if quick return no need of workspace
  if (n == 0 || batch_count == 0) {
    *size_scalars = 0;
    *size_work = 0;
    *size_pivots = 0;
    return;
  }

  // size of scalars (constants)
  *size_scalars = sizeof(T) * 3;

  if (n <= POTF2_MAX_SMALL_SIZE(T)) {
    *size_work = 0;
    *size_pivots = 0;
    return;
  }

  // size of workspace
  // TODO: replace with rocBLAS call
  *size_work = sizeof(T) * ((n - 1) / ROCBLAS_DOT_NB + 2) * batch_count;

  // size of array to store pivots
  *size_pivots = sizeof(T) * batch_count;
}

template <typename T>
cublasStatus_t rocsolver_potf2_potrf_argCheck(cublasHandle_t handle,
                                              const int n, const int lda, T A,
                                              int *info,
                                              const int batch_count = 1) {
  // order is important for unit tests:

  // // 1. invalid/non-supported values
  // if(uplo != rocblas_fill_upper && uplo != rocblas_fill_lower)
  //     return rocblas_status_invalid_value;

  // 2. invalid size
  if (n < 0 || lda < n || batch_count < 0)
    return CUBLAS_STATUS_INVALID_VALUE;

  // // skip pointer check if querying memory size
  // if(rocblas_is_device_memory_size_query(handle))
  //     return rocblas_status_continue;

  // 3. invalid pointers
  if ((n && !A) || (batch_count && !info))
    return CUBLAS_STATUS_INVALID_VALUE;

  // return rocblas_status_continue;
  // return CUBLAS_STATUS
  return CUBLAS_STATUS_SUCCESS;
}

/**
 * ------------------------------------------------------
 * Perform Cholesky factorization for small n by n matrix.
 * The function executes in a single thread block.
 * ------------------------------------------------------
 **/
template <typename T, typename I>
__device__ static void potf2_simple(bool const is_upper, I const n, T *const A,
                                    I *const info) {
  auto const lda = n;
  bool const is_lower = (!is_upper);

  auto const i_start = threadIdx.x;
  auto const i_inc = blockDim.x;
  auto const j_start = threadIdx.y;
  auto const j_inc = blockDim.y;
  assert(blockDim.z == 1);

  auto const tid = threadIdx.x + threadIdx.y * blockDim.x +
                   threadIdx.z * (blockDim.x * blockDim.y);
  auto const nthreads = (blockDim.x * blockDim.y) * blockDim.z;

  auto const j0_start = tid;
  auto const j0_inc = nthreads;

  if (is_lower) {
    // ---------------------------------------------------
    // [  l11     ]  * [ l11'   vl21' ]  =  [ a11       ]
    // [ vl21  L22]    [        L22' ]     [ va21, A22 ]
    //
    //
    //   assume l11 is scalar 1x1 matrix
    //
    //   (1) l11 * l11' = a11 =>  l11 = sqrt( abs(a11) ), scalar computation
    //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
    //   (3) L22 * L22' + vl21 * vl21' = A22
    //
    //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
    //   (3b) L22 * L22' = A22,   cholesky factorization, tail recursion
    // ---------------------------------------------------

    for (I kcol = 0; kcol < n; kcol++) {
      auto kk = idx_lower(kcol, kcol, lda);
      auto const akk = std::real(A[kk]);
      bool const isok = (akk > 0) && (std::isfinite(akk));
      if (!isok) {
        if (tid == 0) {
          A[kk] = akk;
          // Fortran 1-based index
          if (*info == 0)
            *info = kcol + 1;
        }
        break;
      }

      auto const lkk = std::sqrt(akk);
      if (tid == 0) {
        A[kk] = lkk;
      }

      __syncthreads();

      // ------------------------------------------------------------
      //   (2) vl21 * l11' = va21 =>  vl21 = va21/ l11', scale vector
      // ------------------------------------------------------------

      auto const conj_lkk = conj(lkk);
      for (I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc) {
        auto const j0k = idx_lower(j0, kcol, lda);

        A[j0k] = (A[j0k] / conj_lkk);
      }

      __syncthreads();

      // ------------------------------------------------------------
      //   (3a) A22 = A22 - vl21 * vl21',  symmetric rank-1 update
      //
      //   note: update lower triangular part
      // ------------------------------------------------------------

      for (I j = (kcol + 1) + j_start; j < n; j += j_inc) {
        auto const vj = A[idx_lower(j, kcol, lda)];
        for (I i = (kcol + 1) + i_start; i < n; i += i_inc) {
          bool const lower_part = (i >= j);
          if (lower_part) {
            auto const vi = A[idx_lower(i, kcol, lda)];
            auto const ij = idx_lower(i, j, lda);

            A[ij] = A[ij] - vi * conj(vj);
          }
        }
      }

      __syncthreads();

    } // end for kcol
  } else {
    // --------------------------------------------------
    // [u11'        ] * [u11    vU12 ] = [ a11     vA12 ]
    // [vU12'   U22']   [       U22  ]   [ vA12'   A22  ]
    //
    // (1) u11' * u11 = a11 =?  u11 = sqrt( abs( a11 ) )
    // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
    //     or vU12 = vA12/u11'
    // (3) vU12' * vU12 + U22'*U22 = A22
    //
    // (3a) A22 = A22 - vU12' * vU12
    // (3b) U22' * U22 = A22,  cholesky factorization, tail recursion
    // --------------------------------------------------

    for (I kcol = 0; kcol < n; kcol++) {
      auto const kk = idx_upper(kcol, kcol, lda);
      auto const akk = std::real(A[kk]);
      bool const isok = (akk > 0) && (std::isfinite(akk));
      if (!isok) {
        if (tid == 0) {
          A[kk] = akk;
          // Fortran 1-based index
          if (*info == 0)
            *info = kcol + 1;
        }

        break;
      }

      auto const ukk = std::sqrt(akk);
      if (tid == 0) {
        A[kk] = ukk;
      }
      __syncthreads();

      // ----------------------------------------------
      // (2) vU12' * u11 = vA12', or u11' * vU12 = vA12
      // ----------------------------------------------
      for (I j0 = (kcol + 1) + j0_start; j0 < n; j0 += j0_inc) {
        auto const kj0 = idx_upper(kcol, j0, lda);

        A[kj0] = A[kj0] / ukk;
      }

      __syncthreads();

      // -----------------------------
      // (3a) A22 = A22 - vU12' * vU12
      //
      // note: update upper triangular part
      // -----------------------------
      for (I j = (kcol + 1) + j_start; j < n; j += j_inc) {
        auto const vj = A[idx_upper(kcol, j, lda)];
        for (I i = (kcol + 1) + i_start; i < n; i += i_inc) {
          bool const upper_part = (i <= j);
          if (upper_part) {
            auto const vi = A[idx_upper(kcol, i, lda)];
            auto const ij = idx_upper(i, j, lda);

            A[ij] = A[ij] - conj(vi) * vj;
          }
        }
      }

      __syncthreads();

    } // end for kcol
  }
}

/*************************************************************
    Templated kernels are instantiated in separate cpp
    files in order to improve compilation times and reduce
    the library size.
*************************************************************/

template <typename T, typename U>
__global__ void potf2_kernel_small(const bool is_upper, const int n, U AA,
                                   const int shiftA, const int lda,
                                   const int64_t strideA, int *const info) {
  bool const is_lower = (!is_upper);

  auto const i_start = threadIdx.x;
  auto const i_inc = blockDim.x;
  auto const j_start = threadIdx.y;
  auto const j_inc = blockDim.y;
  assert(blockDim.z == 1);

  // --------------------------------
  // note hipGridDim_z == batch_count
  // --------------------------------
  auto const bid = blockIdx.z;
  assert(AA != nullptr);
  assert(info != nullptr);

  T *const A = load_ptr_batch(AA, bid, shiftA, strideA);
  int *const info_bid = info + bid;

  assert(A != nullptr);

  // -----------------------------------------
  // assume n by n matrix will fit in LDS cache
  // -----------------------------------------
  extern __shared__ int lsmem[];
  T *Ash = reinterpret_cast<T *>(lsmem);

  // --------------------------------------------------------
  // factoring Lower triangular matrix may be slightly faster
  // due to simpler index calculation down a column
  // --------------------------------------------------------
  bool const use_compute_lower = true;

  // ------------------------------------
  // copy n by n packed matrix into shared memory
  // ------------------------------------
  __syncthreads();

  if (is_lower) {
    for (int j = j_start; j < n; j += j_inc) {
      for (int i = j + i_start; i < n; i += i_inc) {
        auto const ij = i + j * static_cast<int64_t>(lda);
        auto const ij_packed = idx_lower(i, j, n);

        Ash[ij_packed] = A[ij];
      }
    }
  } else {
    for (int j = j_start; j < n; j += j_inc) {
      for (int i = i_start; i <= j; i += i_inc) {
        auto const ij = i + j * static_cast<int64_t>(lda);
        auto const ij_packed =
            (use_compute_lower) ? idx_lower(j, i, n) : idx_upper(i, j, n);

        auto const aij = A[ij];
        Ash[ij_packed] = (use_compute_lower) ? conj(aij) : aij;
      }
    }
  }

  __syncthreads();

  bool const is_up = (use_compute_lower) ? false : is_upper;
  potf2_simple<T>(is_up, n, Ash, info_bid);

  __syncthreads();

  // -------------------------------------
  // copy n by n packed matrix into global memory
  // -------------------------------------
  if (is_lower) {
    for (int j = j_start; j < n; j += j_inc) {
      for (int i = j + i_start; i < n; i += i_inc) {
        auto const ij = i + j * static_cast<int64_t>(lda);
        auto const ij_packed = idx_lower(i, j, n);

        A[ij] = Ash[ij_packed];
      }
    }
  } else {
    for (int j = j_start; j < n; j += j_inc) {
      for (int i = i_start; i <= j; i += i_inc) {
        auto const ij = i + j * static_cast<int64_t>(lda);
        auto const ij_packed =
            (use_compute_lower) ? idx_lower(j, i, n) : idx_upper(i, j, n);

        auto const aij_packed = Ash[ij_packed];
        A[ij] = (use_compute_lower) ? conj(aij_packed) : aij_packed;
      }
    }
  }

  __syncthreads();
}

template <typename T, typename U>
cublasStatus_t
potf2_run_small(cublasHandle_t handle, const cublasFillMode_t uplo, const int n,
                U A, const int shiftA, const int lda, const int64_t strideA,
                int *info, const int batch_count) {

  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  size_t lmemsize = sizeof(T) * (n * (n + 1)) / 2;

  bool const is_upper = (uplo == CUBLAS_FILL_MODE_UPPER);
  potf2_kernel_small<T, U>
      <<<dim3(1, 1, batch_count), dim3(BS2, BS2, 1), lmemsize, stream>>>(
          is_upper, n, A, shiftA, lda, strideA, info);
  return CUBLAS_STATUS_SUCCESS;
}

template <typename T, typename U>
cublasStatus_t rocsolver_potf2_template(cublasHandle_t handle,
                                        const cublasFillMode_t uplo,
                                        const int n, U A, const int shiftA,
                                        const int lda, const int64_t strideA,
                                        int *info, const int batch_count,
                                        T *scalars, T *work, T *pivots) {

  // quick return if zero instances in batch
  if (batch_count == 0)
    return CUBLAS_STATUS_SUCCESS;

  cudaStream_t stream;
  cublasGetStream(handle, &stream);

  int blocksReset = (batch_count - 1) / BS1 + 1;
  dim3 gridReset(blocksReset, 1, 1);
  dim3 threads(BS1, 1, 1);

  // info=0 (starting with a positive definite matrix)
  reset_info<<<gridReset, threads, 0, stream>>>(info, batch_count, 0);
  // ROCSOLVER_LAUNCH_KERNEL(reset_info, gridReset, threads, 0, stream, info,
  // batch_count, 0);

  // quick return if no dimensions
  if (n == 0)
    return CUBLAS_STATUS_SUCCESS;

  // everything must be executed with scalars on the device
  cublasPointerMode_t old_mode;
  cublasGetPointerMode(handle, &old_mode);
  cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

  if (n <= POTRF_BLOCKSIZE(T)) {
    // ----------------------
    // use specialized kernel
    // ----------------------
    potf2_run_small<T>(handle, uplo, n, A, shiftA, lda, strideA, info,
                       batch_count);
  } else {
    // (TODO: When the matrix is detected to be non positive definite, we need
    // to prevent GEMV and SCAL to modify further the input matrix; ideally with
    // no synchronizations.)

    if (uplo == CUBLAS_FILL_MODE_UPPER) {
      // Compute the Cholesky factorization A = U'*U.
      for (int j = 0; j < n; ++j) {
        // Compute U(J,J) and test for non-positive-definiteness.
        rocblasCall_dot<T>(handle, j, A, shiftA + idx2D(0, j, lda), 1, strideA,
                           A, shiftA + idx2D(0, j, lda), 1, strideA,
                           batch_count, pivots, work);

        ROCSOLVER_LAUNCH_KERNEL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1),
                                0, stream, A, shiftA, strideA, idx2D(j, j, lda),
                                j, pivots, info);

        // Compute elements J+1:N of row J
        if (j < n - 1) {

          rocblasCall_gemv<T>(handle, j, n - j - 1, scalars, 0, A,
                              shiftA + idx2D(0, j + 1, lda), lda, strideA, A,
                              shiftA + idx2D(0, j, lda), 1, strideA,
                              scalars + 2, 0, A, shiftA + idx2D(j, j + 1, lda),
                              lda, strideA, batch_count, nullptr);
          rocblasCall_scal<T>(handle, n - j - 1, pivots, 1, A,
                              shiftA + idx2D(j, j + 1, lda), lda, strideA,
                              batch_count);
        }
      }
    } else {
      // Compute the Cholesky factorization A = L'*L.
      for (int j = 0; j < n; ++j) {
        // Compute L(J,J) and test for non-positive-definiteness.
        rocblasCall_dot<T>(handle, j, A, shiftA + idx2D(j, 0, lda), lda,
                           strideA, A, shiftA + idx2D(j, 0, lda), lda, strideA,
                           batch_count, pivots, work);

        ROCSOLVER_LAUNCH_KERNEL(sqrtDiagOnward<T>, dim3(batch_count), dim3(1),
                                0, stream, A, shiftA, strideA, idx2D(j, j, lda),
                                j, pivots, info);

        // Compute elements J+1:N of column J
        if (j < n - 1) {
          rocblasCall_gemv<T>(handle, n - j - 1, j, scalars, 0, A,
                              shiftA + idx2D(j + 1, 0, lda), lda, strideA, A,
                              shiftA + idx2D(j, 0, lda), lda, strideA,
                              scalars + 2, 0, A, shiftA + idx2D(j + 1, j, lda),
                              1, strideA, batch_count, nullptr);

          rocblasCall_scal<T>(handle, n - j - 1, pivots, 1, A,
                              shiftA + idx2D(j + 1, j, lda), 1, strideA,
                              batch_count);
        }
      }
    }
  }

  cublasSetPointerMode(handle, old_mode);
  return CUBLAS_STATUS_SUCCESS;
}
