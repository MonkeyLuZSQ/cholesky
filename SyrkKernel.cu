#include "Common.cuh"

template <typename API_INT,
          rocblas_int MIN_NB,
          bool        BATCHED,
          bool        TWOK,
          bool        HERK,
          typename T,
          typename TScala,
          typename TScalb,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syr2k_her2k_template(rocblas_handle    handle,
                                                     rocblas_fill      uplo,
                                                     rocblas_operation trans,
                                                     rocblas_int       n,
                                                     API_INT           k,
                                                     const TScala*     alpha_in,
                                                     TConstPtr         dA_in,
                                                     rocblas_stride    offset_a,
                                                     API_INT           lda,
                                                     rocblas_stride    stride_a,
                                                     TConstPtr         dB_in,
                                                     rocblas_stride    offset_b,
                                                     API_INT           ldb,
                                                     rocblas_stride    stride_b,
                                                     const TScalb*     beta_in,
                                                     TPtr              dC_in,
                                                     rocblas_stride    offset_c,
                                                     API_INT           ldc,
                                                     rocblas_stride    stride_c,
                                                     rocblas_int       batch_count)
{
    // quick return
    if(!n || !batch_count)
        return rocblas_status_success;

    // Copy over alpha and beta
    TScala alpha_h;
    TScalb beta_h;
    RETURN_IF_ROCBLAS_ERROR(rocblas_copy_alpha_beta_to_host_if_on_device(
        handle, alpha_in, beta_in, alpha_h, beta_h, k));
    auto saved_pointer_mode = handle->push_pointer_mode(rocblas_pointer_mode_host);

    // Note: alpha and beta always copied over to host by now
    if(*beta_in == 1 && (k == 0 || *alpha_in == 0))
        return rocblas_status_success;

    bool ab_calc_invalid = !alpha_in || (*alpha_in != 0 && (!dA_in || !dB_in));
    if(!dC_in || (k && ab_calc_invalid))
        return rocblas_status_invalid_pointer;

    // upgrade to complex if needed
    // TODO: Graph safety?
    const T alpha_val = (T)(*alpha_in);
    const T beta_val  = (T)(*beta_in);

    const T* alpha = &alpha_val;
    const T* beta  = &beta_val;

    // Can't use block-recursive algorithm with batched version
    // Can use block-recursive algorithm with strided_batched when batch_count == 1
    if(!BATCHED && batch_count == 1)
    {
        return rocblas_internal_syr2k_syrkx_block_recursive_template<API_INT,
                                                                     MIN_NB,
                                                                     TWOK,
                                                                     HERK,
                                                                     T>(handle,
                                                                        uplo,
                                                                        trans,
                                                                        n,
                                                                        k,
                                                                        alpha,
                                                                        dA_in,
                                                                        offset_a,
                                                                        lda,
                                                                        dB_in,
                                                                        offset_b,
                                                                        ldb,
                                                                        beta,
                                                                        dC_in,
                                                                        offset_c,
                                                                        ldc);
    }

    API_INT a_s1 = rocblas_operation_none == trans ? 1 : lda;
    API_INT b_s1 = rocblas_operation_none == trans ? 1 : ldb;
    API_INT c_s1 = 1, c_s2 = ldc;

    rocblas_int nb = MIN_NB;
    rocblas_int i_diag, n_diag;

    rocblas_int n_nb, rem, i_start = 0;

    n_nb = n / nb; // number of diagonal blocks of size nb
    rem  = n % nb; // size of remainder block when n is not multiple of nb

    const T alpha_conj = conj(*alpha);

    TPtr      dC = dC_in;
    TConstPtr dB = dB_in;
    TConstPtr dA = dA_in;

    static constexpr int syr2k_SCALE_DIM_X = 128;
    static constexpr int syr2k_SCALE_DIM_Y = 8;
    rocblas_int          gx                = (n - 1) / (syr2k_SCALE_DIM_X) + 1;
    rocblas_int          gy                = (n - 1) / (syr2k_SCALE_DIM_Y) + 1;
    dim3                 syr2k_scale_grid(gx, gy, batch_count);
    dim3                 syr2k_scale_threads(syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y);

    // first scale C so we can use directly for output without work buffer
    ROCBLAS_LAUNCH_KERNEL(
        (rocblas_syr2k_scale_kernel<API_INT, syr2k_SCALE_DIM_X, syr2k_SCALE_DIM_Y, HERK>),
        syr2k_scale_grid,
        syr2k_scale_threads,
        0,
        handle->get_stream(),
        uplo == rocblas_fill_upper,
        n,
        k,
        *alpha,
        *beta,
        dC,
        ldc,
        BATCHED ? offset_c : stride_c);

    if(k == 0)
        return rocblas_status_success;

    // n_nb diagonal blocks of size nb
    for(int i_nb = 0; i_nb < n_nb; i_nb++)
    {
        i_diag = i_nb * nb; // diag block at c[i_diag, i_diag], size is nb

        // clang-format off
        rocblas_internal_syr2k_her2k_non_recursive_template<API_INT, MIN_NB, BATCHED, TWOK, HERK>(
                handle, uplo, trans, nb, k, alpha,
                dA, OFFSET_A(i_diag),         lda, stride_a,
                dB, OFFSET_B(i_diag),         ldb, stride_b,
                dC, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    // remainder diagonal block of size n_diag < nb
    if(rem != 0)
    {
        i_diag = n_nb * nb; // diag block at c[i_diag, i_diag], size is n_diag
        n_diag = n - i_diag;

        // clang-format off
        rocblas_internal_syr2k_her2k_non_recursive_template<API_INT, MIN_NB, BATCHED, TWOK, HERK>(
                handle, uplo, trans, n_diag, k, alpha,
                dA, OFFSET_A(i_diag),         lda, stride_a,
                dB, OFFSET_B(i_diag),         ldb, stride_b,
                dC, OFFSET_C(i_diag, i_diag), ldc, stride_c, batch_count);
        // clang-format on
    }

    rocblas_operation trans_orig
        = rocblas_operation_none == trans
              ? rocblas_operation_none
              : (HERK ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose);
    rocblas_operation trans_opp
        = rocblas_operation_none == trans
              ? (HERK ? rocblas_operation_conjugate_transpose : rocblas_operation_transpose)
              : rocblas_operation_none;

    // calls to gemm with m == n == nb.
    // Start with nb == MIN_NB, and each iteration of the outer loop:
    // - nb doubles
    // - the number of gemm calls in the inner loop halves.
    for(nb = MIN_NB, i_start = MIN_NB; i_start < n; i_start += nb, nb *= 2)
    {
        rocblas_int stride = nb * 2;
        n_nb               = (n - i_start) / stride;
        rem                = (n - i_start) % stride;
        if(rem >= nb)
        {
            rem = 0;
            n_nb += 1;
        }
        // n_nb gemm blocks of size nb x nb
        for(int i = 0; i < n_nb; i++)
        {
            rocblas_int i1 = i_start + (i * stride);
            rocblas_int i2 = i1 - nb;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, trans_orig, trans_opp, nb, nb, k, alpha,
                     dA, OFFSET_A(i1),     lda, stride_a,
                     dB, OFFSET_B(i2),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                        handle, trans_orig, trans_opp, nb, nb, k, (HERK? &alpha_conj : alpha),
                        dB, OFFSET_B(i1),     ldb, stride_b,
                        dA, OFFSET_A(i2),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                    // clang-format on
                }
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, trans_orig, trans_opp, nb, nb, k, alpha,
                     dA, OFFSET_A(i2),     lda, stride_a,
                     dB, OFFSET_B(i1),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                        handle, trans_orig, trans_opp, nb, nb, k,(HERK? &alpha_conj : alpha),
                        dB, OFFSET_B(i2),     ldb, stride_b,
                        dA, OFFSET_A(i1),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                    // clang-format on
                }
            }
        }

        // remainder gemm block of size n1 x nb where n1 < nb
        if(rem != 0)
        {
            rocblas_int i1 = i_start + n_nb * stride;
            rocblas_int i2 = i1 - nb;
            rocblas_int n1 = n - i1;

            if(rocblas_fill_lower == uplo)
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, trans_orig, trans_opp, n1, nb, k, alpha,
                     dA, OFFSET_A(i1),     lda, stride_a,
                     dB, OFFSET_B(i2),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                        handle, trans_orig, trans_opp, n1, nb, k,  (HERK? &alpha_conj : alpha),
                        dB, OFFSET_B(i1),     ldb, stride_b,
                        dA, OFFSET_A(i2),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i1, i2), ldc, stride_c, batch_count)));
                    // clang-format on
                }
            }
            else
            {
                // clang-format off
                RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                     handle, trans_orig, trans_opp, nb, n1, k, alpha,
                     dA, OFFSET_A(i2),     lda, stride_a,
                     dB, OFFSET_B(i1),     ldb, stride_b, &beta_1<T>,
                     dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                // clang-format on

                if(TWOK)
                {
                    // clang-format off
                    RETURN_IF_ROCBLAS_ERROR( (rocblas_internal_gemm_64<BATCHED>(
                        handle, trans_orig, trans_opp, nb, n1, k, (HERK? &alpha_conj : alpha),
                        dB, OFFSET_B(i2),     ldb, stride_b,
                        dA, OFFSET_A(i1),     lda, stride_a, &beta_1<T>,
                        dC, OFFSET_C(i2, i1), ldc, stride_c, batch_count)));
                    // clang-format on
                }
            }
        }
    }

    return rocblas_status_success;
}


/**
  * T is base type, i.e. float, double, rocblas_float_complex, or rocblas_double_complex
  * TScal is base type of scalars, for HERM == false, TScal == T, for HERM == true, TScal == real_t<T>
  * TConstPtr is either: const T* OR const T* const*
  * TPtr      is either:       T* OR       T* const*
  */
template <rocblas_int NB,
          bool        BATCHED,
          bool        HERM,
          typename T,
          typename TScal,
          typename TConstPtr,
          typename TPtr>
rocblas_status rocblas_internal_syrk_herk_template(rocblas_handle    handle,
                                                   rocblas_fill      uplo,
                                                   rocblas_operation trans_a,
                                                   rocblas_int       n,
                                                   rocblas_int       k,
                                                   const TScal*      alpha,
                                                   TConstPtr         A,
                                                   rocblas_stride    offset_a,
                                                   rocblas_int       lda,
                                                   rocblas_stride    stride_A,
                                                   const TScal*      beta,
                                                   TPtr              C,
                                                   rocblas_stride    offset_c,
                                                   rocblas_int       ldc,
                                                   rocblas_stride    stride_C,
                                                   rocblas_int       batch_count)
{
    // quick returns handled in rocblas_internal_syr2k_her2k_template
    constexpr bool TWOK = false;
    return rocblas_internal_syr2k_her2k_template<rocblas_int, NB, BATCHED, TWOK, HERM, T>(
        handle,
        uplo,
        trans_a,
        n,
        k,
        alpha,
        A,
        offset_a,
        lda,
        stride_A,
        A,
        offset_a,
        lda,
        stride_A,
        beta,
        C,
        offset_c,
        ldc,
        stride_C,
        batch_count);
}


#define ROCBLAS_INTERNAL_SYRK_HERK_PARAMS                                                   \
    handle, uplo, trans_a, n, k, alpha, A, offset_a, lda, stride_A, beta, C, offset_c, ldc, \
        stride_C, batch_count

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_template(rocblas_handle    handle,
                                   rocblas_fill      uplo,
                                   rocblas_operation trans_a,
                                   rocblas_int       n,
                                   rocblas_int       k,
                                   const T*          alpha,
                                   const T*          A,
                                   rocblas_stride    offset_a,
                                   rocblas_int       lda,
                                   rocblas_stride    stride_A,
                                   const T*          beta,
                                   T*                C,
                                   rocblas_stride    offset_c,
                                   rocblas_int       ldc,
                                   rocblas_stride    stride_C,
                                   rocblas_int       batch_count)
{
    constexpr bool BATCHED = false;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_CSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDZSYRK_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

template <typename T>
ROCBLAS_INTERNAL_EXPORT_NOINLINE rocblas_status
    rocblas_internal_syrk_batched_template(rocblas_handle    handle,
                                           rocblas_fill      uplo,
                                           rocblas_operation trans_a,
                                           rocblas_int       n,
                                           rocblas_int       k,
                                           const T*          alpha,
                                           const T* const*   A,
                                           rocblas_stride    offset_a,
                                           rocblas_int       lda,
                                           rocblas_stride    stride_A,
                                           const T*          beta,
                                           T* const*         C,
                                           rocblas_stride    offset_c,
                                           rocblas_int       ldc,
                                           rocblas_stride    stride_C,
                                           rocblas_int       batch_count)
{
    constexpr bool BATCHED = true;
    constexpr bool HERM    = false;
    if constexpr(std::is_same_v<T, float>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, double>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_SDSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_float_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_CZSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);
    else if constexpr(std::is_same_v<T, rocblas_double_complex>)
        return rocblas_internal_syrk_herk_template<ROCBLAS_CZSYRK_BATCHED_NB, BATCHED, HERM, T>(
            ROCBLAS_INTERNAL_SYRK_HERK_PARAMS);

    return rocblas_status_not_implemented;
}

// syrk
template <bool BATCHED, typename T, typename U, typename V, std::enable_if_t<!rocblas_is_complex<T>, int> = 0>
rocblas_status rocblasCall_syrk_herk(rocblas_handle handle,
                                     rocblas_fill uplo,
                                     rocblas_operation transA,
                                     rocblas_int n,
                                     rocblas_int k,
                                     U alpha,
                                     V A,
                                     rocblas_stride offsetA,
                                     rocblas_int lda,
                                     rocblas_stride strideA,
                                     U beta,
                                     V C,
                                     rocblas_stride offsetC,
                                     rocblas_int ldc,
                                     rocblas_stride strideC,
                                     rocblas_int batch_count)
{
    // TODO: How to get alpha and beta for trace logging
    ROCBLAS_ENTER("syrk", "uplo:", uplo, "trans:", transA, "n:", n, "k:", k, "shiftA:", offsetA,
                  "lda:", lda, "shiftC:", offsetC, "ldc:", ldc, "bc:", batch_count);

    using S = decltype(std::real(T{}));

    if constexpr(BATCHED)
        return rocblas_internal_syrk_batched_template(
            handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA,
            lda, strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
    else
        return rocblas_internal_syrk_template(
            handle, uplo, transA, n, k, cast2constType<S>(alpha), cast2constType<T>(A), offsetA,
            lda, strideA, cast2constType<S>(beta), C, offsetC, ldc, strideC, batch_count);
}