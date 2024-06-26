#include "../gmx_blas.h"
#include "../gmx_lapack.h"

void F77_FUNC(sgetrs,
              SGETRS)(const char* trans, int* n, int* nrhs, float* a, int* lda, int* ipiv, float* b, int* ldb, int* info)
{
    int   a_dim1, a_offset, b_dim1, b_offset;
    int   notran;
    int   c__1 = 1;
    int   c_n1 = -1;
    float one  = 1.0;

    a_dim1   = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    --ipiv;
    b_dim1   = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    *info  = 0;
    notran = (*trans == 'N' || *trans == 'n');

    if (*n <= 0 || *nrhs <= 0)
        return;

    if (notran)
    {
        F77_FUNC(slaswp, SLASWP)(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c__1);
        F77_FUNC(strsm, STRSM)
        ("Left", "Lower", "No transpose", "Unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);

        F77_FUNC(strsm, STRSM)
        ("Left", "Upper", "No transpose", "Non-unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);
    }
    else
    {
        F77_FUNC(strsm, STRSM)
        ("Left", "Upper", "Transpose", "Non-unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);
        F77_FUNC(strsm, STRSM)
        ("Left", "Lower", "Transpose", "Unit", n, nrhs, &one, &a[a_offset], lda, &b[b_offset], ldb);

        F77_FUNC(slaswp, SLASWP)(nrhs, &b[b_offset], ldb, &c__1, n, &ipiv[1], &c_n1);
    }

    return;
}
