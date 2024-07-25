#include <cblas.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../include/normalization.h"


/*  Appies Min-max feature scaling to data
    n: number of items
    m: item size
    data is a (m, n) column major matrix   */
void norm_minmax(value_t *data, int n, int m)
{
    const int mem_size = (n + 2 * m) * sizeof(value_t);
    void *const mem_block = malloc(mem_size);

    value_t *restrict const min  = mem_block;
    value_t *restrict const max  = min + m;
    value_t *restrict const ones = max + m;
    value_t *restrict dptr = data;

    for (int i = 0; i < n; i++) ones[i] = 1.0;

    memcpy(min, dptr, m * sizeof(value_t));
    memcpy(max, dptr, m * sizeof(value_t));


    for (int i = 0, j = 0; i < m * n; j++, i++) {
        const value_t val = dptr[i];
        j = (j == m) ? 0 : j;
        min[j] = (val < min[j]) ? val : min[j];
        max[j] = (val > max[j]) ? val : max[j];
    }


    /*  do max = 1/(max - min) */
    for (int j = 0; j < m; j++) {
        max[j] -= min[j];
        max[j] = (max[j] != 0.0) ? (1.0 / max[j]) : max[j];
    }

    cblas_ger(CblasColMajor, m, n, -1.0, min, 1, ones, 1, dptr, m);

    for (int i = 0, j = 0; i < m * n; j++, i++) {
        j = (j == m) ? 0 : j;
        dptr[i] *= max[j];
    }

    free(mem_block);
}


/*  Appies Standard score scaling to data
    n: number of items
    m: item size (number of features)        */
void norm_standard(value_t *data, int n, int m)
{
    int i;
    value_t *mean, *deviation, *ones;

    mean =      malloc(m * sizeof(value_t));
    deviation = malloc(m * sizeof(value_t));
    ones =      malloc(n * sizeof(value_t));

    for (i = 0; i < n; i++) ones[i] = 1.0;

    /* mean = 1/n * D * [1]     | (m,1) = (m,n) * (n,1) */
    cblas_gemv(CblasColMajor, CblasNoTrans,
        m, n, 1.0 / n, data, m, ones, 1, 0.0, mean, 1);

    /* D' = D - mean * [1]^T    | (m,n) = (m,n) - (m,1) * (n,1)^T */
    cblas_ger(CblasColMajor,
        m, n, -1.0, mean, 1, ones, 1, data, m);

    /* Calculating the inverse of standard deviation for each feature */
    for (i = 0; i < m; i++) {
        value_t norm = cblas_nrm2(n, data + i, m);
        deviation[i] = (norm == 0.0) ? 1.0 : _sqrt(n) / norm;
    }

    /* Scale each row of the matrix by the corresponding deviation */
    for (i = 0; i < m; ++i)
        cblas_scal(n, deviation[i], data + i, m);

    free(mean);
    free(deviation);
    free(ones);
}
