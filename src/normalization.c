#include <cblas.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include "../include/normalization.h"


/*  Appies Min-max feature scaling to data
    n: number of items
    m: item size                            */
void norm_minmax(value_t *data, int n, int m)
{
    value_t *min, *max;
    int i, j, item_size = m * sizeof(value_t);

    min = malloc(item_size);
    max = malloc(item_size);

    memcpy(max, data, item_size);
    memcpy(min, data, item_size);

    for (i = 0; i < m; i++) {
        for (j = m + i; j < n*m; j += m) {
            min[i] = (data[j] < min[i]) ? data[j] : min[i];
            max[i] = (data[j] > max[i]) ? data[j] : max[i];
        }
        max[i] -= min[i];
    }

    for (i = 0; i < m; i++) {
        if (max[i] == 0) {
            for (j = i; j < n*m; j += m)
                data[j] = 0.5;
            continue;
        }
        for (j = i; j < n*m; j += m)
            data[j] = (data[j] - min[i]) / max[i];
    }

    free(min);
    free(max);
}


/*  Appies Standard score scaling to data
    n: number of items
    m: item size (number of features)        */
void norm_standard(value_t *data, int n, int m)
{
    int i;
    value_t *mean, *deviation, *ones;

    mean = malloc(m * sizeof(value_t));
    deviation = malloc(m * sizeof(value_t));
    ones = malloc(n * sizeof(value_t));

    for (i = 0; i < n; i++) ones[i] = 1.0;

    /* mean = 1/n * D * [1]     | (m,1) = (m,n) * (n,1) */
    cblas_dgemv(CblasColMajor, CblasNoTrans,
        m, n, 1.0 / n, data, m, ones, 1, 0.0, mean, 1);

    /* D' = D - mean * [1]^T    | (m,n) = (m,n) - (m,1) * (n,1)^T */
    cblas_dger(CblasColMajor,
        m, n, -1.0, mean, 1, ones, 1, data, m);

    /* Calculating the inverse of standard deviation for each feature */
    for (i = 0; i < m; i++) {
        value_t norm = cblas_dnrm2(n, data + i, m);
        if (norm == 0.0) {
            deviation[i] = 1.0;
        } else {
            deviation[i] = sqrt(n) / norm;
        }
    }

    /* Scale each row of the matrix by the corresponding deviation */
    for (i = 0; i < m; ++i)
        cblas_dscal(n, deviation[i], data + i, m);

    free(mean);
    free(deviation);
    free(ones);
}
