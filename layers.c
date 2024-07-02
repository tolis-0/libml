#include <cblas.h>
#include "nn.h"


void dense_forward(dim_t d, const value_t *x, const weight_t *W, value_t *y)
{
    int input_d = d[0], output_d = d[1];

    /* y = W^T * x */
    cblas_dgemv(CblasRowMajor, CblasTrans,
        input_d, output_d, 1.0, W, output_d,
        x, 1, 0.0, y, 1);
}

void dense_backward(dim_t d, value_t *x, weight_t *W,
    int calc_x, grad_t *Gy, grad_t *Gx, grad_t *GW)
{
    int input_d = d[0], output_d = d[1], i, j, k = 0;

    /* GW = x * Gy^T */
    for (i = 0; i < input_d; i++)
        for (j = 0; j < output_d; j++)
            GW[k++] = x[i] * Gy[j];

    /* Gx = W^T * Gy */
    if (calc_x) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
            input_d, output_d, 1.0, W, input_d,
            Gy, 1, 0.0, Gx, 1);
    }
}
