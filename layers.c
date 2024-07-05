#include <cblas.h>
#include "nn.h"


/*  Computes the result y of a single forward pass in the dense layer */
void dense_forward(dim_t d, const value_t *x, const weight_t *W,
    int has_bias, const weight_t *b, value_t *y)
{
    int input_d = d[0], output_d = d[1];

    /* y = W^T * x      | (d1,1) = (d0,d1)^T * (d0,1) */
    cblas_dgemv(CblasColMajor, CblasTrans,
        input_d, output_d, 1.0, W, input_d, x, 1, 0.0, y, 1);

    /* y = b + y        | (d1,1) = (d1,1) + (d1,1) */
    if (has_bias) cblas_daxpy(output_d, 1.0, b, 1, y, 1);
}


/*  Computes the gradients of a single backward pass where the output
    and gradient of the output of the layer is given. Gb is always Gy */
void dense_backward(dim_t d, value_t *x, weight_t *W,
    int calc_x, grad_t *Gy, grad_t *Gx, grad_t *GW)
{
    int input_d = d[0], output_d = d[1], i, j, k = 0;

    /* GW = x * Gy^T    | (d0,d1) = (d0,1) * (d1,1)^T */
    for (i = 0; i < input_d; i++)
        for (j = 0; j < output_d; j++)
            GW[k++] = x[i] * Gy[j];

    /* Gx = W * Gy      | (d0,1) = (d0,d1) * (d1,1) */
    if (__builtin_expect(!!calc_x, 1)) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
            input_d, output_d, 1.0, W, input_d,
            Gy, 1, 0.0, Gx, 1);
    }
}
