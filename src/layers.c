#include <cblas.h>
#include "../include/nn.h"


/*  Computes the result y of a single forward pass in the dense layer */
void dense_forward(const dim_t d, const value_t *x, const weight_t *W,
    int has_bias, const weight_t *b, value_t *y)
{
    int d0 = d[0], d1 = d[1]; // d0 is input dimension and d1 is output

    /* y = W^T * x      | (d1,1) = (d0,d1)^T * (d0,1) */
    cblas_dgemv(CblasColMajor, CblasTrans,
        d0, d1, 1.0, W, d0, x, 1, 0.0, y, 1);

    /* y = b + y        | (d1,1) = (d1,1) + (d1,1) */
    if (has_bias) {
        cblas_daxpy(d1, 1.0, b, 1, y, 1);
    }
}


/*  Computes the gradients of a single backward pass where the output
    and gradient of the output of the layer is given. Gb is always Gy */
void dense_backward(const dim_t d, const value_t *x, const weight_t *W,
    int calc_x, const grad_t *Gy, grad_t *Gx, grad_t *GW)
{
    int d0 = d[0], d1 = d[1], i, j, k = 0;

    /* GW = x * Gy^T    | (d0,d1) = (d0,1) * (d1,1)^T */
    for (j = 0; j < d1; j++)
        for (i = 0; i < d0; i++)
            GW[k++] = x[i] * Gy[j];

    /* Gx = W * Gy      | (d0,1) = (d0,d1) * (d1,1) */
    if (calc_x) {
        cblas_dgemv(CblasColMajor, CblasNoTrans,
            d0, d1, 1.0, W, d0, Gy, 1, 0.0, Gx, 1);
    }
}


/*  Computes the result Y of a batch forward pass in the dense layer */
void batch_dense_forward(const dim3_t d, const value_t *x, const weight_t *W,
    int has_bias, const weight_t *b, const value_t *ones, value_t *y)
{
    int d0 = d[0], d1 = d[1], k = d[2]; // k is batch size

    /* Y = W^T * X      | (d1,k) = (d0,d1)^T * (d0,k) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
        d1, k, d0, 1.0, W, d0, x, d0, 0.0, y, d1);

    /* Y += b * [1]^T   | (d1,k) += (d1,1) * (k,1)^T */
    if (has_bias) {
        cblas_dger(CblasColMajor, d1, k, 1.0, b, 1, ones, 1, y, d1);
    }
}


void batch_dense_backward(const dim3_t d, const value_t *x, const weight_t *W,
    int calc_x, const value_t *ones, const grad_t *Gy, grad_t *Gx, grad_t *GW,
    int has_bias, grad_t *Gb)
{
    int d0 = d[0], d1 = d[1], k = d[2];

    /* GW = 1/k * X * GY^T  | (d0,d1) = (d0,k) * (d1,k)^T */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
        d0, d1, k, 1.0 / k, x, d0, Gy, d1, 0.0, GW, d0);

    /* GX = W * GY          | (d0,k) = (d0,d1) * (d1,k) */
    if (calc_x) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            d0, k, d1, 1.0, W, d0, Gy, d1, 0.0, Gx, d0);
    }

    /* GB = 1/k * GY * [1]  | (d1,1) = (d1,k) * (k,1) */
    if (has_bias) {
        cblas_dgemv(CblasColMajor, CblasNoTrans,
            d1, k, 1.0 / k, Gy, d1, ones, 1, 0.0, Gb, 1);
    }
}
