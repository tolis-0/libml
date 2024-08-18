#include <cblas.h>
#include "../../include/nn.h"


/*  Computes the result y of a single forward pass in the dense layer */
void dense_forward(cdim_t d, cvrp_t x, cwrp_t w,
    _Bool hb, cwrp_t b, vrp_t y)
{
    const int d0 = d[0], d1 = d[1]; // d0 is input dimension and d1 is output

    /* y = W^T * x      | (d1,1) = (d0,d1)^T * (d0,1) */
    cblas_gemv(CblasColMajor, CblasTrans,
        d0, d1, 1.0, w, d0, x, 1, 0.0, y, 1);

    /* y = b + y        | (d1,1) = (d1,1) + (d1,1) */
    if (hb) {
        cblas_axpy(d1, 1.0, b, 1, y, 1);
    }
}


/*  Computes the gradients of a single backward pass where the output
    and gradient of the output of the layer is given. Gb is always Gy */
void dense_backward(cdim_t d, cvrp_t x, cwrp_t w,
    _Bool cx, cgrp_t Gy, grp_t Gx, grp_t Gw)
{
    const int d0 = d[0], d1 = d[1];
    int i, j, k = 0;

    /* Gw = x * Gy^T    | (d0,d1) = (d0,1) * (d1,1)^T */
    for (j = 0; j < d1; j++)
        for (i = 0; i < d0; i++)
            Gw[k++] = x[i] * Gy[j];

    /* Gx = W * Gy      | (d0,1) = (d0,d1) * (d1,1) */
    if (cx) {
        cblas_gemv(CblasColMajor, CblasNoTrans,
            d0, d1, 1.0, w, d0, Gy, 1, 0.0, Gx, 1);
    }
}


/*  Computes the result y of a batch forward pass in the dense layer */
void batch_dense_forward(cdim3_t d, cvrp_t x, cwrp_t w,
    _Bool hb, cwrp_t b, cvrp_t ones, vrp_t y)
{
    const int d0 = d[0], d1 = d[1], k = d[2]; // k is batch size

    /* y = W^T * x      | (d1,k) = (d0,d1)^T * (d0,k) */
    cblas_gemm(CblasColMajor, CblasTrans, CblasNoTrans,
        d1, k, d0, 1.0, w, d0, x, d0, 0.0, y, d1);

    /* y += b * [1]^T   | (d1,k) += (d1,1) * (k,1)^T */
    if (hb) {
        cblas_ger(CblasColMajor, d1, k, 1.0, b, 1, ones, 1, y, d1);
    }
}


void batch_dense_backward(cdim3_t d, cvrp_t x, cwrp_t w,
    _Bool cx, cvrp_t ones, cgrp_t Gy, grp_t Gx, grp_t Gw, _Bool hb, grp_t Gb)
{
    const int d0 = d[0], d1 = d[1], k = d[2];

    /* Gw = 1/k * x * Gy^T  | (d0,d1) = (d0,k) * (d1,k)^T */
    cblas_gemm(CblasColMajor, CblasNoTrans, CblasTrans,
        d0, d1, k, 1.0 / k, x, d0, Gy, d1, 0.0, Gw, d0);

    /* Gx = W * Gy          | (d0,k) = (d0,d1) * (d1,k) */
    if (cx) {
        cblas_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
            d0, k, d1, 1.0, w, d0, Gy, d1, 0.0, Gx, d0);
    }

    /* Gb = 1/k * Gy * [1]  | (d1,1) = (d1,k) * (k,1) */
    if (hb) {
        cblas_gemv(CblasColMajor, CblasNoTrans,
            d1, k, 1.0 / k, Gy, d1, ones, 1, 0.0, Gb, 1);
    }
}
