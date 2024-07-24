#include <cblas.h>
#include "../include/nn.h"


/*  Implements grad = y - t */
void loss_diff_grad(int d, cvrp_t y, cvrp_t t, vrp_t grad)
{
    cblas_copy(d, y, 1, grad, 1);
    cblas_axpy(d, -1.0, t, 1, grad, 1);
}


/*  Calculates mean squared error */
value_t loss_mse(int n, cvrp_t y, cvrp_t t)
{
    value_t diff, mse = 0.0;

    for (int i = 0; i < n; i++) {
        diff = y[i] - t[i];
        mse += diff * diff;
    }

    return mse / n;
}
