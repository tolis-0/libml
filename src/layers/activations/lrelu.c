#include "../../../include/nn.h"

#define FP_ZERO __ml_fpc(0.0)
#define FP_A    __ml_fpc(0.01)

/*
 * Leaky ReLU with param a = 0.01
 */
void lrelu_forward(int d, cvrp_t x, vrp_t y)
{
    for (int i = 0; i < d; i++)
        y[i] = x[i] > FP_ZERO ? x[i] : FP_A * x[i];
}


void lrelu_backward(int d, cvrp_t x, cgrp_t g_y, grp_t g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = x[i] > FP_ZERO ? g_y[i] : FP_A * g_y[i];
}
