#include <math.h>
#include "../../../include/nn.h"

#define FP_ONE  __ml_fpc(1.0)

/*
 * Hyperbolic Tangent
 */
void tanh_forward(int d, cvrp_t x, vrp_t y)
{
    for (int i = 0; i < d; i++)
        y[i] = _tanh(x[i]);
}

void tanh_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = (FP_ONE - y[i] * y[i]) * g_y[i];
}
