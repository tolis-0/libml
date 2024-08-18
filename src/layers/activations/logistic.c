#include <math.h>
#include "../../../include/nn.h"

#define FP_ONE  __ml_fpc(1.0)

/*
 * Logistic (sometimes referred to as just Sigmoid or Soft Step)
 */
void logistic_forward(int d, cvrp_t x, vrp_t y)
{
    for (int i = 0; i < d; i++)
        y[i] = FP_ONE / (FP_ONE + _exp(-x[i]));
}

void logistic_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = (y[i] * (FP_ONE - y[i])) * g_y[i];
}
