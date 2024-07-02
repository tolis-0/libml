#include <math.h> 
#include "nn.h"


void relu_forward(int d, const value_t *x, value_t *y)
{
    for (int i = 0; i < d; i++)
        y[i] = x[i] > 0.0 ? x[i] : 0.0;
}

void relu_backward(int d, const value_t *x, const grad_t *g_y, grad_t *g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = x[i] > 0.0 ? g_y[i] : 0.0;
}


void logistic_forward(int d, const value_t *x, value_t *y)
{
    for (int i = 0; i < d; i++)
        y[i] = 1.0 / (1.0 + exp(-x[i]));
}

void logistic_backward(int d, const value_t *y, const grad_t *g_y, grad_t *g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = (y[i] * (1 - y[i])) * g_y[i];
}
