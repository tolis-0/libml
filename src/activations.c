#include <math.h> 
#include "../include/nn.h"


/*  ReLU (Rectified Linear Unit) activation function */
void relu_forward(int d, cvrp_t x, vrp_t y)
{
    for (int i = 0; i < d; i++)
        y[i] = x[i] > 0.0 ? x[i] : 0.0;
}

void relu_backward(int d, cvrp_t x, cgrp_t g_y, grp_t g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = x[i] > 0.0 ? g_y[i] : 0.0;
}


/*  Leaky ReLU activation function */
void lrelu_forward(int d, cvrp_t x, vrp_t y)
{
    for (int i = 0; i < d; i++)
        y[i] = x[i] > 0.0 ? x[i] : 0.01 * x[i];
}


void lrelu_backward(int d, cvrp_t x, cgrp_t g_y, grp_t g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = x[i] > 0.0 ? g_y[i] : 0.01 * g_y[i];
}


/*  Logistic activation function */
void logistic_forward(int d, cvrp_t x, vrp_t y)
{
    for (int i = 0; i < d; i++)
        y[i] = 1.0 / (1.0 + _exp(-x[i]));
}

void logistic_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = (y[i] * (1 - y[i])) * g_y[i];
}


/*  Hyperbolic Tangent activation function */
void tanh_forward(int d, cvrp_t x, vrp_t y)
{
    for (int i = 0; i < d; i++)
        y[i] = _tanh(x[i]);
}

void tanh_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x)
{
    for (int i = 0; i < d; i++)
        g_x[i] = (1 - y[i] * y[i]) * g_y[i];
}
