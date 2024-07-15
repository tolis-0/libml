#include <stdio.h>
#include "../test.h"
#include "../../include/nn.h"


#define logistic_backward_test(name, N, Y, Gy, ...) \
    do {                                            \
        value_t Gx[N] = __nan_array(N);             \
        const value_t exp_Gx[N] = __VA_ARGS__;      \
        logistic_backward(N, Y, Gy, Gx);            \
        __exp_check(name, N, Gx, exp_Gx, 1e-10);    \
    } while (0)


int main ()
{
    __title("activations/logistic_backward");

    logistic_backward_test("positive Gy", 4,
        ((value_t[])    {0.668187772168, 0.908877038985, 0.1418510649,   0.5}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
                        {0.022171287329, 0.016563913398, 0.036518802086, 0.1});

    logistic_backward_test("negative Gy", 4,
        ((value_t[])    {0.2, 0.4, 0.6, 0.8}),
        ((grad_t[])     {-0.1, -0.2, -0.3, -0.4}),
                        {-0.016, -0.048, -0.072, -0.064});

    logistic_backward_test("small Y", 4,
        ((value_t[])    {0.01, 0.001, 0.00017, 0.0000096}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
                        {0.00099, 0.0001998, 0.00005099133, 0.000003839963136});

    logistic_backward_test("big Y", 4,
        ((value_t[])    {0.99, 0.999, 0.99983, 0.9999904}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
                        {0.00099, 0.0001998, 0.00005099133, 0.000003839963136});

    logistic_backward_test("zero Gy", 6,
        ((value_t[])    {0.99998475, 0.3512384, 1.0, 0.0, 0.0001675, 0.5}),
        ((grad_t[])     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
                        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0});

    logistic_backward_test("zero Y", 6,
        ((value_t[])    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
        ((grad_t[])     {0.1, 0.2, 0.3, -0.3, -0.2, -0.1}),
                        {0.0, 0.0, 0.0,  0.0,  0.0,  0.0});

    logistic_backward_test("one Y", 6,
        ((value_t[])    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
        ((grad_t[])     {0.1, 0.2, 0.3, -0.3, -0.2, -0.1}),
                        {0.0, 0.0, 0.0,  0.0,  0.0,  0.0});
}
