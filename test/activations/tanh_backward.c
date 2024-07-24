#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "../test.h"
#include "../../include/nn.h"


#define tanh_backward_test(name, N, Y, GY, GX)  \
    do {                                        \
        assert((N) > 0);                        \
        assert(__arr_count(Y) == (N));          \
        assert(__arr_count(GY) == (N));         \
        assert(__arr_count(GX) == (N));         \
                                                \
        value_t Gx[N] = __nan_array(N);         \
                                                \
        const value_t *y = Y;                   \
        const grad_t *Gy = GY;                  \
        const grad_t *exp_Gx = GX;              \
                                                \
        tanh_backward(N, y, Gy, Gx);            \
        __exp_check_lf(name, N, Gx, 1e-6);      \
    } while (0)


int main()
{
    __title("activations/tanh_backward");


    tanh_backward_test("positive Gy", 4,
        ((value_t[])    {0.60436778, 0.98009640, -0.94680601, 0.0}),
        ((grad_t[])     {0.1,        0.2,         0.3,        0.4}),
        ((grad_t[])     {0.06347395, 0.00788220,  0.03106751, 0.4}));


    tanh_backward_test("negative Gy", 4,
        ((value_t[])    {0.2, 0.4, 0.6, 0.8}),
        ((grad_t[])     {-0.1, -0.2, -0.3, -0.4}),
        ((grad_t[])     {-0.096, -0.168, -0.192, -0.144}));


    tanh_backward_test("small Y", 4,
        ((value_t[])    {0.01, 0.001, 0.00017, 0.0000096}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
        ((grad_t[])     {0.09999, 0.19999999, 0.299999999, 0.39999999996}));


    tanh_backward_test("big Y", 4,
        ((value_t[])    {0.99, 0.999, 0.99983, 0.9999904}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
        ((grad_t[])     {0.00199, 0.0003998, 0.00010199133, 0.000007679963}));


    tanh_backward_test("zero Gy", 6,
        ((value_t[])    {0.99998475, 0.3512384, 1.0, 0.0, 0.0001675, 0.5}),
        ((grad_t[])     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
        ((grad_t[])     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));


    tanh_backward_test("zero Y", 6,
        ((value_t[])    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
        ((grad_t[])     {0.1, 0.2, 0.3, -0.3, -0.2, -0.1}),
        ((grad_t[])     {0.1, 0.2, 0.3, -0.3, -0.2, -0.1}));


    tanh_backward_test("one Y", 6,
        ((value_t[])    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
        ((grad_t[])     {0.1, 0.2, 0.3, -0.3, -0.2, -0.1}),
        ((grad_t[])     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));


    return 0;
}
