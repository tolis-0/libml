#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define lrelu_backward_test(name, N, X, GY, GX) \
    do {                                        \
        assert((N) > 0);                        \
        assert(__arr_count(X) == (N));          \
        assert(__arr_count(GY) == (N));         \
        assert(__arr_count(GX) == (N));         \
                                                \
        value_t Gx[N] = __nan_array(N);         \
                                                \
        const value_t *x = X;                   \
        const grad_t *Gy = GY;                  \
        const grad_t *exp_Gx = GX;              \
                                                \
        lrelu_backward(N, x, Gy, Gx);           \
        __exp_check_lf(name, N, Gx, 1e-8);      \
    } while (0)


int main ()
{
    __title("activations/lrelu_backward");


    lrelu_backward_test("positive Gy", 10,
        ((value_t[])    {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((grad_t[])     {0.1, 0.2,  0.3, 0.4, 0.5,  0.6,  0.7, 0.8,  0.9,  1.0}),
        ((grad_t[])     {0.1, 0.2, 3e-3,4e-3, 0.5, 6e-3, 7e-3, 0.8, 9e-3, 0.01}));


    lrelu_backward_test("negative Gy", 10,
        ((value_t[])    { 0.7,  2.3, -1.8,  0.0,  0.1, -0.1, -2.7,  4.0, -3.0, -0.0}),
        ((grad_t[])     {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0}),
        ((grad_t[])     {-0.1, -0.2,-3e-3,-4e-3, -0.5,-6e-3,-7e-3, -0.8,-9e-3,-0.01}));


    lrelu_backward_test("zero Gy", 10,
        ((value_t[])    {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((grad_t[])     {0.0, 0.0,  0.0, 0.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0}),
        ((grad_t[])     {0.0, 0.0,  0.0, 0.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0}));


    lrelu_backward_test("positive x", 10,
        ((value_t[])    {1.0, 2.0,  3.0, 4.0, 5.0,  6.0,  7.0, 8.0,  9.0, 10.0}),
        ((grad_t[])     {0.1, 0.2,  0.3, 0.4, 0.5,  0.6,  0.7, 0.8,  0.9,  0.0}),
        ((grad_t[])     {0.1, 0.2,  0.3, 0.4, 0.5,  0.6,  0.7, 0.8,  0.9,  0.0}));


    lrelu_backward_test("negative x", 10,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0}),
        ((grad_t[])     { 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,   0.0}),
        ((grad_t[])     {1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,   0.0}));


    lrelu_backward_test("zero x", 10,
        ((value_t[])    { 0.0, 0.0,  0.0,  0.0, 0.0,  0.0, 0.0,  0.0,  0.0, 0.0}),
        ((grad_t[])     {-0.1, 0.2, -0.3, -0.4, 0.5, -0.6, 0.7, -0.8, -0.9, 0.0}),
        ((grad_t[])     {-1e-3,2e-3,-3e-3,-4e-3,5e-3,-6e-3,7e-3,-8e-3,-9e-3,0.0}));


    lrelu_backward_test("random example", 10,
        ((value_t[])    {1.0, -2.0, 3.0, -4.0,  5.0, 1e9, -1e9, 1e-19, -1e-19, 0.0}),
        ((grad_t[])     {0.1,  0.2, 0.3,  0.4, -0.5, 0.6,  0.7,   0.8,    0.9, 0.0}),
        ((grad_t[])     {0.1, 2e-3, 0.3, 4e-3, -0.5, 0.6, 7e-3,   0.8,   9e-3, 0.0}));


    return 0;
}
