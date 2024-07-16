#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define relu_backward_test(name, N, X, GY, GX)  \
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
        relu_backward(N, x, Gy, Gx);            \
        __exp_check_lf(name, N, Gx, 1e-50);     \
    } while (0)


int main ()
{
    __title("activations/relu_backward");


    relu_backward_test("positive Gy", 10,
        ((value_t[])    {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((grad_t[])		{0.1, 0.2,  0.3, 0.4, 0.5,  0.6,  0.7, 0.8,  0.9,  1.0}),
        ((grad_t[])     {0.1, 0.2,  0.0, 0.0, 0.5,  0.0,  0.0, 0.8,  0.0,  0.0}));


    relu_backward_test("negative Gy", 10,
        ((value_t[])    { 0.7,  2.3, -1.8,  0.0,  0.1, -0.1, -2.7,  4.0, -3.0, -0.0}),
        ((grad_t[])     {-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9, -1.0}),
        ((grad_t[])     {-0.1, -0.2,  0.0,  0.0, -0.5,  0.0,  0.0, -0.8,  0.0,  0.0}));


    relu_backward_test("zero Gy", 10,
        ((value_t[])    {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((grad_t[])     {0.0, 0.0,  0.0, 0.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0}),
        ((grad_t[])     {0.0, 0.0,  0.0, 0.0, 0.0,  0.0,  0.0, 0.0,  0.0,  0.0}));


    relu_backward_test("positive x", 10,
        ((value_t[])    {1.0, 2.0,  3.0, 4.0, 5.0,  6.0,  7.0, 8.0,  9.0, 10.0}),
        ((grad_t[])     {0.1, 0.2,  0.3, 0.4, 0.5,  0.6,  0.7, 0.8,  0.9,  0.0}),
        ((grad_t[])     {0.1, 0.2,  0.3, 0.4, 0.5,  0.6,  0.7, 0.8,  0.9,  0.0}));


    relu_backward_test("negative x", 10,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0}),
        ((grad_t[])     { 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,   0.0}),
        ((grad_t[])     { 0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,   0.0}));


    relu_backward_test("zero x", 10,
        ((value_t[])    { 0.0, 0.0,  0.0,  0.0, 0.0,  0.0, 0.0,  0.0,  0.0, 0.0}),
        ((grad_t[])     {-0.1, 0.2, -0.3, -0.4, 0.5, -0.6, 0.7, -0.8, -0.9, 0.0}),
        ((grad_t[])     { 0.0, 0.0,  0.0,  0.0, 0.0,  0.0, 0.0,  0.0,  0.0, 0.0}));


    relu_backward_test("random example", 10,
        ((value_t[])    {1.0, -2.0, 3.0, -4.0,  5.0, 1e39, -1e39, 1e-29, -1e-29, 0.0}),
        ((grad_t[])     {0.1,  0.2, 0.3,  0.4, -0.5,  0.6,   0.7,   0.8,    0.9, 0.0}),
        ((grad_t[])     {0.1,  0.0, 0.3,  0.0, -0.5,  0.6,   0.0,   0.8,    0.0, 0.0}));


    return 0;
}
