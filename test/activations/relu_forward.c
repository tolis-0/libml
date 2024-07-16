#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define relu_forward_test(name, N, X, Y)        \
    do {                                        \
        assert((N) > 0);                        \
        assert(__arr_count(X) == (N));          \
        assert(__arr_count(Y) == (N));          \
                                                \
        value_t y[N] = __nan_array(N);          \
                                                \
        const value_t *x = X;                   \
        const value_t *exp_y = Y;               \
                                                \
        relu_forward(N, x, y);                  \
        __exp_check_lf(name, N, y, 1e-50);      \
    } while (0)


int main ()
{
    __title("activations/relu_forward");


    relu_forward_test("random example", 10,
        ((value_t[])    {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((value_t[])    {0.7, 2.3,  0.0, 0.0, 0.1,  0.0,  0.0, 4.0,  0.0,  0.0}));


    relu_forward_test("positive values", 6,
        ((value_t[])    {1.0, 2.0, 3.0, 4.0, 5.0, 10.0}),
        ((value_t[])    {1.0, 2.0, 3.0, 4.0, 5.0, 10.0}));


    relu_forward_test("negative values", 6,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0, -5.0, -10.0}),
        ((value_t[])    {0.0,   0.0,  0.0,  0.0,  0.0,   0.0}));


    relu_forward_test("mixed values", 6,
        ((value_t[])    {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}),
        ((value_t[])    {0.0,  2.0,  0.0, 4.0,  0.0, 6.0}));


    relu_forward_test("zero values", 4,
        ((value_t[])    {0.0, 0.0, 0.0, 0.0}),
        ((value_t[])    {0.0, 0.0, 0.0, 0.0}));


    relu_forward_test("large values", 8,
        ((value_t[])    {2e5, -2e5, 7e10, -7e10, 3e20, -3e20, 5e40, -5e40}),
        ((value_t[])    {2e5,  0.0, 7e10,   0.0, 3e20,   0.0, 5e40,   0.0}));


    relu_forward_test("small values", 8,
        ((value_t[])    {1e-6, -1e-6, 2e-9, -2e-9, 3e-19, -3e-19, 4e-39, -4e-39}),
        ((value_t[])    {1e-6,   0.0, 2e-9,   0.0, 3e-19,    0.0, 4e-39,    0.0}));


    return 0;
}
