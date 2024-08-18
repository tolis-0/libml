#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define lrelu_forward_test(name, N, X, Y)       \
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
        lrelu_forward(N, x, y);                 \
        __exp_check_lf(name, N, y, 1e-7);       \
    } while (0)


int main ()
{
    __title("activations/lrelu_forward");


    lrelu_forward_test("random example", 10,
        ((value_t[])    {0.7,2.3,-1.8  ,0.0,0.1,-0.1, -2.7,  4.0,-3.0, -0.0}),
        ((value_t[])    {0.7,2.3,-0.018,0.0,0.1,-1e-3,-0.027,4.0,-0.03, 0.0}));


    lrelu_forward_test("positive values", 6,
        ((value_t[])    {1.0, 2.0, 3.0, 4.0, 5.0, 10.0}),
        ((value_t[])    {1.0, 2.0, 3.0, 4.0, 5.0, 10.0}));


    lrelu_forward_test("negative values", 6,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0, -5.0, -10.0}),
        ((value_t[])    {-0.01,-0.02,-0.03,-0.04,-0.05,-0.1}));


    lrelu_forward_test("mixed values", 6,
        ((value_t[])    {-1.0, 2.0, -3.0, 4.0, -5.0, 6.0}),
        ((value_t[])    {-0.01,2.0,-0.03, 4.0,-0.05, 6.0}));


    lrelu_forward_test("zero values", 4,
        ((value_t[])    {0.0, 0.0, 0.0, 0.0}),
        ((value_t[])    {0.0, 0.0, 0.0, 0.0}));


    lrelu_forward_test("large values", 7,
        ((value_t[])    {2e5, -2e5, 7e10, -7e10, 5e15, -4e15, 3e20}),
        ((value_t[])    {2e5, -2e3, 7e10, -7e8,  5e15, -4e13, 3e20}));


    lrelu_forward_test("small values", 8,
        ((value_t[])    {1e-6, -1e-6, 2e-9, -2e-9, 6e-15, -4e-15, 3e-19, -3e-19}),
        ((value_t[])    {1e-6, -1e-8, 2e-9, -2e-11,6e-15, -4e-17, 3e-19, -3e-21}));


    return 0;
}
