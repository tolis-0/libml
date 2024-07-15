#include <stdio.h>
#include "../test.h"
#include "../../include/nn.h"


#define dense_forward_test(name, N, M, X, W, opt, B, ...)   \
    do {                                                    \
        value_t y[M] = __nan_array(M);                      \
        const dim_t d = {(N), (M)};                         \
        const value_t exp_y[M] = __VA_ARGS__;               \
        dense_forward(d, X, W, opt, B, y);                  \
        __exp_check(name, M, y, exp_y, 1e-9);               \
    } while (0)


int main ()
{
    __title("layers/dense_forward");

    dense_forward_test("2x3 without bias", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    0,  NULL,
                        {0.23, 0.53, 0.83});

    dense_forward_test("2x3 with bias", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    1,  ((weight_t[])   {2.0, 3.0, 4.0}),
                        {2.23, 3.53, 4.83});

    dense_forward_test("1x1 without bias", 1, 1,
        ((value_t[])    {2.0}),
        ((weight_t[])   {3.0}),
    0,  NULL,
                        {6.0});

    dense_forward_test("4x4 without bias", 4, 4,
        ((value_t[])    {1.0, 2.0, 3.0, 4.0}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4,
                         0.5, 0.6, 0.7, 0.8,
                         0.9, 1.0, 1.1, 1.2,
                         1.3, 1.4, 1.5, 1.6}),
    0,  NULL,
                        {3.0, 7.0, 11.0, 15.0});

    dense_forward_test("3x2 random values", 3, 2,
        ((value_t[])    {1.0, -1.0, 2.0}),
        ((weight_t[])   {-0.5, 0.3, 0.1, 0.4, -0.2, -0.7}),
    1,  ((weight_t[])   {-1.0, 2.0}),
                        {-1.6, 1.2});

    dense_forward_test("2x3 with bias and zero weights", 2, 3,
        ((value_t[])    {1.0, 1.0}),
        ((weight_t[])   {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
    1,  ((weight_t[])   {4.7, -1.8, 0.0}),
                        {4.7, -1.8, 0.0});

    return 0;
}
