#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"

#define dense_forward_test(name, N, M, X, W, opt, B, ...)   \
    do {                                                    \
        value_t y[M] = {0};                                 \
        const dim_t d = {(N), (M)};                         \
        const value_t exp_y[M] = __VA_ARGS__;               \
        dense_forward(d, X, W, opt, B, y);                  \
        __exp_check(name, M, y, exp_y, 1e-9);               \
    } while (0)

int main ()
{
    __title("layers/dense_forward");

    dense_forward_test("test0", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    0,  NULL,
                        {0.23, 0.53, 0.83});

    dense_forward_test("test1", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    1,  ((weight_t[])   {2.0, 3.0, 4.0}),
                        {2.23, 3.53, 4.83});

    return 0;
}
