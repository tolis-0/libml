#include <stdio.h>
#include "../test.h"
#include "../../include/nn.h"


#define dense_backward_test(name, N, M, X, W, Gy, opt, ...)     \
    do {                                                        \
        value_t Gx[N] = __nan_array(N);                         \
        value_t GW[(M)*(N)] = __nan_array((M)*(N));             \
        const dim_t d = {(N), (M)};                             \
        const value_t exp_Gx[(N) + (N)*(M)] = __VA_ARGS__;      \
        const weight_t *exp_GW = exp_Gx + (N)*!!(opt);          \
        dense_backward(d, X, W, opt, Gy,                        \
            ((opt) ? Gx : NULL), GW);                           \
        if (opt) __exp_check_lf(name " (Gx)", N, Gx, 1e-9);     \
        __exp_check_lf(name " (GW)", (N)*(M), GW, 1e-9);        \
    } while (0)


int main ()
{
    __title("layers/dense_backward");

        dense_backward_test("2x3 calcx", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65}),
    1,                  {0.535, 0.7,
                         0.315, 0.36, 0.385, 0.44, 0.455, 0.52});

        dense_backward_test("2x3 no calcx", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65}),
    0,                  {0.315, 0.36, 0.385, 0.44, 0.455, 0.52});

        dense_backward_test("3x2 calcx", 3, 2,
        ((value_t[])    {0.7, 0.8, 0.9}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.35, 0.75}),
    1,                  {0.335, 0.445, 0.555,
                         0.245, 0.28, 0.315, 0.525, 0.6, 0.675});

    return 0;
}
