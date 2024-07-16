#include <stdio.h>
#include "../test.h"
#include "../../include/nn.h"


#define batch_dense_backward_test(name, N, M, K, X, W, Gy, opt1, opt2, ...) \
    do {                                                                    \
        value_t Gx[(N)*(K)] = __nan_array((N)*(K));                         \
        value_t GW[(M)*(N)] = __nan_array((M)*(N));                         \
        value_t Gb[M] = __nan_array(M);                                     \
        const value_t ones[K] = __val_array(K, 1.0);                        \
        const dim3_t d = {(N), (M), (K)};                                   \
        const value_t exp_Gx[(N)*(K) + (N)*(M) + (M)] = __VA_ARGS__;        \
        const weight_t *exp_GW = exp_Gx + (N)*(K)*!!(opt1);                 \
        const weight_t *exp_Gb = exp_GW + (M)*(N);                          \
        batch_dense_backward(d, X, W, opt1, ones, Gy,                       \
            ((opt1) ? Gx : NULL), GW, opt2, ((opt2) ? Gb : NULL));          \
        if (opt1) __exp_check_lf(name " (Gx)", N, Gx, 1e-9);                \
        if (opt2) __exp_check_lf(name " (Gb)", N, Gb, 1e-9);                \
        __exp_check_lf(name " (GW)", (N)*(M), GW, 1e-9);                    \
    } while (0)


int main ()
{
    __title("layers/batch_dense_backward");

    batch_dense_backward_test("2x3 k=2 w/ bias calcx", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65, 0.24, 0.44, 0.64}),
    1,  1,              {0.535, 0.7, 0.476, 0.608,
                         0.3135, 0.348, 0.4785, 0.528, 0.6435, 0.708,
                         0.345, 0.495, 0.645});

    batch_dense_backward_test("2x3 k=2 w/o bias no calcx", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65, 0.24, 0.44, 0.64}),
    0,  0,              {0.3135, 0.348, 0.4785, 0.528, 0.6435, 0.708});

    return 0;
}
