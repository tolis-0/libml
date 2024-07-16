#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define batch_dense_backward_test(name, N, M, K, X, W, GY,          \
                                  opt1, GX, GW, opt2, GB)           \
    do {                                                            \
        assert((N) > 0);                                            \
        assert((M) > 0);                                            \
        assert((K) > 0);                                            \
        assert(__arr_count(X) == (N)*(K));                          \
        assert(__arr_count(W) == (N)*(M));                          \
        assert(__arr_count(GY) == (M)*(K));                         \
        assert(__arr_count_null(GX) == (N)*(K) || !(opt1));         \
        assert(__arr_count(GW) == (N)*(M));                         \
        assert(__arr_count_null(GB) == (M) || !(opt2));             \
                                                                    \
        value_t Gx[(N)*(K)] = __nan_array((N)*(K));                 \
        value_t Gw[(M)*(N)] = __nan_array((M)*(N));                 \
        value_t Gb[M] = __nan_array(M);                             \
        const value_t ones[K] = __val_array(K, 1.0);                \
                                                                    \
        const dim3_t d = {(N), (M), (K)};                           \
        const value_t *x = X;                                       \
        const weight_t *w = W;                                      \
        const grad_t *Gy = GY;                                      \
        const grad_t *exp_Gx = GX;                                  \
        const grad_t *exp_Gw = GW;                                  \
        const grad_t *exp_Gb = GB;                                  \
                                                                    \
        batch_dense_backward(d, x, w, opt1, ones,                   \
                             Gy, Gx, Gw, opt2, Gb);                 \
        if (opt1) __exp_check_lf(name " (Gx)", N, Gx, 1e-9);        \
        if (opt2) __exp_check_lf(name " (Gb)", N, Gb, 1e-9);        \
        __exp_check_lf(name " (Gw)", (N)*(M), Gw, 1e-9);            \
    } while (0)


int main ()
{
    __title("layers/batch_dense_backward");


    batch_dense_backward_test("2x3 k=2 w/ bias", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65, 0.24, 0.44, 0.64}),
    1,  ((grad_t[])     {0.535, 0.7, 0.476, 0.608}),
        ((grad_t[])     {0.3135, 0.348, 0.4785, 0.528, 0.6435, 0.708}),
    1,  ((grad_t[])     {0.345, 0.495, 0.645}));


    batch_dense_backward_test("2x3 k=2 w/ bias no calcx", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65, 0.24, 0.44, 0.64}),
    0,  NULL,
        ((grad_t[])     {0.3135, 0.348, 0.4785, 0.528, 0.6435, 0.708}),
    0,  NULL);


    return 0;
}
