#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define dense_backward_test(name, N, M, X, W, GY, opt, GX, GW)  \
    do {                                                        \
        assert((N) > 0);                                        \
        assert((M) > 0);                                        \
        assert(__arr_count(X) == (N));                          \
        assert(__arr_count(W) == (N)*(M));                      \
        assert(__arr_count(GY) == (M));                         \
        assert(__arr_count_null(GX) == (N) || !(opt));          \
        assert(__arr_count(GW) == (N)*(M));                     \
                                                                \
        grad_t Gx[N] = __nan_array(N);                          \
        grad_t Gw[(M)*(N)] = __nan_array((M)*(N));              \
                                                                \
        const dim_t d = {(N), (M)};                             \
        const value_t *x = X;                                   \
        const weight_t *w = W;                                  \
        const grad_t *Gy = GY;                                  \
        const grad_t *exp_Gx = GX;                              \
        const grad_t *exp_Gw = GW;                              \
                                                                \
        dense_backward(d, x, w, opt, Gy, Gx, Gw);               \
        if (opt) __exp_check_lf(name " (Gx)", N, Gx, 1e-6);     \
        __exp_check_lf(name " (Gw)", (N)*(M), Gw, 1e-6);        \
    } while (0)


int main ()
{
    __title("layers/dense_backward");


    dense_backward_test("2x3 calcx", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65}),
    1,  ((grad_t[])     {0.535, 0.7}),
        ((grad_t[])     {0.315, 0.36, 0.385, 0.44, 0.455, 0.52}));


    dense_backward_test("2x3 no calcx", 2, 3,
        ((value_t[])    {0.7, 0.8}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.45, 0.55, 0.65}),
    0,  NULL,
        ((grad_t[])     {0.315, 0.36, 0.385, 0.44, 0.455, 0.52}));


    dense_backward_test("3x2 calcx", 3, 2,
        ((value_t[])    {0.7, 0.8, 0.9}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
        ((grad_t[])     {0.35, 0.75}),
    1,  ((grad_t[])     {0.335, 0.445, 0.555}),
        ((grad_t[])     {0.245, 0.28, 0.315, 0.525, 0.6, 0.675}));


    return 0;
}
