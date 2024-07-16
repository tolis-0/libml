#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define batch_dense_forward_test(name, N, M, K, X, W, opt, B, Y)    \
    do {                                                            \
        assert((N) > 0);                                            \
        assert((M) > 0);                                            \
        assert((K) > 0);                                            \
        assert(__arr_count(X) == (N)*(K));                          \
        assert(__arr_count(W) == (N)*(M));                          \
        assert(__arr_count_null(B) == (M) || !(opt));               \
        assert(__arr_count(Y) == (M)*(K));                          \
                                                                    \
        value_t y[(M)*(K)] = __nan_array((M)*(K));                  \
        const value_t ones[K] = __val_array(K, 1.0);                \
                                                                    \
        const dim3_t d = {(N), (M), (K)};                           \
        const value_t *x = X;                                       \
        const weight_t *w = W;                                      \
        const weight_t *b = B;                                      \
        const value_t *exp_y = Y;                                   \
                                                                    \
        batch_dense_forward(d, x, w, opt, b, ones, y);              \
        __exp_check_lf(name, (M)*(K), y, 1e-9);                     \
    } while (0)


int main ()
{
    __title("layers/batch_dense_forward");


    batch_dense_forward_test("2x3 k=2 w/o bias", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    0,  NULL,
        ((value_t[])    {0.23, 0.53, 0.83, 0.41, 0.95, 1.49}));


    batch_dense_forward_test("2x3 k=2 w/ bias", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    1,  ((weight_t[])   {2.0, 3.0, 4.0}),
        ((value_t[])    {2.23, 3.53, 4.83, 2.41, 3.95, 5.49}));


    batch_dense_forward_test("1x1 k=1 w/o bias", 1, 1, 1,
        ((value_t[])    {1.0}),
        ((weight_t[])   {2.0}),
    0,  NULL,
        ((value_t[])    {2.0}));


    batch_dense_forward_test("3x2 k=4 w/o bias", 3, 2, 4,
        ((value_t[])    {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    0,  NULL,
        ((value_t[])    {0.14, 0.32, 0.32, 0.77, 0.5, 1.22, 0.68, 1.67}));


    batch_dense_forward_test("2x2 k=3 w/ bias", 2, 2, 3,
        ((value_t[])    {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4}),
    1,  ((weight_t[])   {1.0, 2.0}),
        ((value_t[])    {1.5, 3.1, 2.1, 4.5, 2.7, 5.9}));


    batch_dense_forward_test("2x2 k=2 zeros w/o bias", 2, 2, 2,
        ((value_t[])    {0.0, 0.0, 0.0, 0.0}),
        ((weight_t[])   {0.0, 0.0, 0.0, 0.0}),
    0,  NULL,
        ((value_t[])    {0.0, 0.0, 0.0, 0.0}));


    batch_dense_forward_test("2x2 k=2 negative w/o bias", 2, 2, 2,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0}),
        ((weight_t[])   {-0.1, -0.2, -0.3, -0.4}),
    0,  NULL,
        ((value_t[])    {0.5, 1.1, 1.1, 2.5}));


    batch_dense_forward_test("3x2 k=4 random values w/ bias", 3, 2, 4,
        ((value_t[])    {0.5, -0.2, 0.3, 0.0, -0.5, 1.2, -1.0, 0.8, 0.7, 1.5, -1.5, -0.3}),
        ((weight_t[])   {0.4, -0.1, 0.2, -0.3, 0.6, 0.5}),
    1,  ((weight_t[])   {0.1, -0.2}),
        ((value_t[])    {0.38, -0.32, 0.39, 0.1, -0.24, 0.93, 0.79, -1.7}));


    return 0;
}
