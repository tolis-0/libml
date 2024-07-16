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


    batch_dense_forward_test("2x3 k=2 without bias", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    0,  NULL,
        ((value_t[])    {0.23, 0.53, 0.83, 0.41, 0.95, 1.49}));


    batch_dense_forward_test("2x3 k=2 with bias", 2, 3, 2,
        ((value_t[])    {0.7, 0.8, 1.3, 1.4}),
        ((weight_t[])   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6}),
    1,  ((weight_t[])   {2.0, 3.0, 4.0}),
        ((value_t[])    {2.23, 3.53, 4.83, 2.41, 3.95, 5.49}));


    return 0;
}
