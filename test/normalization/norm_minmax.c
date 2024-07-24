#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/normalization.h"


#define norm_minmax_test(name, N, M, D, normD)  \
    do {                                        \
        const int size = (N)*(M);               \
        assert((N) > 0);                        \
        assert((M) > 0);                        \
        assert(__arr_count(D) == size);         \
        assert(__arr_count(normD) == size);     \
                                                \
        value_t *const d = D;                   \
        const value_t *const exp_d = normD;     \
                                                \
        norm_minmax(d, (N), (M));               \
        __exp_check_lf(name, size, d, 1e-6);    \
    } while (0)


int main ()
{
    __title("activations/relu_forward");


    norm_minmax_test("5x4 example", 5, 4,
        ((value_t[])    {-7.4,      1.3,       0.48,     -2.467,
                          6.1,     -0.1,       0.71,     -2.000,
                         -3.8,     -1.7,       1.00,     -3.000,
                          2.3,      0.8,       0.00,     -2.384,
                          8.7,     -0.6,       0.25,     -3.000}),
        ((value_t[])    {0.0,       1.0,       0.48,      0.533,
                         0.8385093, 0.5333334, 0.71,      1.0,
                         0.2236025, 0.0,       1.00,      0.0,
                         0.6024845, 0.8333333, 0.00,      0.616,
                         1.0,       0.3666667, 0.25,      0.0}));


    norm_minmax_test("2x3 simple example", 2, 3,
        ((value_t[])    {1.0, 4.0, 5.0,
                         3.0, 2.0, 5.0}),
        ((value_t[])    {0.0, 1.0, 0.5,
                         1.0, 0.0, 0.5}));


    norm_minmax_test("1x4 negative values", 1, 4,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0}),
        ((value_t[])    {0.5,   0.5,  0.5,  0.5}));


    norm_minmax_test("4x1 negative values", 4, 1,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0}),
        ((value_t[])    {1.0, 0.6666667, 0.3333333, 0.0}));


    norm_minmax_test("3x3 identical values", 3, 3,
        ((value_t[])    {5.0, 5.0, 5.0,
                         5.0, 5.0, 5.0,
                         5.0, 5.0, 5.0}),
        ((value_t[])    {0.5, 0.5, 0.5,
                         0.5, 0.5, 0.5,
                         0.5, 0.5, 0.5}));


    norm_minmax_test("4x4 large range", 4, 4,
        ((value_t[])    {100.0,     200.0,     300.0,     400.0,
                         500.0,     600.0,     700.0,     800.0,
                         900.0,     1000.0,    1100.0,    1200.0,
                         1300.0,    1400.0,    1500.0,    1600.0}),
        ((value_t[])    {0.0,       0.0,       0.0,       0.0,
                         0.3333333, 0.3333333, 0.3333333, 0.3333333,
                         0.6666667, 0.6666667, 0.6666667, 0.6666667,
                         1.0,       1.0,       1.0,       1.0}));


    return 0;
}
