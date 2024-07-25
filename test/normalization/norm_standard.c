#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/normalization.h"


#define norm_standard_test(name, N, M, D, normD)    \
    do {                                            \
        const int size = (N)*(M);                   \
        assert((N) > 0);                            \
        assert((M) > 0);                            \
        assert(__arr_count(D) == size);             \
        assert(__arr_count(normD) == size);         \
                                                    \
        value_t *const d = D;                       \
        const value_t *const exp_d = normD;         \
                                                    \
        norm_standard(d, (N), (M));                 \
        __exp_check_lf(name, size, d, 1e-6);        \
    } while (0)


int main ()
{
    __title("normalization/norm_standard");


    norm_standard_test("5x4 example", 5, 4,
        ((value_t[])    {-7.4, 1.3, 0.48, -2.467,
                         6.1, -0.1, 0.71, -2.000,
                        -3.8, -1.7, 1.00, -3.000,
                         2.3,  0.8, 0.00, -2.384,
                         8.7, -0.6, 0.25, -3.000}),
        ((value_t[])    {-1.4283029,  1.2883040, -0.0229741,  0.2682729,
                          0.8190268, -0.0378913,  0.6375311,  1.4822597,
                         -0.8290149, -1.5535430,  1.4703419, -1.1172838,
                          0.1864451,  0.8146628, -1.4014196,  0.4840349,
                          1.2518459, -0.5115325, -0.6834792, -1.1172838}));


    norm_standard_test("2x3 simple example", 2, 3,
        ((value_t[])    {1.0, 4.0, 5.0,
                         3.0, 2.0, 5.0}),
        ((value_t[])    {-1.0, 1.0, 0.0,
                         1.0, -1.0, 0.0}));


    norm_standard_test("1x4 negative values", 1, 4,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0}),
        ((value_t[])    {0.0, 0.0, 0.0, 0.0}));


    norm_standard_test("4x1 negative values", 4, 1,
        ((value_t[])    {-1.0, -2.0, -3.0, -4.0}),
        ((value_t[])    {1.341641, 0.4472136, -0.4472136, -1.341641}));


    norm_standard_test("3x3 identical values", 3, 3,
        ((value_t[])    {5.0, 5.0, 5.0,
                         5.0, 5.0, 5.0,
                         5.0, 5.0, 5.0}),
        ((value_t[])    {0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0,
                         0.0, 0.0, 0.0}));


    norm_standard_test("4x4 large range", 4, 4,
        ((value_t[])    {100.0,       200.0,      300.0,      400.0,
                         500.0,       600.0,      700.0,      800.0,
                         900.0,      1000.0,     1100.0,     1200.0,
                         1300.0,     1400.0,     1500.0,     1600.0}),
        ((value_t[])    {-1.341641,  -1.341641,  -1.341641,  -1.341641,
                         -0.4472136, -0.4472136, -0.4472136, -0.4472136,
                          0.4472136,  0.4472136,  0.4472136,  0.4472136,
                          1.341641,   1.341641,   1.341641,   1.341641}));

    return 0;
}
