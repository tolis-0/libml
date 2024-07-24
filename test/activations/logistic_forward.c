#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define logistic_forward_test(name, N, X, Y)    \
    do {                                        \
        assert((N) > 0);                        \
        assert(__arr_count(X) == (N));          \
        assert(__arr_count(Y) == (N));          \
                                                \
        value_t y[N] = __nan_array(N);          \
                                                \
        const value_t *x = X;                   \
        const value_t *exp_y = Y;               \
                                                \
        logistic_forward(N, x, y);              \
        __exp_check_lf(name, N, y, 1e-6);       \
    } while (0)


int main()
{
    __title("activations/logistic_forward");


    logistic_forward_test("random example", 10,
        ((value_t[]) {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((value_t[]) {0.66818777, 0.90887703, 0.14185106, 0.5, 0.52497918,
        0.47502081, 0.06297335, 0.98201379, 0.047425873, 0.5}));


    logistic_forward_test("positive values", 10,
        ((value_t[]) {1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0}),
        ((value_t[]) {0.731058578, 0.880797078, 0.9525741268, 0.98201379, 0.993307149,
        0.9999546, 0.99999969, 0.999999998, 0.99999999, 1.0}));


    logistic_forward_test("negative values", 10,
        ((value_t[]) {-1.0, -2.0, -3.0, -4.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0}),
        ((value_t[]) {0.2689414214, 0.119202922, 0.047425873, 0.01798621,
            0.0066928509, 0.000045397, 3.05902e-7, 0.0, 0.0, 0.0}));


    logistic_forward_test("zero values", 4,
        ((value_t[]) {0.0, 0.0, 0.0, 0.0}),
        ((value_t[]) {0.5, 0.5, 0.5, 0.5}));


    logistic_forward_test("large values", 10,
        ((value_t[]) {100.0, -100.0, 1e5, -1e5, 1e10, -1e10, 1e20, -1e20, 1e40, -1e40}),
        ((value_t[]) {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0}));


    logistic_forward_test("small values", 8,
        ((value_t[]) {1e-6, -1e-6, 1e-9, -1e-9, 1e-17, -1e-17, 1e-37, -1e-37}),
        ((value_t[]) {0.50000025, 0.49999975, 0.50000000025, 0.49999999975,
            0.5, 0.5, 0.5, 0.5}));


    return 0;
}
