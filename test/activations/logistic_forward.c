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
        __exp_check_lf(name, N, y, 1e-10);      \
    } while (0)


int main() {
    __title("activations/logistic_forward");


    logistic_forward_test("random example", 10,
        ((value_t[]) {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((value_t[]) {0.668187772168, 0.908877038985, 0.1418510649, 0.5, 0.524979187479,
        0.475020812521, 0.062973356057, 0.982013790038, 0.0474258731776, 0.5}));


    logistic_forward_test("positive values", 10,
        ((value_t[]) {1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0}),
        ((value_t[]) {0.73105857863, 0.880797077978, 0.952574126822, 0.982013790038, 0.993307149076,
        0.999954602131, 0.999999694098, 0.999999997939, 0.999999999986, 1.0}));


    logistic_forward_test("negative values", 10,
        ((value_t[]) {-1.0, -2.0, -3.0, -4.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0}),
        ((value_t[]) {0.26894142137, 0.119202922022, 0.047425873178, 0.017986209962,
            0.006692850924, 0.000045397869, 3.05902e-7, 2.0611e-9, 1.388e-11, 0.0}));


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
