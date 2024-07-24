#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define tanh_forward_test(name, N, X, Y)        \
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
        tanh_forward(N, x, y);                  \
        __exp_check_lf(name, N, y, 1e-6);       \
    } while (0)


int main()
{
    __title("activations/tanh_forward");


    tanh_forward_test("random example", 10,
        ((value_t[]) {0.7, 2.3, -1.8, 0.0, 0.1, -0.1, -2.7, 4.0, -3.0, -0.0}),
        ((value_t[]) {0.60436778, 0.98009640, -0.94680601, 0.0, 0.09966799,
        -0.09966799, -0.99100745, 0.99932930, -0.99505475, 0.0}));


    tanh_forward_test("positive values", 10,
        ((value_t[]) {1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0}),
        ((value_t[]) {0.76159416, 0.96402758, 0.99505475, 0.99932930, 0.99990920,
        1.0, 1.0, 1.0, 1.0, 1.0}));


    tanh_forward_test("negative values", 10,
        ((value_t[]) {-1.0, -2.0, -3.0, -4.0, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0}),
        ((value_t[]) {-0.76159416, -0.96402758, -0.99505475, -0.99932930,
            -0.99990920, -1.0, -1.0, -1.0, -1.0, -1.0}));


    tanh_forward_test("zero values", 4,
        ((value_t[]) {0.0, 0.0, 0.0, 0.0}),
        ((value_t[]) {0.0, 0.0, 0.0, 0.0}));


    tanh_forward_test("large values", 10,
        ((value_t[]) {100.0, -100.0, 1e5, -1e5, 1e10, -1e10, 1e20, -1e20, 1e40, -1e40}),
        ((value_t[]) {1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0}));


    tanh_forward_test("small values", 8,
        ((value_t[]) {1e-6, -1e-6, 1e-9, -1e-9, 1e-17, -1e-17, 1e-37, -1e-37}),
        ((value_t[]) {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));


    return 0;
}
