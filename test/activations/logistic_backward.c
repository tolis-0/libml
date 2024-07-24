#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define logistic_backward_test(name, N, Y, GY, GX)  \
    do {                                            \
        assert((N) > 0);                            \
        assert(__arr_count(Y) == (N));              \
        assert(__arr_count(GY) == (N));             \
        assert(__arr_count(GX) == (N));             \
                                                    \
        value_t Gx[N] = __nan_array(N);             \
                                                    \
        const value_t *y = Y;                       \
        const grad_t *Gy = GY;                      \
        const grad_t *exp_Gx = GX;                  \
                                                    \
        logistic_backward(N, y, Gy, Gx);            \
        __exp_check_lf(name, N, Gx, 1e-6);          \
    } while (0)


int main ()
{
    __title("activations/logistic_backward");


    logistic_backward_test("positive Gy", 4,
        ((value_t[])    {0.66818777, 0.90887703, 0.14185106, 0.5}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
        ((grad_t[])     {0.02217128, 0.01656391, 0.03651880, 0.1}));


    logistic_backward_test("negative Gy", 4,
        ((value_t[])    {0.2, 0.4, 0.6, 0.8}),
        ((grad_t[])     {-0.1, -0.2, -0.3, -0.4}),
        ((grad_t[])     {-0.016, -0.048, -0.072, -0.064}));


    logistic_backward_test("small Y", 4,
        ((value_t[])    {0.01, 0.001, 0.00017, 0.0000096}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
        ((grad_t[])     {0.00099, 0.0001998, 0.000050991, 0.00000383996}));


    logistic_backward_test("big Y", 4,
        ((value_t[])    {0.99, 0.999, 0.99983, 0.9999904}),
        ((grad_t[])     {0.1, 0.2, 0.3, 0.4}),
        ((grad_t[])     {0.00099, 0.0001998, 0.000050991, 0.00000383996}));


    logistic_backward_test("zero Gy", 6,
        ((value_t[])    {0.99998475, 0.3512384, 1.0, 0.0, 0.0001675, 0.5}),
        ((grad_t[])     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
        ((grad_t[])     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}));


    logistic_backward_test("zero Y", 6,
        ((value_t[])    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0}),
        ((grad_t[])     {0.1, 0.2, 0.3, -0.3, -0.2, -0.1}),
        ((grad_t[])     {0.0, 0.0, 0.0,  0.0,  0.0,  0.0}));


    logistic_backward_test("one Y", 6,
        ((value_t[])    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0}),
        ((grad_t[])     {0.1, 0.2, 0.3, -0.3, -0.2, -0.1}),
        ((grad_t[])     {0.0, 0.0, 0.0,  0.0,  0.0,  0.0}));


    return 0;
}
