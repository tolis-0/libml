#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../test.h"
#include "../../include/nn.h"
#include "../../src/nn/nn_internal.h"


#define nn_forward_pass_test(name, nn, W, B, X, VAL)        \
    do {                                                    \
        const int w_n = nn->weights.total;                  \
        const int b_n = nn->biases.total;                   \
        const int i_n = NN_INPUT_DIMS(nn);                  \
        const int i_v = __arr_count(VAL);                   \
        const int l_n = nn->num_of_layers;                  \
                                                            \
        __assert_size(W, w_n);                              \
        __assert_size_null(B, b_n);                         \
        __assert_size(X, i_n);                              \
                                                            \
        const weight_t *w = W;                              \
        const weight_t *b = B;                              \
        value_t *x = X;                                     \
        const value_t *exp_val = VAL;                       \
        value_t *val = malloc(i_v * sizeof(value_t));       \
                                                            \
        /* Set the custom weights and biases */             \
        memcpy(nn->weights.ptr, w, w_n * sizeof(weight_t)); \
        memcpy(nn->biases.ptr, b, b_n * sizeof(weight_t));  \
                                                            \
        _nn_alloc_interm(nn, 1);                            \
        NN_INPUT(nn) = x;                                   \
        _nn_forward_pass(nn);                               \
                                                            \
        for (int i = 0, val_i = 0; i < l_n; i++) {          \
            const int n = nn->num_of_dims[i];               \
            const value_t *o = nn->interm.outputs[i];       \
            memcpy(val + val_i, o, n * sizeof(value_t));    \
            val_i += n;                                     \
        }                                                   \
                                                            \
        __exp_check_lf(name " (outputs)", i_v, val, 1e-6);  \
                                                            \
        _nn_free_interm(nn);                                \
        free(val);                                          \
    } while (0)


int main ()
{
    __title("nn/nn_forward_pass");


    nn_spec_t spec1[] = {
        nnl_input(2),
        nnl_dense(2, 1, RELU_OP, NO_REG),
        nnl_dense(2, 1, LOGISTIC_OP, NO_REG),
        NN_SPEC_END
    };

    nn_struct_t *nn1 = nn_create(spec1);

    nn_forward_pass_test("2:2:2 rs w/ b, test 1", nn1,
        ((weight_t[])   {0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8}),
        ((weight_t[])   {-0.04, 0.07, 0.03, -0.12}),
        ((value_t[])    {1.4, -0.9}),
        ((value_t[])    {0.28, 0.13, 0.28, 0.13, 0.092, -0.212, 0.52298379, 0.4471976})
    );

    nn_forward_pass_test("2:2:2 rs w/ b, test 2", nn1,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {1.6, -0.5}),
        ((value_t[])    {2.43, -1.86, 2.43, 0.0, -1.451, 0.879, 0.1898477, 0.7066149})
    );

    nn_forward_pass_test("2:2:2 rs w/ b, test 3", nn1,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {-1.0, 1.0}),
        ((value_t[])    {-2.3, 1.6, 0.0, 1.6, -2.31, 3.35, 0.09029814, 0.9661048})
    );

    nn_forward_pass_test("2x2x2 rs w/ b, test 3", nn1,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {0.5, 0.4}),
        ((value_t[])    {0.19, -0.29, 0.19, 0.0, 0.117, 0.207, 0.52921668, 0.551566})
    );

    nn_destroy(nn1);

    return 0;
}
