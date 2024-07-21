#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../test.h"
#include "../../include/nn.h"


#define nn_forward_pass_test(name, nn, W, B, X, VAL)        \
    do {                                                    \
        const int w_n = nn->total_weights;                  \
        const unsigned long b_n = nn->total_biases;         \
        const int i_n = nn->input_dims;                     \
        const int i_v = __arr_count(VAL);                   \
        const int l_n = nn->n_layers;                       \
                                                            \
        assert(__arr_count(W) == w_n);                      \
        assert(__arr_count_null(B) == b_n);                 \
        assert(__arr_count(X) == i_n);                      \
                                                            \
        const weight_t *w = W;                              \
        const weight_t *b = B;                              \
        value_t *x = X;                                     \
        const value_t *exp_val = VAL;                       \
        value_t *val = malloc(i_v * sizeof(value_t));       \
                                                            \
        /*  Set the custom weights and biases */            \
        memcpy(nn->weights_ptr, w, w_n * sizeof(weight_t)); \
        memcpy(nn->biases_ptr, b, b_n * sizeof(weight_t));  \
                                                            \
        nn->outputs[-1] = x;                                \
        nn_forward_pass(nn);                                \
                                                            \
        for (int i = 0, val_i = 0; i < l_n; i++) {          \
            const int n = nn->n_dims[i];                    \
            const value_t *o = nn->outputs[i];              \
            memcpy(val + val_i, o, n * sizeof(value_t));    \
            val_i += n;                                     \
        }                                                   \
                                                            \
        __exp_check_lf(name " (outputs)", i_v, val, 1e-9);  \
                                                            \
        assert(nn->output == nn->outputs[l_n - 1]);         \
                                                            \
        free(val);                                          \
    } while (0)


int main ()
{
    __title("nn/nn_forward_pass");


    nn_spec_t spec1[] = {
        input_layer(2),
        dense_layer(2, b, relu),
        dense_layer(2, b, logistic),
        output_layer()
    };

    nn_struct_t *nn1 = nn_create(spec1);

    nn_forward_pass_test("2:2:2 rs w/ b, test 1", nn1,
        ((weight_t[])   {0.1, -0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8}),
        ((weight_t[])   {-0.04, 0.07, 0.03, -0.12}),
        ((value_t[])    {1.4, -0.9}),
        ((value_t[])    {0.28, 0.13, 0.28, 0.13, 0.092, -0.212, 0.522983791, 0.4471976146})
    );

    nn_forward_pass_test("2:2:2 rs w/ b, test 2", nn1,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {1.6, -0.5}),
        ((value_t[])    {2.43, -1.86, 2.43, 0.0, -1.451, 0.879, 0.1898477127, 0.7066149537})
    );

    nn_forward_pass_test("2:2:2 rs w/ b, test 3", nn1,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {-1.0, 1.0}),
        ((value_t[])    {-2.3, 1.6, 0.0, 1.6, -2.31, 3.35, 0.0902981447, 0.9661048358})
    );

    nn_forward_pass_test("2x2x2 rs w/ b, test 3", nn1,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {0.5, 0.4}),
        ((value_t[])    {0.19, -0.29, 0.19, 0.0, 0.117, 0.207, 0.5292166787, 0.5515660021})
    );

    nn_destroy(nn1);

    return 0;
}
