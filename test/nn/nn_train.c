#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../test.h"
#include "../../include/nn.h"


#define nn_train_test(name, nn, E, K, S, W, B, X, T, uW, uB)        \
    do {                                                            \
        const int w_n = nn->total_weights;                          \
        const unsigned b_n = nn->total_biases;                      \
        const int i_n = nn->input_dims;                             \
        const int o_n = nn->output_dims;                            \
                                                                    \
        assert(__arr_count(W) == w_n);                              \
        assert(__arr_count(uW) == w_n);                             \
        assert(__arr_count_null(B) == b_n);                         \
        assert(__arr_count_null(uB) == b_n);                        \
        assert(__arr_count(X) == i_n * (S));                        \
        assert(__arr_count(T) == o_n * (S));                        \
                                                                    \
        const weight_t *w = W;                                      \
        const weight_t *b = B;                                      \
        value_t *x = X;                                             \
        value_t *t = T;                                             \
        const weight_t *exp_uw = uW;                                \
        const weight_t *exp_ub = uB;                                \
                                                                    \
        /*  Set the custom weights and biases */                    \
        memcpy(nn->weights_ptr, w, w_n * sizeof(weight_t));         \
        if (b_n) memcpy(nn->biases_ptr, b, b_n * sizeof(weight_t)); \
                                                                    \
        nn_train(nn, (E), (K), (S), x, t);                          \
                                                                    \
        const grad_t *uw = nn->weights_ptr;                         \
        const grad_t *ub = nn->biases_ptr;                          \
        __exp_check_lf(name " (updated weights)", w_n, uw, 1e-5);   \
        __exp_check_lf(name " (updated biases)", b_n, ub, 1e-5);    \
    } while (0)


int main()
{
    __title("nn/nn_train");

    nn_spec_t spec[] = {
        input_layer(2),
        dense_layer(2, b, lrelu),
        dense_layer(2, b, logistic),
        output_layer()
    };

    nn_struct_t *nn = nn_create(spec);
    nn->learning_rate = 0.02;
    nn->stochastic = 0;

    /*  Computed with train1.m */
    nn_train_test("2:2:2 lt w/ b, e=3, k=2, s=6", nn, 3, 2, 6,
        ((weight_t[])   {-0.47, -0.23, -0.59, 0.41, 0.33, -0.60, 0.27, 0.56}),
        ((weight_t[])   {0.01, -0.01, -0.01, 0.01}),
        ((value_t[])    {2.14, -1.36,
                         1.76, 1.45,
                         -1.84, 2.94,
                         1.6, -0.5,
                         -1.0, 1.0,
                         0.5, 0.4}),
        ((value_t[])    {1.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0,
                         1.0, 0.0,
                         0.0, 1.0,
                         1.0, 0.0}),
        ((weight_t[])   {-0.4696314, -0.2304351, -0.5953907, 0.41645821,
                         0.32903533, -0.6051592, 0.2708944, 0.5648219}),
        ((weight_t[])   {0.00971497, -0.0060607, 0.001345, -0.00160437})
    );

    nn_destroy(nn);

    return 0;
}
