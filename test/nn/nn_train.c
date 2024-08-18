#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "../test.h"
#include "../../include/nn.h"


#define nn_train_test(name, nn, E, K, S, W, B, X, T, uW, uB)        \
    do {                                                            \
        const int w_n = nn->weights.total;                          \
        const int b_n = nn->biases.total;                           \
        const int i_n = NN_INPUT_DIMS(nn);                          \
        const int o_n = NN_OUTPUT_DIMS(nn);                         \
                                                                    \
        __assert_size(W, w_n);                                      \
        __assert_size(uW, w_n);                                     \
        __assert_size_null(B, b_n);                                 \
        __assert_size_null(uB, b_n);                                \
        __assert_size(X, i_n * (S));                                \
        __assert_size(T, o_n * (S));                                \
                                                                    \
        const weight_t *w = W;                                      \
        const weight_t *b = B;                                      \
        value_t *x = X;                                             \
        value_t *t = T;                                             \
        const weight_t *exp_uw = uW;                                \
        const weight_t *exp_ub = uB;                                \
                                                                    \
        /*  Set the custom weights and biases */                    \
        memcpy(nn->weights.ptr, w, w_n * sizeof(weight_t));         \
        if (b_n) memcpy(nn->biases.ptr, b, b_n * sizeof(weight_t)); \
                                                                    \
        nn_train(nn, (E), (K), (S), x, t);                          \
                                                                    \
        const grad_t *uw = nn->weights.ptr;                         \
        const grad_t *ub = nn->biases.ptr;                          \
        __exp_check_lf(name " (updated weights)", w_n, uw, 1e-5);   \
        __exp_check_lf(name " (updated biases)", b_n, ub, 1e-5);    \
    } while (0)


int main()
{
    __title("nn/nn_train");

    nn_spec_t spec[] = {
        nnl_input(2),
        nnl_dense(2, 1, LRELU_OP, NO_REG),
        nnl_dense(2, 1, LOGISTIC_OP, NO_REG),
        NN_SPEC_END
    };

    nn_struct_t *nn = nn_create(spec);
    nn->learning_rate = 0.02;
    nn->stochastic = 0;

    /* Computed with train1.m */
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
