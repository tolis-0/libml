#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../test.h"
#include "../../include/nn.h"
#include "../../src/nn/nn_internal.h"


#define nn_batch_backward_pass_test(name, nn, K, W, B, X,           \
                                    VAL, T, oG, uW, uB)             \
    do {                                                            \
        const int w_n = nn->total_weights;                          \
        const unsigned b_n = nn->total_biases;                      \
        const int i_n = nn->input_dims;                             \
        const int i_v = __arr_count(VAL);                           \
        const int l_n = nn->n_layers;                               \
        const int o_n = nn->output_dims;                            \
                                                                    \
        int n_outputs = 0;                                          \
        for (int i = 0; i < l_n; i++)                               \
            n_outputs += nn->n_dims[i];                             \
                                                                    \
        assert(__arr_count(W) == w_n);                              \
        assert(__arr_count(uW) == w_n);                             \
        assert(__arr_count_null(B) == b_n);                         \
        assert(__arr_count_null(uB) == b_n);                        \
        assert(__arr_count(X) == i_n * (K));                        \
        assert(__arr_count(VAL) == (K) * n_outputs);                \
        assert(__arr_count(oG) == (K) * o_n);                       \
        assert(__arr_count(T) == (K) * o_n);                        \
                                                                    \
        const weight_t *w = W;                                      \
        const weight_t *b = B;                                      \
        value_t *x = X;                                             \
        const value_t *exp_val = VAL;                               \
        value_t *val = malloc(i_v * sizeof(value_t));               \
        const value_t *t = T;                                       \
        const grad_t *exp_og = oG;                                  \
        const weight_t *exp_uw = uW;                                \
        const weight_t *exp_ub = uB;                                \
                                                                    \
        /*  Set the custom weights and biases */                    \
        memcpy(nn->weights_ptr, w, w_n * sizeof(weight_t));         \
        if (b_n) memcpy(nn->biases_ptr, b, b_n * sizeof(weight_t)); \
                                                                    \
        _nn_alloc_batch(nn, (K));                                   \
        _nn_alloc_grad(nn, (K));                                    \
        nn->batch_outputs[-1] = x;                                  \
        nn_batch_forward_pass(nn, (K));                             \
                                                                    \
        for (int k = 0, val_i = 0; k < K; k++) {                    \
            for (int i = 0; i < l_n; i++) {                         \
                const int n = nn->n_dims[i];                        \
                const value_t *o = nn->batch_outputs[i] + k * n;    \
                memcpy(val + val_i, o, n * sizeof(value_t));        \
                val_i += n;                                         \
            }                                                       \
        }                                                           \
                                                                    \
        __exp_check_lf(name " (outputs)", i_v, val, 1e-9);          \
                                                                    \
        assert(nn->output == nn->batch_outputs[l_n - 1]);           \
                                                                    \
        loss_diff_grad((K) * o_n, nn->output, t, nn->g_out);        \
                                                                    \
        const grad_t *og = nn->g_out;                               \
        __exp_check_lf(name " (g_out)", (K) * o_n, og, 1e-9);       \
                                                                    \
        nn_batch_backward_pass(nn, (K));                            \
                                                                    \
        const weight_t *uw = nn->weights_ptr;                       \
        const weight_t *ub = nn->biases_ptr;                        \
        __exp_check_lf(name " (updated weights)", w_n, uw, 1e-8);   \
        __exp_check_lf(name " (updated biases)", b_n, ub, 1e-8);    \
                                                                    \
        _nn_free_batch(nn);                                         \
        _nn_free_grad(nn);                                          \
        free(val);                                                  \
    } while (0)


int main ()
{
    __title("nn/nn_batch_backward_pass");


    nn_spec_t spec1[] = {
        input_layer(2),
        dense_layer(2, b, relu),
        dense_layer(2, b, logistic),
        output_layer()
    };

    nn_struct_t *nn1 = nn_create(spec1);
    nn1->learning_rate = 0.03;

    /*  Computed with testgrad1.m */
    nn_batch_backward_pass_test("2:2:2 rs w/ b, k=3 test 1", nn1, 3,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {1.6, -0.5,
                         -1.0, 1.0,
                         0.5, 0.4}),
        ((value_t[])    {2.43, -1.86, 2.43, 0.0, -1.451, 0.879, 0.1898477127, 0.7066149537,
                         -2.3, 1.6, 0.0, 1.6, -2.31, 3.35, 0.0902981447, 0.9661048358,
                         0.19, -0.29, 0.19, 0.0, 0.117, 0.207, 0.5292166787, 0.5515660021}),
        ((value_t[])    {1.0, 0.0,
                         0.0, 1.0,
                         1.0, 0.0}),
        ((grad_t[])     {-0.8101522873, 0.7066149537,
                         0.0902981447, -0.0338951642,
                         -0.4707833213, 0.5515660021}),
        ((weight_t[])   {1.2972861025, -0.8998362794, -1.1001408786, 0.4001408786,
                         -0.6967492175, -1.6001186798, 0.2961811215, 2.0000177590}),
        ((weight_t[])   {-0.1025420394, 0.1001408786, 0.252344824, 0.147181965})
    );

    /*  Computed with testgrad2.m */
    nn_batch_backward_pass_test("2:2:2 rs w/ b, k=3 test 2", nn1, 3,
        ((weight_t[])   {-0.47, -0.23, -0.59, 0.41, 0.33, -0.60, 0.27, 0.56}),
        ((weight_t[])   {0.01, -0.01, -0.01, 0.01}),
        ((value_t[])    {2.14, -1.36,
                         1.76, 1.45,
                         -1.84, 2.94}),
        ((value_t[])    {-0.683, -1.8302, 0.0, 0.0, -0.01, 0.01, 0.4975000208, 0.5024999792,
                         -1.1507, -0.4539, 0.0, 0.0, -0.01, 0.01, 0.4975000208, 0.5024999792,
                         0.1986, 2.2810, 0.1986, 2.2810, -1.313062, 1.340982, 0.2119749118, 0.7926513846}),
        ((value_t[])    {1.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0}),
        ((grad_t[])     {-0.5024999792, 0.5024999792,
                         -0.5024999792, 0.5024999792,
                         0.2119749118, -0.2073486154}),
        ((weight_t[])   {-0.4699543024, -0.2300730168, -0.5907420593, 0.4111856816,
                         0.3299296785, -0.6008076706, 0.2700676805, 0.5607773378}),
        ((weight_t[])   {0.009975164366, -0.009596706923, -0.007841649091, 0.007828351079})
    );


    nn_destroy(nn1);

    return 0;
}
