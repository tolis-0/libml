#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../test.h"
#include "../../include/nn.h"
#include "../../src/nn/nn_internal.h"


#define nn_batch_backward_pass_test(name, nn, K, W, B, X,           \
                                    VAL, T, gO, gW, gB)             \
    do {                                                            \
        const int w_n = nn->total_weights;                          \
        const unsigned b_n = nn->total_biases;                      \
        const int i_n = nn->input_dims;                             \
        const int i_v = __arr_count(VAL);                           \
        const int l_n = nn->n_layers;                               \
        const int o_n = nn->output_dims * (K);                      \
                                                                    \
        int n_outputs = 0;                                          \
        for (int i = 0; i < l_n; i++)                               \
            n_outputs += nn->n_dims[i];                             \
                                                                    \
        assert(__arr_count(W) == w_n);                              \
        assert(__arr_count(gW) == w_n);                             \
        assert(__arr_count_null(B) == b_n);                         \
        assert(__arr_count_null(gB) == b_n);                        \
        assert(__arr_count(X) == i_n * (K));                        \
        assert(__arr_count(VAL) == (K) * n_outputs);                \
        assert(__arr_count(gO) == o_n);                             \
        assert(__arr_count(T) == o_n);                              \
                                                                    \
        const weight_t *w = W;                                      \
        const weight_t *b = B;                                      \
        value_t *x = X;                                             \
        const value_t *exp_val = VAL;                               \
        value_t *val = malloc(i_v * sizeof(value_t));               \
        const value_t *t = T;                                       \
        const grad_t *exp_go = gO;                                  \
        const weight_t *exp_gw = gW;                                \
        const weight_t *exp_gb = gB;                                \
                                                                    \
        /*  Set the custom weights and biases */                    \
        memcpy(nn->weights_ptr, w, w_n * sizeof(weight_t));         \
        if (b_n) memcpy(nn->biases_ptr, b, b_n * sizeof(weight_t)); \
                                                                    \
        _nn_alloc_batch(nn, (K), __func__, __FILE__, __LINE__);     \
        _nn_alloc_grad(nn, (K), __func__, __FILE__, __LINE__);      \
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
        __exp_check_lf(name " (outputs)", i_v, val, 1e-6);          \
                                                                    \
        assert(nn->output == nn->batch_outputs[l_n - 1]);           \
                                                                    \
        loss_diff_grad(o_n, nn->output, t, nn->g_out);              \
                                                                    \
        const grad_t *go = nn->g_out;                               \
        __exp_check_lf(name " (output gradients)", o_n, go, 1e-6);  \
                                                                    \
        nn_batch_backward_pass(nn, (K));                            \
                                                                    \
        const grad_t *gw = nn->gw_ptr;                              \
        const grad_t *gb = nn->gb_ptr;                              \
        __exp_check_lf(name " (weight gradients)", w_n, gw, 1e-6);  \
        __exp_check_lf(name " (bias gradients)", b_n, gb, 1e-6);    \
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

    /*  Computed with testgrad1.m */
    nn_batch_backward_pass_test("2:2:2 rs w/ b, k=3 test 1", nn1, 3,
        ((weight_t[])   {1.3, -0.9, -1.1, 0.4, -0.7, -1.6, 0.3, 2.0}),
        ((weight_t[])   {-0.1, 0.1, 0.25, 0.15}),
        ((value_t[])    {1.6, -0.5,
                         -1.0, 1.0,
                         0.5, 0.4}),
        ((value_t[])    {2.43, -1.86, 2.43, 0.0, -1.451, 0.879, 0.1898477127, 0.7066149537,
                         -2.3, 1.6,   0.0, 1.6,  -2.31,  3.35,  0.0902981447, 0.9661048358,
                         0.19, -0.29, 0.19, 0.0,  0.117, 0.207, 0.5292166787, 0.5515660021}),
        ((value_t[])    {1.0, 0.0,
                         0.0, 1.0,
                         1.0, 0.0}),
        ((grad_t[])     {-0.8101522873, 0.7066149537,
                         0.0902981447, -0.0338951642,
                         -0.4707833213, 0.5515660021}),
        ((grad_t[])     {0.0904632488, -0.0054573536, 0.0046959529, -0.0046959529,
                        -0.1083594171, 0.0039559925, 0.1272959506, -0.0005919683}),
        ((grad_t[])     {0.0847346460, -0.0046959529, -0.0781608007, 0.0939344824})
    );

    /*  Computed with testgrad2.m */
    nn_batch_backward_pass_test("2:2:2 rs w/ b, k=3 test 2", nn1, 3,
        ((weight_t[])   {-0.47, -0.23, -0.59, 0.41, 0.33, -0.60, 0.27, 0.56}),
        ((weight_t[])   {0.01, -0.01, -0.01, 0.01}),
        ((value_t[])    {2.14, -1.36,
                         1.76, 1.45,
                         -1.84, 2.94}),
        ((value_t[])    {-0.683, -1.8302,  0.0, 0.0,       -0.01, 0.01,         0.4975000208, 0.5024999792,
                         -1.1507, -0.4539, 0.0, 0.0,       -0.01, 0.01,         0.4975000208, 0.5024999792,
                         0.1986, 2.2810,   0.1986, 2.2810, -1.313062, 1.340982, 0.2119749118, 0.7926513846}),
        ((value_t[])    {1.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0}),
        ((grad_t[])     {-0.5024999792, 0.5024999792,
                         -0.5024999792, 0.5024999792,
                         0.2119749118, -0.2073486154}),
        ((grad_t[])     {-0.0015232522, 0.0024338921,  0.0247353087, -0.0395227215,
                          0.0023440504, 0.0269223522, -0.0022560176, -0.0259112600}),
        ((grad_t[])     {0.0008278544, -0.0134431025, -0.0719450303, 0.0723882974})
    );

    nn_destroy(nn1);


    nn_spec_t spec2[] = {
        input_layer(2),
        dense_layer(2, b, lrelu),
        dense_layer(2, b, tanh),
        output_layer()
    };

    nn_struct_t *nn2 = nn_create(spec2);
    nn2->learning_rate = 0.02;

    /*  Computed with testgrad3.m */
    nn_batch_backward_pass_test("2:2:2 lt w/ b, k=6", nn2, 6,
        ((weight_t[])   {-0.47, -0.23, -0.59, 0.41, 0.33, -0.60, 0.27, 0.56}),
        ((weight_t[])   {0.01, -0.01, -0.01, 0.01}),
        ((value_t[])    {2.14, -1.36,
                         1.76, 1.45,
                         -1.84, 2.94,
                         1.6, -0.5,
                         -1.0, 1.0,
                         0.5, 0.4}),
        ((value_t[])    {-0.683, -1.8302, -0.00683, -0.018302, -0.0012727, -0.00209322, -0.001272699,-0.002093216,
                         -1.1507, -0.4539,-0.011507, -0.004539,-0.01107391, 0.00435127, -0.011073457, 0.004351242,
                         0.1986, 2.281,    0.1986, 2.2810,     -1.313062, 1.340982,     -0.865048137, 0.871907909,
                         -0.627, -1.159,  -0.00627, -0.01159,  -0.0051151, 0.0018167,   -0.005115055, 0.001816698,
                         0.25, 0.99,       0.25, 0.99,         -0.5215, 0.6319,         -0.478856886, 0.559359126,
                         -0.317, -0.141,  -0.00317, -0.00141,  -0.0102001, 0.0083545,   -0.010199746, 0.008354305}),
        ((value_t[])    {1.0, 0.0,
                         1.0, 0.0,
                         0.0, 1.0,
                         1.0, 0.0,
                         0.0, 1.0,
                         1.0, 0.0}),
        ((grad_t[])     {-1.001272699, -0.002093216,
                         -1.011073457,  0.004351242,
                         -0.865048137, -0.128092090,
                         -1.005115055,  0.001816698,
                         -0.478856886, -0.440640874,
                         -1.010199746,  0.008354305}),
        ((grad_t[])     {0.0551848231, -0.0731901496, -0.0373887885, 0.0642465538,
                        -0.0179213288, -0.1376678633, -0.0136443903, -0.061636002}),
        ((grad_t[])     {-0.0494891481, 0.03159155, -0.7690304239, -0.0535095194})
    );

    nn_destroy(nn2);

    return 0;
}
