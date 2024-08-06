#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "../test.h"
#include "../../include/nn.h"
#include "../../src/nn/nn_internal.h"


/*  Check sizes of given expected value arrays */
#define nn_batch_backward_pass_size_check(nn, n_vals,       \
                        K, W, B, X, VAL, T, gO, gW, gB)     \
    do {                                                    \
        _Static_assert((K)  >  0, "batch size > 0");        \
                                                            \
        assert(__arr_count(W)  == nn->total_weights);       \
        assert(__arr_count(gW) == nn->total_weights);       \
        assert(__arr_count_null(B)  == (unsigned) nn->total_biases);   \
        assert(__arr_count_null(gB) == (unsigned) nn->total_biases);   \
        assert(__arr_count(X) == nn->input_dims * (K));     \
        assert(__arr_count(VAL) == n_vals * (K));           \
        assert(__arr_count(gO) == nn->output_dims * (K));   \
        assert(__arr_count(T) == nn->output_dims * (K));    \
    } while (0)


/*  Does the whole process of forward and backward pass
    in order to check the outputs, output gradients and weight/bias gradients */
static inline void nn_batch_backward_pass_test_grads(const char *name,
    nn_struct_t *nn, int batch_size, int iterm_n,
    const weight_t *w, const weight_t *b, value_t *x, const value_t *t,
    const value_t *exp_val,
    const grad_t *exp_go,
    const weight_t *exp_gw,
    const weight_t *exp_gb)
{
    const int w_n = nn->total_weights;
    const unsigned b_n = nn->total_biases;
    const int l_n = nn->n_layers;
    const int o_n = nn->output_dims * batch_size;
    const grad_t *gw = nn->gw_ptr;
    const grad_t *gb = nn->gb_ptr;

    /*  Need to manually handle memory management */
    _nn_alloc_batch(nn, batch_size, __func__, __FILE__, __LINE__);
    _nn_alloc_grad(nn, batch_size, __func__, __FILE__, __LINE__);
    value_t *val = malloc(iterm_n * sizeof(value_t));
    grad_t *go = malloc(o_n * sizeof(grad_t));

    /*  Set the custom weights and biases */
    memcpy(nn->weights_ptr, w, w_n * sizeof(weight_t));
    if (b_n) memcpy(nn->biases_ptr, b, b_n * sizeof(weight_t));

    nn->batch_outputs[-1] = x;
    nn_batch_forward_pass(nn, batch_size);
    loss_diff_grad(o_n, nn->output, t, nn->g_out);
    assert(nn->output == nn->batch_outputs[l_n - 1]);

    /*  Copy the values of intermediate values and output gradients */
    memcpy(go, nn->g_out, o_n * sizeof(grad_t));
    for (int k = 0, val_i = 0; k < batch_size; k++) {
        for (int i = 0; i < l_n; i++) {
            const int n = nn->n_dims[i];
            const value_t *o = nn->batch_outputs[i] + k * n;
            memcpy(val + val_i, o, n * sizeof(value_t));
            val_i += n; // index in the val array
        }
    }

    nn_batch_backward_pass(nn, batch_size);

    printf("\nTest %s\n", name);
    __exp_check_lf("intermediate values  ", iterm_n, val, 1e-6);
    __exp_check_lf("output gradients     ", o_n, go, 1e-6);
    __exp_check_lf("weight gradients     ", w_n, gw, 1e-6);
    __exp_check_lf("bias gradients       ", b_n, gb, 1e-6);

    _nn_free_batch(nn);
    _nn_free_grad(nn);
    free(val);
    free(go);
}


#define nn_batch_backward_pass_test(name, nn, K, W, B, X,           \
                                    VAL, T, gO, gW, gB)             \
    do {                                                            \
        int n_vals = 0;                                             \
        for (int i = 0; i < nn->n_layers; i++)                      \
            n_vals += nn->n_dims[i];                                \
                                                                    \
        nn_batch_backward_pass_size_check(nn, n_vals, K, W, B, X,   \
                                          VAL, T, gO, gW, gB);      \
        nn_batch_backward_pass_test_grads(name, nn, K, n_vals * (K),\
                                          W, B, X, T,               \
                                          VAL, gO, gW, gB);         \
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
    nn_batch_backward_pass_test("r:l 2->2, k=3 (1)", nn1, 3,
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
    nn_batch_backward_pass_test("r:l 2->2, k=3 (2)", nn1, 3,
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
    nn_batch_backward_pass_test("lr:t 2->2, k=6", nn2, 6,
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
