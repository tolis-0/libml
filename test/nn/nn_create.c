#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


/*  Statically check if the macro values are sane */
#define nn_create_macro_check(v1, v2, v3, v4, v5, v6,   \
                              e1, e2, e3, e4, e5, e6)   \
    do {                                                \
        _Static_assert((v1) >  0, "layers > 0");        \
        _Static_assert((v2) >  0, "input dims > 0");    \
        _Static_assert((v3) >  0, "output dims > 0");   \
        _Static_assert((v4) >  0, "total weights > 0"); \
        _Static_assert((v5) >= 0, "total biases >= 0"); \
        _Static_assert((v6) >  0, "max dims > 0");      \
        __Static_assert_size(e1, v1);                   \
        __Static_assert_size(e2, (v1) + 1);             \
        __Static_assert_size(e3, v1);                   \
        __Static_assert_size(e4, v1);                   \
        __Static_assert_size(e5, v1);                   \
        __Static_assert_size(e6, v1);                   \
    } while (0)


/*  Check for values that should and shouldn't be null in nn */
static inline void nn_create_test_struct_nulls(const nn_struct_t *nn)
{
    const int nhb = nn->total_biases == 0;

    __assert_not_null(nn);
    __assert_not_null(nn->op_types);
    __assert_not_null(nn->n_dims);
    __assert_not_null(nn->n_weights);
    __assert_not_null(nn->weights_ptr);
    __assert_not_null(nn->weights);
    __assert_not_null(nn->n_biases);
    __assert_not_null(nn->biases);
    __assert_not_null(nn->reg_type);
    __assert_not_null(nn->reg_p);
    __assert_not_null(nn->outputs);
    __assert_not_null(nn->batch_outputs);
    __assert_not_null(nn->ones);
    __assert_not_null(nn->gw_ptr);
    __assert_not_null(nn->gw);
    __assert_not_null(nn->gb);
    __assert_null(nn->output);
    __assert_null(nn->g_out);
    __assert_null(nn->g_in);
    __assert_cond_null(nhb, nn->biases_ptr);
    __assert_cond_null(nhb, nn->gb_ptr);
}


/*  Compare values of various parameters of the neural network
    with given expected ones                                    */
static inline void nn_create_test_vals(const nn_struct_t *nn,
    int nl, int id, int od, int tw, int tb, int md)
{
    assert(nn->n_layers      == nl && "Number of layers mismatch");
    assert(nn->input_dims    == id && "Number of input dimensions mismatch");
    assert(nn->output_dims   == od && "Number of output dimensions mismatch");
    assert(nn->total_weights == tw && "Number of total weights mismatch");
    assert(nn->total_biases  == tb && "Number of total biases mismatch");
    assert(nn->go_n          == md && "Number of max dimensions mismatch");

    /* initial non-constant values */
    assert(nn->k == 0 && "k (batch size of allocated intermediate value arrays)"
        "should initially be 0");
    assert(nn->g_k == 0 && "g_k (batch size of allocated gradient arrays)"
        "should initially be 0");
    assert(nn->ones_n == 16 && "allocated size of ones array should initially be 16");

    assert(nn->learning_rate > 0.0 && "learning rate should be positive");
}


/*  Test values of various parameters in each layer
    of the neural network with expected ones            */
static inline void nn_create_test_params(const char *name,
    const nn_struct_t *nn,
    const nn_ops_t *exp_op_types,
    const int *exp_n_dims,
    const int *exp_n_weights,
    const int *exp_n_biases,
    const nn_reg_t *exp_reg_type,
    const weight_t *exp_reg_p)
{
    const int n = nn->n_layers;

    const nn_ops_t *const op_types = nn->op_types;
    const int *const n_dims = nn->n_dims - 1;
    const int *const n_weights = nn->n_weights;
    const int *const n_biases = nn->n_biases;
    const nn_reg_t *const reg_type = nn->reg_type;
    const weight_t *const reg_p = nn->reg_p;

    printf("\nTest %s\n", name);
    __exp_check_d("op_types   ", n, op_types);
    __exp_check_d("n_dims     ", n + 1, n_dims);
    __exp_check_d("n_weights  ", n, n_weights);
    __exp_check_d("n_biases   ", n, n_biases);
    __exp_check_d("reg_type   ", n, reg_type);
    __exp_check_f("reg_p      ", n, reg_p, 1e-50);
}


/*  Test the pointers of weights and biases of the neural network */
static inline void nn_create_test_wb_pointers(const nn_struct_t *nn)
{
    const int nl = nn->n_layers;
    const weight_t *w_p = nn->weights_ptr;
    const weight_t *b_p = nn->biases_ptr;
    const grad_t  *gw_p = nn->gw_ptr;
    const grad_t  *gb_p = nn->gb_ptr;

    for (int i = 0; i < nl; i++) {
        __assert_not_null(nn->outputs[i]);
        __assert_null(nn->batch_outputs[i]);

        if (nn->n_weights[i] == 0) {
            __assert_null(nn->weights[i]);
            __assert_null(nn->gw[i]);
        } else {
            assert(nn->weights[i] == w_p && "weights pointer mismatch");
            assert(nn->gw[i] == gw_p && "gradients of weights pointer mismatch");
        }

        if (nn->n_biases[i] == 0) {
            __assert_null(nn->biases[i]);
            __assert_null(nn->gb[i]);
        } else {
            assert(nn->biases[i] == b_p && "biases pointer mismatch");
            assert(nn->gb[i] == gb_p && "gradients of biases pointer mismatch");
        }

        w_p += nn->n_weights[i];
        b_p += nn->n_biases[i];
        gw_p += nn->n_weights[i];
        gb_p += nn->n_biases[i];
    }
}


/*  General testing macro for nn_create
    v1: Number of layers (including activation functions)
    v2: Number of input dimensions
    v3: Number of output dimensions
    v4: Total number of weights
    v5: Total number if biases
    e1: Type of each layer
    e2: Dimensions of each layer
    e3: Weights of each layer
    e4: Biases of each layer
    e5: Type of regularization of each layer
    e6: Regularization parameter of each layer                         */
#define nn_create_test(name, sp, v1, v2, v3, v4, v5, v6,                \
                       e1, e2, e3, e4, e5, e6)                          \
    do {                                                                \
        nn_create_macro_check(v1, v2, v3, v4, v5, v6,                   \
                              e1, e2, e3, e4, e5, e6);                  \
                                                                        \
        nn_spec_t *spec = sp;                                           \
        nn_struct_t *nn = nn_create(spec);                              \
                                                                        \
        nn_create_test_struct_nulls(nn);                                \
        nn_create_test_vals(nn, v1, v2, v3, v4, v5, v6);                \
        nn_create_test_params(name, nn, e1, e2, e3, e4, e5, e6);        \
        nn_create_test_wb_pointers(nn);                                 \
                                                                        \
        nn_destroy(nn);                                                 \
    } while (0)


int main ()
{
    __title("nn/nn_create");


    nn_create_test("lr:lr:s 784->10",
        ((nn_spec_t[]){
            input_layer(784),
            dense_layer(512, b, lrelu, l1(0.02)),
            dense_layer(256, b, lrelu, l2(0.01)),
            dense_layer(10, b, logistic),
            output_layer()
        }),
        6, 784, 10, 535040, 778, 784,
        ((nn_ops_t[]) {DENSE_OP, LRELU_OP, DENSE_OP, LRELU_OP,
                       DENSE_OP, LOGISTIC_OP}),
        ((int[])      {784, 512, 512, 256, 256, 10, 10}),
        ((int[])      {401408, 0, 131072, 0, 2560, 0}),
        ((int[])      {512, 0, 256, 0, 10, 0}),
        ((nn_reg_t[]) {L1, NONE, L2, NONE, NONE, NONE}),
        ((weight_t[]) {0.02, 0.0, 0.01, 0.0, 0.0, 0.0})
    );


    nn_create_test("r 4->4",
        ((nn_spec_t[]){
            input_layer(4),
            dense_layer(4, b, relu, l1(0.7)),
            output_layer()
        }),
        2, 4, 4, 16, 4, 4,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP}),
        ((int[])      {4, 4, 4}),
        ((int[])      {16, 0}),
        ((int[])      {4, 0}),
        ((nn_reg_t[]) {L1, NONE}),
        ((weight_t[]) {0.7, 0.0})
    );


    nn_create_test("l:l:l:l:s 32->4",
        ((nn_spec_t[]){
            input_layer(32),
            dense_layer(64, b, linear),
            dense_layer(32, x, linear),
            dense_layer(16, x, linear, l2(0.03)),
            dense_layer(8, x, linear, l1(0.07)),
            dense_layer(4, b, logistic),
            output_layer()
        }),
        6, 32, 4, 4768, 68, 64,
        ((nn_ops_t[]) {DENSE_OP, DENSE_OP, DENSE_OP,
                       DENSE_OP, DENSE_OP, LOGISTIC_OP}),
        ((int[])      {32, 64, 32, 16, 8, 4, 4}),
        ((int[])      {2048, 2048, 512, 128, 32, 0}),
        ((int[])      {64, 0, 0, 0, 4, 0}),
        ((nn_reg_t[]) {NONE, NONE, L2, L1, NONE, NONE}),
        ((weight_t[]) {0.0, 0.0, 0.03, 0.07, 0.0, 0.0})
    );


    nn_create_test("rts 7->3",
        ((nn_spec_t[]){
            input_layer(7),
            dense_layer(3, x, relu),
            tanh_layer(),
            logistic_layer(),
            output_layer()
        }),
        4, 7, 3, 21, 0, 7,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP, TANH_OP, LOGISTIC_OP}),
        ((int[])      {7, 3, 3, 3, 3}),
        ((int[])      {21, 0, 0, 0}),
        ((int[])      {0, 0, 0, 0}),
        ((nn_reg_t[]) {NONE, NONE, NONE, NONE}),
        ((weight_t[]) {0.0, 0.0, 0.0, 0.0})
    );


    nn_create_test("lr:t 2->2",
        ((nn_spec_t[]){
            input_layer(2),
            dense_layer(2, b, linear),
            lrelu_layer(),
            dense_layer(2, b, tanh),
            output_layer()
        }),
        4, 2, 2, 8, 4, 2,
        ((nn_ops_t[]) {DENSE_OP, LRELU_OP, DENSE_OP, TANH_OP}),
        ((int[])      {2, 2, 2, 2, 2}),
        ((int[])      {4, 0, 4, 0}),
        ((int[])      {2, 0, 2, 0}),
        ((nn_reg_t[]) {NONE, NONE, NONE, NONE}),
        ((weight_t[]) {0.0, 0.0, 0.0, 0.0})
    );


    return 0;
}
