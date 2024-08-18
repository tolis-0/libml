#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


/* Statically check if the macro values are sane */
#define nn_create_macro_check(v1, v2, v3, v4, v5, v6,   \
                              e1, e2, e3, e4, e5)       \
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
    } while (0)


/*  Check for values that should and shouldn't be null in nn */
static inline void nn_create_test_struct_nulls(const nn_struct_t *nn)
{
    const int nhb = nn->biases.total == 0;

    __assert_not_null(nn);
    __assert_not_null(nn->num_of_dims);
    __assert_not_null(nn->operation_type);
    __assert_not_null(nn->regularization);

    __assert_not_null(nn->weights.num);
    __assert_not_null(nn->weights.ptr);
    __assert_not_null(nn->weights.of_layer);
    __assert_not_null(nn->biases.num);
    __assert_cond_null(nhb, nn->biases.ptr);
    __assert_not_null(nn->biases.of_layer);

    __assert_not_null(nn->interm.outputs);
    __assert_not_null(nn->interm.ones);
    __assert_not_null(nn->grad.ptr);
    __assert_not_null(nn->grad.weights);
    __assert_not_null(nn->grad.biases);
    __assert_null(nn->grad.out);
    __assert_null(nn->grad.in);

    __assert_null(NN_INPUT(nn));
    __assert_null(NN_OUTPUT(nn));
}


/*  Compare values of various parameters of the neural network
    with given expected ones                                    */
static inline void nn_create_test_vals(const nn_struct_t *nn,
    int nl, int id, int od, int tw, int tb, int md)
{
    assert(nn->num_of_layers  == nl && "Number of layers mismatch");
    assert(nn->max_dims       == md && "Number of max dimensions mismatch");
    assert(NN_INPUT_DIMS(nn)  == id && "Number of input dimensions mismatch");
    assert(NN_OUTPUT_DIMS(nn) == od && "Number of output dimensions mismatch");
    assert(nn->weights.total  == tw && "Number of total weights mismatch");
    assert(nn->biases.total   == tb && "Number of total biases mismatch");

    /* initial non-constant values */
    assert(nn->interm.batch_size == 0 &&
        "batch size of allocated intermediate value arrays should initially be 0");
    assert(nn->grad.batch_size == 0 &&
        "batch size of allocated gradient arrays should initially be 0");
    assert(nn->interm.ones_size == ONES_SIZE_DEFAULT &&
        "allocated size of ones array should initially be 16");

    assert(nn->learning_rate > 0.0 && "learning rate should be positive");
}


/*  Test values of various parameters in each layer
    of the neural network with expected ones            */
static inline void nn_create_test_params(const char *name,
    const nn_struct_t *nn,
    const nn_op_t *exp_op_types,
    const int *exp_n_dims,
    const int *exp_n_weights,
    const int *exp_n_biases,
    const nn_reg_t *exp_reg)
{
    const int n = nn->num_of_layers;

    const nn_op_t *const op_types = nn->operation_type;
    const int *const n_dims = nn->num_of_dims - 1;
    const int *const n_weights = nn->weights.num;
    const int *const n_biases = nn->biases.num;
    //const nn_reg_t *const reg = nn->regularization;

    printf("\nTest %s\n", name);
    __exp_check_d("op_types   ", n, op_types);
    __exp_check_d("n_dims     ", n + 1, n_dims);
    __exp_check_d("n_weights  ", n, n_weights);
    __exp_check_d("n_biases   ", n, n_biases);
    //__exp_check_d("reg_type   ", n, reg);
}


/*  Test the pointers of weights and biases of the neural network */
static inline void nn_create_test_wb_pointers(const nn_struct_t *nn)
{
    const int nl = nn->num_of_layers;
    const weight_t *w_p = nn->weights.ptr;
    const weight_t *b_p = nn->biases.ptr;
    const grad_t  *gw_p = nn->grad.ptr;
    const grad_t  *gb_p = gw_p + nn->weights.total;

    for (int i = 0; i < nl; i++) {
        __assert_null(nn->interm.outputs[i]);

        if (nn->weights.num[i] == 0) {
            __assert_null(nn->weights.of_layer[i]);
            __assert_null(nn->grad.weights[i]);
        } else {
            assert(nn->weights.of_layer[i] == w_p && "weights pointer mismatch");
            assert(nn->grad.weights[i] == gw_p && "gradients of weights pointer mismatch");
        }

        if (nn->biases.num[i] == 0) {
            __assert_null(nn->biases.of_layer[i]);
            __assert_null(nn->grad.biases[i]);
        } else {
            assert(nn->biases.of_layer[i] == b_p && "biases pointer mismatch");
            assert(nn->grad.biases[i] == gb_p && "gradients of biases pointer mismatch");
        }

        w_p += nn->weights.num[i];
        b_p += nn->biases.num[i];
        gw_p += nn->weights.num[i];
        gb_p += nn->biases.num[i];
    }
}


/*
 * General testing macro for nn_create
 * v1: Number of layers (including activation functions)
 * v2: Number of input dimensions
 * v3: Number of output dimensions
 * v4: Total number of weights
 * v5: Total number if biases
 * e1: Type of each layer
 * e2: Dimensions of each layer
 * e3: Weights of each layer
 * e4: Biases of each layer
 * e5: Regularization of each layer
 */
#define nn_create_test(name, sp, v1, v2, v3, v4, v5, v6,        \
                       e1, e2, e3, e4, e5)                      \
    do {                                                        \
        nn_create_macro_check(v1, v2, v3, v4, v5, v6,           \
                              e1, e2, e3, e4, e5);              \
                                                                \
        nn_spec_t *spec = sp;                                   \
        nn_struct_t *nn = nn_create(spec);                      \
                                                                \
        nn_create_test_struct_nulls(nn);                        \
        nn_create_test_vals(nn, v1, v2, v3, v4, v5, v6);        \
        nn_create_test_params(name, nn, e1, e2, e3, e4, e5);    \
        nn_create_test_wb_pointers(nn);                         \
                                                                \
        nn_destroy(nn);                                         \
    } while (0)


int main ()
{
    __title("nn/nn_create");


    nn_create_test("lr:lr:s 784->10",
        ((nn_spec_t[]){
            nnl_input(784),
            nnl_dense(512, 1, LRELU_OP, reg_l1(0.02)),
            nnl_dense(256, 1, LRELU_OP, reg_l2(0.01)),
            nnl_dense(10, 1, LOGISTIC_OP, NO_REG),
            NN_SPEC_END
        }),
        6, 784, 10, 535040, 778, 784,
        ((nn_op_t[]) {DENSE_OP, LRELU_OP, DENSE_OP, LRELU_OP, DENSE_OP, LOGISTIC_OP}),
        ((int[])      {784, 512, 512, 256, 256, 10, 10}),
        ((int[])      {401408, 0, 131072, 0, 2560, 0}),
        ((int[])      {512, 0, 256, 0, 10, 0}),
        ((nn_reg_t[]) {reg_l1(0.02), NO_REG, reg_l2(0.01), NO_REG, NO_REG, NO_REG})
    );


    nn_create_test("r 4->4",
        ((nn_spec_t[]){
            nnl_input(4),
            nnl_dense(4, 1, RELU_OP, reg_l1(0.7)),
            NN_SPEC_END
        }),
        2, 4, 4, 16, 4, 4,
        ((nn_op_t[]) {DENSE_OP, RELU_OP}),
        ((int[])      {4, 4, 4}),
        ((int[])      {16, 0}),
        ((int[])      {4, 0}),
        ((nn_reg_t[]) {reg_l1(0.7), NO_REG})
    );


    nn_create_test("l:l:l:l:s 32->4",
        ((nn_spec_t[]){
            nnl_input(32),
            nnl_dense(64, 1, EMPTY_OP, NO_REG),
            nnl_dense(32, 0, EMPTY_OP, NO_REG),
            nnl_dense(16, 0, EMPTY_OP, reg_l2(0.03)),
            nnl_dense(8, 0, EMPTY_OP, reg_l1(0.07)),
            nnl_dense(4, 1, LOGISTIC_OP, NO_REG),
            NN_SPEC_END
        }),
        6, 32, 4, 4768, 68, 64,
        ((nn_op_t[]) {DENSE_OP, DENSE_OP, DENSE_OP, DENSE_OP, DENSE_OP, LOGISTIC_OP}),
        ((int[])      {32, 64, 32, 16, 8, 4, 4}),
        ((int[])      {2048, 2048, 512, 128, 32, 0}),
        ((int[])      {64, 0, 0, 0, 4, 0}),
        ((nn_reg_t[]) {NO_REG, NO_REG, reg_l2(0.03), reg_l1(0.07), NO_REG, NO_REG})
    );


    nn_create_test("rts 7->3",
        ((nn_spec_t[]){
            nnl_input(7),
            nnl_dense(3, 0, RELU_OP, NO_REG),
            nnl_tanh(),
            nnl_logistic(),
            NN_SPEC_END
        }),
        4, 7, 3, 21, 0, 7,
        ((nn_op_t[]) {DENSE_OP, RELU_OP, TANH_OP, LOGISTIC_OP}),
        ((int[])      {7, 3, 3, 3, 3}),
        ((int[])      {21, 0, 0, 0}),
        ((int[])      {0, 0, 0, 0}),
        ((nn_reg_t[]) {NO_REG, NO_REG, NO_REG, NO_REG})
    );


    nn_create_test("lr:t 2->2",
        ((nn_spec_t[]){
            nnl_input(2),
            nnl_dense(2, 1, EMPTY_OP, NO_REG),
            nnl_lrelu(),
            nnl_dense(2, 1, TANH_OP, NO_REG),
            NN_SPEC_END
        }),
        4, 2, 2, 8, 4, 2,
        ((nn_op_t[]) {DENSE_OP, LRELU_OP, DENSE_OP, TANH_OP}),
        ((int[])      {2, 2, 2, 2, 2}),
        ((int[])      {4, 0, 4, 0}),
        ((int[])      {2, 0, 2, 0}),
        ((nn_reg_t[]) {NO_REG, NO_REG, NO_REG, NO_REG})
    );


    return 0;
}
