#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define nn_create_test(sp, v1, v2, v3, v4, v5, e1, e2, e3, e4)  \
    do {                                                        \
        const int n = (v1);                                     \
        nn_spec_t *spec = sp;                                   \
        nn_struct_t *nn = nn_create(spec);                      \
                                                                \
        assert(nn != NULL);                                     \
        assert(nn->op_types != NULL);                           \
        assert(nn->n_dims != NULL);                             \
        assert(nn->learning_rate > 0.0);                        \
                                                                \
        assert(nn->n_layers == (v1));                           \
        assert(nn->input_dims == (v2));                         \
        assert(nn->output_dims == (v3));                        \
        assert(nn->total_weights == (v4));                      \
        assert(nn->total_biases == (v5));                       \
                                                                \
        assert(__arr_count(e1) == n);                           \
        assert(__arr_count(e2) == n + 1);                       \
        assert(__arr_count(e3) == n);                           \
        assert(__arr_count(e4) == n);                           \
                                                                \
        const nn_ops_t *op_types = nn->op_types;                \
        const int *n_dims = nn->n_dims - 1;                     \
        const int *n_weights = nn->n_weights;                   \
        const int *n_biases = nn->n_biases;                     \
                                                                \
        const nn_ops_t *exp_op_types = e1;                      \
        const int *exp_n_dims = e2;                             \
        const int *exp_n_weights = e3;                          \
        const int *exp_n_biases = e4;                           \
                                                                \
        puts("\nTest " __to_str_exp(__COUNTER__));              \
        __exp_check_d("op_types   ", n, op_types);              \
        __exp_check_d("n_dims     ", n + 1, n_dims);            \
        __exp_check_d("n_weights  ", n, n_weights);             \
        __exp_check_d("n_biases   ", n, n_biases);              \
                                                                \
        nn_destroy(nn);                                         \
    } while (0)


int main ()
{
    __title("nn/nn_create");


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(784),
            dense_layer(512, b, relu),
            dense_layer(256, b, relu),
            dense_layer(10, b, logistic),
            output_layer()
        }),
        6, 784, 10, 535040, 778,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP, DENSE_OP, RELU_OP,
                       DENSE_OP, LOGISTIC_OP}),
        ((int[])      {784, 512, 512, 256, 256, 10, 10}),
        ((int[])      {401408, 0, 131072, 0, 2560, 0}),
        ((int[])      {512, 0, 256, 0, 10, 0})
    );


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(4),
            dense_layer(4, b, relu),
            output_layer()
        }),
        2, 4, 4, 16, 4,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP}),
        ((int[])      {4, 4, 4}),
        ((int[])      {16, 0}),
        ((int[])      {4, 0}));


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(32),
            dense_layer(64, b, linear),
            dense_layer(32, x, linear),
            dense_layer(16, x, linear),
            dense_layer(8, x, linear),
            dense_layer(4, b, logistic),
            output_layer()
        }),
        6, 32, 4, 4768, 68,
        ((nn_ops_t[]) {DENSE_OP, DENSE_OP, DENSE_OP,
                       DENSE_OP, DENSE_OP, LOGISTIC_OP}),
        ((int[])      {32, 64, 32, 16, 8, 4, 4}),
        ((int[])      {2048, 2048, 512, 128, 32, 0}),
        ((int[])      {64, 0, 0, 0, 4, 0})
    );


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(7),
            dense_layer(3, x, relu),
            relu_layer(),
            logistic_layer(),
            output_layer()
        }),
        4, 7, 3, 21, 0,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP, RELU_OP, LOGISTIC_OP}),
        ((int[])      {7, 3, 3, 3, 3}),
        ((int[])      {21, 0, 0, 0}),
        ((int[])      {0, 0, 0, 0}));


    return 0;
}
