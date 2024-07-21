#include <stdio.h>
#include <assert.h>
#include "../test.h"
#include "../../include/nn.h"


#define nn_create_test(sp, v1, v2, v3, v4, v5, e1, e2, e3, e4, e5, e6)  \
    do {                                                                \
        const int n = (v1);                                             \
        nn_spec_t *spec = sp;                                           \
        nn_struct_t *nn = nn_create(spec);                              \
                                                                        \
        assert(nn != NULL);                                             \
        assert(nn->op_types != NULL);                                   \
        assert(nn->n_dims != NULL);                                     \
        assert(nn->learning_rate > 0.0);                                \
                                                                        \
        assert(nn->n_layers == (v1));                                   \
        assert(nn->input_dims == (v2));                                 \
        assert(nn->output_dims == (v3));                                \
        assert(nn->total_weights == (v4));                              \
        assert(nn->total_biases == (v5));                               \
        assert(nn->k == 0);                                             \
        assert(nn->g_k == 0);                                           \
        assert(nn->ones_n == 16);                                       \
                                                                        \
        int max_w = 0, max_b = 0, max_d = nn->input_dims;               \
        for (int i = 0; i < n; i++) {                                   \
            const int n_w = nn->n_weights[i];                           \
            const int n_b = nn->n_biases[i];                            \
            const int n_d = nn->n_dims[i];                              \
            max_w = (n_w > max_w) ? n_w : max_w;                        \
            max_b = (n_b > max_b) ? n_b : max_b;                        \
            max_d = (n_d > max_d) ? n_d : max_d;                        \
        }                                                               \
                                                                        \
        assert(nn->gw_n == max_w);                                      \
        assert(nn->gb_n == max_b);                                      \
        assert(nn->go_n == max_d);                                      \
                                                                        \
        assert(__arr_count(e1) == n);                                   \
        assert(__arr_count(e2) == n + 1);                               \
        assert(__arr_count(e3) == n);                                   \
        assert(__arr_count(e4) == n);                                   \
        assert(__arr_count(e5) == n);                                   \
        assert(__arr_count(e6) == n);                                   \
                                                                        \
        assert(nn->n_weights != NULL);                                  \
        assert(nn->weights_ptr != NULL);                                \
        assert(nn->weights != NULL);                                    \
        assert(nn->n_biases != NULL);                                   \
        assert(nn->biases_ptr != NULL || nn->total_biases == 0);        \
        assert(nn->biases != NULL);                                     \
        assert(nn->reg_type != NULL);                                   \
        assert(nn->reg_p != NULL);                                      \
        assert(nn->outputs != NULL);                                    \
        assert(nn->batch_outputs != NULL);                              \
        assert(nn->output == NULL);                                     \
        assert(nn->ones != NULL);                                       \
        assert(nn->g_w == NULL);                                        \
        assert(nn->g_b == NULL);                                        \
        assert(nn->g_out == NULL);                                      \
        assert(nn->g_in == NULL);                                       \
                                                                        \
        const nn_ops_t *op_types = nn->op_types;                        \
        const int *n_dims = nn->n_dims - 1;                             \
        const int *n_weights = nn->n_weights;                           \
        const int *n_biases = nn->n_biases;                             \
        const nn_reg_t *reg_type = nn->reg_type;                        \
        const weight_t *reg_p = nn->reg_p;                              \
                                                                        \
        const nn_ops_t *exp_op_types = e1;                              \
        const int *exp_n_dims = e2;                                     \
        const int *exp_n_weights = e3;                                  \
        const int *exp_n_biases = e4;                                   \
        const nn_reg_t *exp_reg_type = e5;                              \
        const weight_t *exp_reg_p = e6;                                 \
                                                                        \
        puts("\nTest " __to_str_exp(__COUNTER__));                      \
        __exp_check_d("op_types   ", n, op_types);                      \
        __exp_check_d("n_dims     ", n + 1, n_dims);                    \
        __exp_check_d("n_weights  ", n, n_weights);                     \
        __exp_check_d("n_biases   ", n, n_biases);                      \
        __exp_check_d("reg_type   ", n, reg_type);                      \
        __exp_check_lf("reg_p      ", n, reg_p, 1e-50);                 \
                                                                        \
        weight_t *w_p = nn->weights_ptr, *b_p = nn->biases_ptr;         \
        for (int i = 0; i < n; i++) {                                   \
            assert(nn->outputs[i] != NULL);                             \
            assert(nn->batch_outputs[i] == NULL);                       \
            if (nn->n_weights[i] == 0) assert(nn->weights[i] == NULL);  \
            else assert(nn->weights[i] == w_p);                         \
            if (nn->n_biases[i] == 0) assert(nn->biases[i] == NULL);    \
            else assert(nn->biases[i] == b_p);                          \
            w_p += nn->n_weights[i], b_p += nn->n_biases[i];            \
        }                                                               \
        nn_destroy(nn);                                                 \
    } while (0)


int main ()
{
    __title("nn/nn_create");


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(784),
            dense_layer(512, b, relu, l1(0.02)),
            dense_layer(256, b, relu, l2(0.01)),
            dense_layer(10, b, logistic),
            output_layer()
        }),
        6, 784, 10, 535040, 778,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP, DENSE_OP, RELU_OP,
                       DENSE_OP, LOGISTIC_OP}),
        ((int[])      {784, 512, 512, 256, 256, 10, 10}),
        ((int[])      {401408, 0, 131072, 0, 2560, 0}),
        ((int[])      {512, 0, 256, 0, 10, 0}),
        ((nn_reg_t[]) {L1, NONE, L2, NONE, NONE, NONE}),
        ((weight_t[]) {0.02, 0.0, 0.01, 0.0, 0.0, 0.0})
    );


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(4),
            dense_layer(4, b, relu, l1(0.7)),
            output_layer()
        }),
        2, 4, 4, 16, 4,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP}),
        ((int[])      {4, 4, 4}),
        ((int[])      {16, 0}),
        ((int[])      {4, 0}),
        ((nn_reg_t[]) {L1, NONE}),
        ((weight_t[]) {0.7, 0.0})
    );


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(32),
            dense_layer(64, b, linear),
            dense_layer(32, x, linear),
            dense_layer(16, x, linear, l2(0.03)),
            dense_layer(8, x, linear, l1(0.07)),
            dense_layer(4, b, logistic),
            output_layer()
        }),
        6, 32, 4, 4768, 68,
        ((nn_ops_t[]) {DENSE_OP, DENSE_OP, DENSE_OP,
                       DENSE_OP, DENSE_OP, LOGISTIC_OP}),
        ((int[])      {32, 64, 32, 16, 8, 4, 4}),
        ((int[])      {2048, 2048, 512, 128, 32, 0}),
        ((int[])      {64, 0, 0, 0, 4, 0}),
        ((nn_reg_t[]) {NONE, NONE, L2, L1, NONE, NONE}),
        ((weight_t[]) {0.0, 0.0, 0.03, 0.07, 0.0, 0.0})
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
        ((int[])      {0, 0, 0, 0}),
        ((nn_reg_t[]) {NONE, NONE, NONE, NONE}),
        ((weight_t[]) {0.0, 0.0, 0.0, 0.0})
    );


    nn_create_test(
        ((nn_spec_t[]){
            input_layer(2),
            dense_layer(2, b, relu),
            dense_layer(2, b, logistic),
            output_layer()
        }),
        4, 2, 2, 8, 4,
        ((nn_ops_t[]) {DENSE_OP, RELU_OP, DENSE_OP, LOGISTIC_OP}),
        ((int[])      {2, 2, 2, 2, 2}),
        ((int[])      {4, 0, 4, 0}),
        ((int[])      {2, 0, 2, 0}),
        ((nn_reg_t[]) {NONE, NONE, NONE, NONE}),
        ((weight_t[]) {0.0, 0.0, 0.0, 0.0})
    );


    return 0;
}
