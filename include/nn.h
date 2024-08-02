#ifndef _NN_H
#define _NN_H

#include <stdint.h>
#include "core/ml_types.h"
#include "opt.h"


/*  Macro to define activation functions for enums */
#define ACTIVATIONS_DEF(x)  \
    RELU##x = 100,          \
    LRELU##x,               \
    LOGISTIC##x,            \
    TANH##x


/*  Types of functions for forward/backward operations */
typedef enum {
    NO_OP,
    DENSE_OP,
    ACTIVATIONS_DEF(_OP)
} nn_ops_t;


/*  Types of layers in the neural network specification */
typedef enum nn_spec_layer {
    OUTPUT,
    INPUT,
    DENSE,
    ACTIVATIONS_DEF()
} nn_spec_layer_t;


/*  Types of layer in the neural network structure 
    These correspond with forward and backward functions */
typedef enum nn_layer {
    DENSE_L,
    ACTIVATIONS_DEF(_L)
} nn_layer_t;


/*  Types of regularization for the weights */
typedef enum nn_reg {
    NONE = 0,
    L1,
    L2
} nn_reg_t;


/*  Types of activation functions */
typedef enum nn_activ {
    LINEAR_ACTIV,
    ACTIVATIONS_DEF(_ACTIV)
} nn_activ_t;


#define nn_bias_b 1
#define nn_bias_x 0

#define nn_reg_
#define nn_reg_l2(p) , .reg = L2, .reg_p = (p)
#define nn_reg_l1(p) , .reg = L1, .reg_p = (p)

#define nn_activ_linear     LINEAR_ACTIV
#define nn_activ_relu       RELU_ACTIV
#define nn_activ_lrelu      LRELU_ACTIV
#define nn_activ_logistic   LOGISTIC_ACTIV
#define nn_activ_tanh       TANH_ACTIV


/*  Macros that create the corresponding nn_spec_t */
#define output_layer() {                \
        .type = OUTPUT,                 \
        .activ = LINEAR_ACTIV           \
    }
#define input_layer(n) {                \
        .type = INPUT,                  \
        .dims = (n),                    \
        .activ = LINEAR_ACTIV           \
    }
/*  reg defaults to 0 (NONE) */
#define dense_layer(n, b, act, ...) {   \
        .type = DENSE,                  \
        .dims = (n),                    \
        .bias = nn_bias_##b,            \
        .activ = nn_activ_##act         \
        nn_reg_##__VA_ARGS__            \
    }
#define relu_layer() {                  \
        .type = RELU,                   \
        .activ = LINEAR_ACTIV           \
    }
#define lrelu_layer() {                 \
        .type = LRELU,                  \
        .activ = LRELU_ACTIV            \
    }
#define logistic_layer() {              \
        .type = LOGISTIC,               \
        .activ = LINEAR_ACTIV           \
    }
#define tanh_layer() {                  \
        .type = TANH,                   \
        .activ = TANH_ACTIV             \
    }


/*  Struct for the specification of a neural network layer */
typedef struct {
    nn_spec_layer_t type;   // type of layer
    int dims;               // number of neurons
    int bias;               // if it has bias (bool)
    nn_activ_t activ;       // activation function
    nn_reg_t reg;           // type of regularization
    weight_t reg_p;         // regularization parameter
} nn_spec_t;


/*  Neural Network structure */
typedef struct {
    int n_layers;
    int input_dims;
    int output_dims;
    nn_ops_t *op_types;
    int *n_dims;

    /*  Model general tunable parameters */
    weight_t learning_rate;
    int stochastic;
    ml_opt_t opt;


    /*  Weights and Biases pointers */
    int total_weights;
    int *n_weights;
    weight_t *weights_ptr;
    weight_t **weights;
    int total_biases;
    int *n_biases;
    weight_t *biases_ptr;
    weight_t **biases;

    /*  Regularization types and parameters */
    nn_reg_t *reg_type;
    weight_t *reg_p;

    /*  Intermediate values and outputs.
        Ones aid in computation.        */
    int k;
    value_t **batch_outputs;
    value_t **outputs;
    value_t *output;
    int ones_n;
    value_t *ones;

    /*  Gradients for weights, biases
        and intermediate values         */
    grad_t *gw_ptr;
    grad_t **gw;
    grad_t *gb_ptr;
    grad_t **gb;
    int g_k;
    int go_n;
    grad_t *g_out;
    grad_t *g_in;

    /*  Manage memory outside of the struct
        0-1: store the addresses of optimizer arrays */
#   define NN_ADDRK_SIZE 2
    void *addr_keeper[NN_ADDRK_SIZE];
} nn_struct_t;


/*  layers/dense.c declarations */
void dense_forward(cdim_t d, cvrp_t x, cwrp_t w,
    int hb, cwrp_t b, vrp_t y);
void dense_backward(cdim_t d, cvrp_t x, cwrp_t w,
    int cx, cgrp_t Gy, grp_t Gx, grp_t Gw);
void batch_dense_forward(cdim3_t d, cvrp_t x, cwrp_t w,
    int hb, cwrp_t b, cvrp_t ones, vrp_t y);
void batch_dense_backward(cdim3_t d, cvrp_t x, cwrp_t w,
    int cx, cvrp_t ones, cgrp_t Gy, grp_t Gx, grp_t Gw, int hb, grp_t Gb);


/*  activations.c declarations */
void relu_forward(int d, cvrp_t x, vrp_t y);
void relu_backward(int d, cvrp_t x, cgrp_t g_y, grp_t g_x);
void lrelu_forward(int d, cvrp_t x, vrp_t y);
void lrelu_backward(int d, cvrp_t x, cgrp_t g_y, grp_t g_x);
void logistic_forward(int d, cvrp_t x, vrp_t y);
void logistic_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x);
void tanh_forward(int d, cvrp_t x, vrp_t y);
void tanh_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x);


/*  loss.c declarations */
void loss_diff_grad(int d, cvrp_t y, cvrp_t t, vrp_t grad);
value_t loss_mse(int n, cvrp_t y, cvrp_t t);


/*  Macros for nn/ functions */
#define nn_create(spec) _nn_create  (spec, __FILE__, __LINE__)
#define nn_destroy(nn) _nn_destroy(nn, __FILE__, __LINE__)
#define nn_train(nn, e, b, s, x, t) _nn_train(nn, e, b, s, x, t, __FILE__, __LINE__)
#define nn_accuracy(nn, s, x, t) _nn_accuracy(nn, s, x, t, __FILE__, __LINE__)
#define nn_predict(nn, k, x, o) _nn_predict(nn, k, x, o, __FILE__, __LINE__)
#define nn_loss(nn, k, x, t) _nn_loss(nn, k, x, t, __FILE__, __LINE__)


/*  nn/ declarations */
nn_struct_t *_nn_create(nn_spec_t *spec, const char *file, int line);
void _nn_destroy(nn_struct_t *nn, const char *file, int line);
void _nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size,
    value_t *x, value_t *t, const char *file, int line);
float _nn_accuracy(nn_struct_t *nn, int size, value_t *x, value_t *t,
    const char *file, int line);
void _nn_predict(nn_struct_t *nn, int k, const value_t *input, value_t *output,
    const char *file, int line);
value_t _nn_loss(nn_struct_t *nn, int k, const value_t *x, const value_t *t,
    const char *file, int line);


// TODO: move to internal
void nn_forward_pass(nn_struct_t *nn);
void nn_batch_forward_pass(nn_struct_t *nn, int batch_size);
void nn_batch_backward_pass(nn_struct_t *nn, int batch_size);


#endif // _NN_H
