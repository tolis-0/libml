#ifndef _NN_H
#define _NN_H

#include <stdint.h>
#include "core/ml_types.h"


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
    weight_t learning_rate;

    nn_ops_t *op_types;
    int *n_dims;

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
    int g_k;
    int gw_n;
    grad_t *g_w;
    int gb_n;
    grad_t *g_b;
    int go_n;
    grad_t *g_out;
    grad_t *g_in;
} nn_struct_t;


/*  layers.c declarations */
void dense_forward(const dim_t d, const value_t *x, const weight_t *W,
    int has_bias, const weight_t *b, value_t *y);
void dense_backward(const dim_t d, const value_t *x, const weight_t *W,
    int calc_x, const grad_t *Gy, grad_t *Gx, grad_t *GW);
void batch_dense_forward(const dim3_t d, const value_t *x, const weight_t *W,
    int has_bias, const weight_t *b, const value_t *ones, value_t *y);
void batch_dense_backward(const dim3_t d, const value_t *x, const weight_t *W,
    int calc_x, const value_t *ones, const grad_t *Gy, grad_t *Gx, grad_t *GW,
    int has_bias, grad_t *Gb);


/*  activations.c declarations */
void relu_forward(int d, const value_t *x, value_t *y);
void relu_backward(int d, const value_t *x, const grad_t *g_y, grad_t *g_x);
void lrelu_forward(int d, const value_t *x, value_t *y);
void lrelu_backward(int d, const value_t *x, const grad_t *g_y, grad_t *g_x);
void logistic_forward(int d, const value_t *x, value_t *y);
void logistic_backward(int d, const value_t *y, const grad_t *g_y, grad_t *g_x);
void tanh_forward(int d, const value_t *x, value_t *y);
void tanh_backward(int d, const value_t *y, const grad_t *g_y, grad_t *g_x);


/*  loss.c declarations */
void loss_diff_grad(int d, const value_t *y, const value_t *t, value_t *grad);
value_t loss_mse(int n, const value_t *y, const value_t *t);


/*  Macros for nn/ functions */
#define nn_create(spec) _nn_create  (spec, __FILE__, __LINE__)
#define nn_destroy(nn) _nn_destroy(nn, __FILE__, __LINE__)
#define nn_train(nn, e, b, s, x, t) _nn_train(nn, e, b, s, x, t, __FILE__, __LINE__)
#define nn_test(nn, s, x, t) _nn_test(nn, s, x, t, __FILE__, __LINE__)
#define nn_predict(o, nn, x) _nn_predict(o, nn, x, __FILE__, __LINE__)
// TODO: remove
#define nn_batch_predict(o, nn, x, k) \
    nn_batch_predict(o, nn, x, k, __FILE__, __LINE__)
#define nn_loss(nn, k, x, t) _nn_loss(nn, k, x, t, __FILE__, __LINE__)


/*  nn/ declarations */
nn_struct_t *_nn_create(nn_spec_t *spec, const char *file, int line);
void _nn_destroy(nn_struct_t *nn, const char *file, int line);
void _nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size,
    value_t *x, value_t *t, const char *file, int line);
float _nn_test(nn_struct_t *nn, int test_size, value_t *x, value_t *t,
    const char *file, int line);
void _nn_predict(value_t *output, nn_struct_t *nn, const value_t *input,
    const char *file, int line);
// TODO: merge with _nn_predict
void _nn_batch_predict(value_t *output, nn_struct_t *nn,
    const value_t *input, int k, const char *file, int line);
value_t _nn_loss(nn_struct_t *nn, int k, const value_t *x, const value_t *t,
    const char *file, int line);

// TODO: move to internal
void nn_forward_pass(nn_struct_t *nn);
void nn_batch_forward_pass(nn_struct_t *nn, int batch_size);
void nn_batch_backward_pass(nn_struct_t *nn, int batch_size);


#endif // _NN_H
