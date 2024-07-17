#ifndef _NN_H
#define _NN_H

#include <stdint.h>


#ifndef _STANDARD_ML_TYPES_
#define _STANDARD_ML_TYPES_
typedef double weight_t;    // type used for weights
typedef double value_t;     // type used for intermediate values in the network
typedef double grad_t;      // type used for gradients
typedef int dim_t[2];       // dimensions of matrix (i,j)
typedef int dim3_t[3];      // dimensions of matrix (i,j) including batch size k
#endif // _STANDARD_ML_TYPES_


/*  Types of functions for forward/backward operations */
typedef enum {
    NO_OP,
    DENSE_OP,
    RELU_OP,
    LOGISTIC_OP
} nn_ops_t;


/*  Types of layers in the neural network specification */
typedef enum nn_spec_layer {
    OUTPUT,
    INPUT,
    DENSE,
    RELU = 100,
    LOGISTIC
} nn_spec_layer_t;


/*  Types of layer in the neural network structure 
    These correspond with forward and backward functions */
typedef enum nn_layer {
    DENSE_L,
    RELU_L,
    LOGISTIC_L
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
    RELU_ACTIV = 100,
    LOGISTIC_ACTIV
} nn_activ_t;


#define nn_bias_b 1
#define nn_bias_x 0

#define nn_reg_
#define nn_reg_l2(p) , .reg = L2, .reg_p = (p)
#define nn_reg_l1(p) , .reg = L1, .reg_p = (p)

#define nn_activ_linear LINEAR_ACTIV
#define nn_activ_relu RELU_ACTIV
#define nn_activ_logistic LOGISTIC_ACTIV


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
#define logistic_layer() {              \
        .type = LOGISTIC,               \
        .activ = LINEAR_ACTIV           \
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

    int total_weights;
    int *n_weights;
    weight_t *weights_ptr;
    weight_t **weights;
    int total_biases;
    int *n_biases;
    weight_t *biases_ptr;
    weight_t **biases;

    nn_reg_t *reg_type;
    weight_t *reg_p;

    value_t **outputs;
    value_t *output;

    /* The arrays below are allocated and freed by train */
    value_t **batch_outputs;
    value_t *ones;
    value_t *g_w;
    value_t *g_b;
    value_t *g_out;
    value_t *g_in;
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
void logistic_forward(int d, const value_t *x, value_t *y);
void logistic_backward(int d, const value_t *y, const grad_t *g_y, grad_t *g_x);


/*  nn/nn_create.c declarations */
#define nn_create(spec) _nn_create(spec, __FILE__, __LINE__)
nn_struct_t *_nn_create(nn_spec_t *spec, const char *file, int line);


/*  nn/nn_destroy.c declarations */
#define nn_destroy(nn) _nn_destroy(nn, __FILE__, __LINE__)
void _nn_destroy(nn_struct_t *nn, const char *file, int line);


/*  nn/nn_forward_pass.c declarations */
void nn_forward_pass(nn_struct_t *nn, value_t *input);
void nn_batch_forward_pass(nn_struct_t *nn, int batch_size);


/*  nn/nn_backward_pass.c declarations */
void nn_batch_backward_pass(nn_struct_t *nn, int batch_size);


/*  nn/nn_train.c declarations */
void nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size,
    value_t *x, value_t *t);


/*  nn/nn_test.c declarations */
double nn_test(nn_struct_t *nn, int test_size, value_t *x, value_t *t);


/*  loss.c declarations */
void loss_diff_grad(int d, const value_t *y, const value_t *t, value_t *grad);


#endif // _NN_H
