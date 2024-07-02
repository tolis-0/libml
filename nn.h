#ifndef _NN_H
#define _NN_H

#include <stdint.h>


typedef double weight_t;    // type used for weights
typedef double value_t;     // type used for intermediate values in the network
typedef double grad_t;      // type used for gradients
typedef int dim_t[2];       // dimensions of matrix (i,j)


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
    value_t *g_out;
    value_t *g_in;
} nn_struct_t;


/*  layers.c declarations */
void dense_forward(dim_t d, const value_t *x, const weight_t *W, value_t *y);
void dense_backward(dim_t d, value_t *x, weight_t *W,
    int calc_x, grad_t *Gy, grad_t *Gx, grad_t *GW);


/*  activations.c declarations */
void relu_forward(int d, const value_t *x, value_t *y);
void relu_backward(int d, const value_t *x, const grad_t *g_y, grad_t *g_x);
void logistic_forward(int d, const value_t *x, value_t *y);
void logistic_backward(int d, const value_t *y, const grad_t *g_y, grad_t *g_x);


/*  nn.c declarations */
nn_struct_t *_nn_create(nn_spec_t *spec, const char *file, int line);
void _nn_destroy(nn_struct_t *nn, const char *file, int line);

/*  Macros that provide debugging info */
#define nn_create(spec) _nn_create(spec, __FILE__, __LINE__)
#define nn_destroy(spec) _nn_destroy(spec, __FILE__, __LINE__)


#endif // _NN_H
