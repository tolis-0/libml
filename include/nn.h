#ifndef _ML_NN_H
#define _ML_NN_H


#include "core/ml_types.h"
#include "opt.h"
#include "error.h"


/*
 * X-Macro helper for activation functions without parameters
 */
#define ML_ACTIVATION_FUNCTIONS_DECLARATIONS        \
   /* Arguments:                                    \
    * 1) name in lowercase                          \
    * 2) name in uppercase                          \
    * 3) requires input x for derivative            \
    * 4) requires output y for derivative */        \
    X(relu,     RELU,       1,  0)                  \
    X(lrelu,    LRELU,      1,  0)                  \
    X(logistic, LOGISTIC,   0,  1)                  \
    X(tanh,     TANH,       0,  1)


/*
 * Types of forward/backward operations
 */
typedef enum {
    EMPTY_OP,
#define X(_, NAME, ...) NAME##_OP,
    ML_ACTIVATION_FUNCTIONS_DECLARATIONS
#undef X
    DENSE_OP
} nn_op_t;

static inline _Bool _nn_is_activ_op(nn_op_t op)
{
    return (op >= RELU_OP && op < DENSE_OP);
}

/*
 * Types of layers in the neural network specification
 */
typedef enum {
    OUTPUT_SPEC,
    INPUT_SPEC,
#define X(_, NAME, ...) NAME##_SPEC,
    ML_ACTIVATION_FUNCTIONS_DECLARATIONS
#undef X
    DENSE_SPEC
} nn_specl_t;

static inline nn_op_t _nn_spec_to_op(nn_specl_t spec)
{
    return spec + (RELU_OP - RELU_SPEC);
}

/*
 * Regularization of neural network layers
 */
typedef struct {
    enum {
        ZERO_REG,
        L1_REG,
        L2_REG
        // TODO: L1L2_REG
    } type;
    weight_t p1;
    weight_t p2;
} nn_reg_t;

/*
 * Use these to define regularization of a layer
 */
#define NO_REG ((nn_reg_t) {.type = ZERO_REG})

static inline nn_reg_t reg_l1(weight_t p)
{
    return ((nn_reg_t) {.type = L1_REG, .p1 = p});
}

static inline nn_reg_t reg_l2(weight_t p)
{
    return ((nn_reg_t) {.type = L2_REG, .p2 = p});
}


/*
 * Struct for the specification of a neural network layer
 */
typedef struct {
    nn_specl_t type;
    nn_op_t activ;
    int dims;
    nn_reg_t reg;
    _Bool hb;
} nn_spec_t;

#define NN_SPEC_END ((nn_spec_t){.type = OUTPUT_SPEC})

/*
 * Use nnl_<type> to specify neural network layers
 * nnl_input is always and only at the start
 */
static inline nn_spec_t nnl_input(int input_dims)
{
    return (nn_spec_t) {
        .type   = INPUT_SPEC,
        .activ  = EMPTY_OP,
        .dims   = input_dims,
        .reg    = NO_REG,
        .hb     = 0
    };
}

static inline nn_spec_t nnl_dense(int dims, _Bool hb, nn_op_t op, nn_reg_t reg)
{
    return (nn_spec_t) {
        .type   = DENSE_SPEC,
        .activ  = op,
        .dims   = dims,
        .hb     = hb,
        .reg    = reg
    };
}

/*
 * X-Macro to automate constructor functions
 * for activation function layers
 */
#define X(name, NAME, ...)                  \
static inline nn_spec_t nnl_##name(void)    \
{                                           \
    return (nn_spec_t) {                    \
        .type   = NAME##_SPEC,              \
        .activ  = EMPTY_OP                  \
    };                                      \
}
ML_ACTIVATION_FUNCTIONS_DECLARATIONS
#undef X

/*
 * Substructure for weights
 */
typedef struct {
    int total;              /* total weights */
    int *num;               /* number of weights per layer */
    weight_t *ptr;          /* pointer to all weights */
    weight_t **of_layer;    /* pointers to weights of layer i */
} nn_weights_t;

/*
 * Substructure for intermediate values
 */
typedef struct {
    int batch_size;     /* memory in outputs is allocated for this batch size */
#   define ONES_SIZE_DEFAULT 16
    int ones_size;      /* size of the ones array */
    value_t **outputs;
    value_t *ones;
} nn_intermv_t;

/*
 * Substructure for gradients
 */
typedef struct {
    int batch_size;     /* memory in gradients is allocated for this batch size */
    grad_t *ptr;
    grad_t **weights;
    grad_t **biases;
    grad_t *out;
    grad_t *in;
} nn_gradv_t;

/*
 * Neural Network structure
 */
typedef struct {
    int num_of_layers;
    int max_dims;
    int *num_of_dims;
    nn_op_t  *operation_type;
    nn_reg_t *regularization;

    /* Model substructs */
    nn_weights_t weights;
    nn_weights_t biases;
    nn_intermv_t interm;
    nn_gradv_t grad;

    /* Hyperparameters configurable by the user */
    weight_t learning_rate;
    ml_opt_t optimizer;
    _Bool stochastic;

    /* Other settings */
    unsigned int seed;

    /* Manage memory outside of the struct
     * 0-1: store the addresses of optimizer arrays */
#   define MEM_ADDR_SIZE 2
    void *mem_addr[MEM_ADDR_SIZE];
} nn_struct_t;

/*
 * Macros to access some common struct elements and properties
 */
#define NN_INPUT(nn)        ((nn)->interm.outputs[-1])
#define NN_OUTPUT(nn)       ((nn)->interm.outputs[(nn)->num_of_layers - 1])
#define NN_INPUT_DIMS(nn)   ((nn)->num_of_dims[-1])
#define NN_OUTPUT_DIMS(nn)  ((nn)->num_of_dims[(nn)->num_of_layers - 1])


/* layers/dense.c declarations */
void dense_forward(cdim_t d, cvrp_t x, cwrp_t w, _Bool hb, cwrp_t b, vrp_t y);
void dense_backward(cdim_t d, cvrp_t x, cwrp_t w, _Bool cx, cgrp_t Gy, grp_t Gx, grp_t Gw);
void batch_dense_forward(cdim3_t d, cvrp_t x, cwrp_t w, _Bool hb, cwrp_t b, cvrp_t ones, vrp_t y);
void batch_dense_backward(cdim3_t d, cvrp_t x, cwrp_t w, _Bool cx, cvrp_t ones, cgrp_t Gy, grp_t Gx, grp_t Gw, _Bool hb, grp_t Gb);


/* layers/activations/ declarations */
void relu_forward(int d, cvrp_t x, vrp_t y);
void relu_backward(int d, cvrp_t x, cgrp_t g_y, grp_t g_x);
void lrelu_forward(int d, cvrp_t x, vrp_t y);
void lrelu_backward(int d, cvrp_t x, cgrp_t g_y, grp_t g_x);
void logistic_forward(int d, cvrp_t x, vrp_t y);
void logistic_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x);
void tanh_forward(int d, cvrp_t x, vrp_t y);
void tanh_backward(int d, cvrp_t y, cgrp_t g_y, grp_t g_x);


/* loss.c declarations */
void loss_diff_grad(int d, cvrp_t y, cvrp_t t, vrp_t grad);
value_t loss_mse(int n, cvrp_t y, cvrp_t t);


/* Macros for nn/ functions */
#define nn_create(args...)      (__ml_error_update(nn_create),   _nn_create(args))
#define nn_destroy(args...)     (__ml_error_update(nn_destroy),  _nn_destroy(args))
#define nn_train(args...)       (__ml_error_update(nn_train),    _nn_train(args))
#define nn_accuracy(args...)    (__ml_error_update(nn_accuracy), _nn_accuracy(args))
#define nn_predict(args...)     (__ml_error_update(nn_predict),  _nn_predict(args))
#define nn_loss(args...)        (__ml_error_update(nn_loss),     _nn_loss(args))


/* nn/ declarations */
nn_struct_t *_nn_create(nn_spec_t *spec);
void _nn_destroy(nn_struct_t *nn);
void _nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size, value_t *x, value_t *t);
float _nn_accuracy(nn_struct_t *nn, int size, value_t *x, value_t *t);
void _nn_predict(nn_struct_t *nn, int batch_size, const value_t *input, value_t *output);
value_t _nn_loss(nn_struct_t *nn, int batch_size, const value_t *x, const value_t *t);


#endif // _ML_NN_H
