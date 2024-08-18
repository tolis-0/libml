#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*
 * Helper function to measure number of layers in the network
 * and run some initial structure checks
 */
static void _nn_measure_layers(nn_struct_t *nn, nn_spec_t *spec)
{
    int i, nl;

    for (i = 0, nl = -1; spec[i].type != OUTPUT_SPEC; i++, nl++)
        if (spec[i].activ != EMPTY_OP) nl++;

    __ml_assert(nl > 0, "invalid number of layers");
    __ml_assert(spec[0].type == INPUT_SPEC, "missing input on neural network");

    nn->num_of_layers = nl;
}


/*
 * Helper function to allocate memory for most of the network fields
 */
static void _nn_alloc(nn_struct_t *nn)
{
    const int nl = nn->num_of_layers;

    /* nn->num_of_dims[-1] would be input dimensions for ease */
    __ml_malloc_check(nn->num_of_dims, int, nl + 1);
    nn->num_of_dims++;
    __ml_malloc_check(nn->operation_type, nn_op_t, nl);

    __ml_malloc_check(nn->weights.num, int, nl);
    __ml_malloc_check(nn->biases.num, int, nl);

    /* intialize double pointers to NULL */
    __ml_calloc_check(nn->weights.of_layer, weight_t *, nl);
    __ml_calloc_check(nn->biases.of_layer, weight_t *, nl);
    __ml_calloc_check(nn->grad.weights, grad_t *, nl);
    __ml_calloc_check(nn->grad.biases, grad_t *, nl);

    __ml_malloc_check(nn->regularization, nn_reg_t, nl);

    /* nn->interm.outputs[-1] would point to the input
     * also init to NULL */
    __ml_calloc_check(nn->interm.outputs, value_t *, nl + 1);
    nn->interm.outputs++;
}


/* Basic initializations for activation functions */
static void _nn_activ_layer_init(nn_struct_t *nn, nn_op_t op, int dims, int j)
{
    nn->operation_type[j] = op,
    nn->num_of_dims[j] = dims,
    nn->weights.num[j] = 0,
    nn->biases.num[j] = 0,
    nn->weights.of_layer[j] = NULL,
    nn->biases.of_layer[j] = NULL,
    nn->regularization[j] = NO_REG;
}

/*
 * Helper function to create the layers in the neural network struct
 */
static void _nn_create_layers(nn_struct_t *nn, nn_spec_t *spec)
{
    nn_specl_t layer_type;
    int i, j, dims, max_d;

    nn->num_of_dims[-1] = dims = spec[0].dims;
    max_d = 0;

    __ml_assert(dims > 0, "invalid number of input dimensions: %d", dims);

    /*
     * i: layer specified in nn_spec
     * j: layer in nn_struct (counts activation functions)
     * dims: keeps count of current dimensions (useful for
     * layers that keep current number of dims) e.g. nnl_relu()
     * We know that spec[0].type is INPUT_SPEC so we skip it
     */
    for (j = 0, i = 1; spec[i].type != OUTPUT_SPEC; j++, i++) {
        layer_type = spec[i].type;
        max_d = (dims > max_d) ? dims : max_d;
        switch (layer_type) {
            case DENSE_SPEC:
                nn->operation_type[j] = DENSE_OP;
                /* previous dims * current dims */
                nn->weights.num[j] = dims * spec[i].dims;
                /* update dims */
                nn->num_of_dims[j] = dims = spec[i].dims;
                nn->biases.num[j] = spec[i].hb ? dims : 0;
                nn->regularization[j] = spec[i].reg;
                /* handle activation functions of dense layer */
                if (spec[i].activ != EMPTY_OP) {
                    __ml_assert(_nn_is_activ_op(spec[i].activ),
                        "activation function of spec layer %d is not valid", i);
                    _nn_activ_layer_init(nn, spec[i].activ, dims, ++j);
                }
                break;

            /*
             * Using X-macro for cases of activation functions
             */
#           define X(_, NAME, ...)                              \
            case NAME##_SPEC:                                   \
                _nn_activ_layer_init(nn, NAME##_OP, dims, j);   \
                break;
            ML_ACTIVATION_FUNCTIONS_DECLARATIONS
#           undef X

            case INPUT_SPEC:
                _ml_throw_error("input layer in the middle of the network");
                break;
            default:
                _ml_throw_error("operation type of spec layer %d is not valid", spec[i].type);
        }
    }

    nn->max_dims = max_d;
}


/*
 * Helper function to create the weights and biases
 */
static void _nn_create_weights(nn_struct_t *nn)
{
    int i, total_w, total_b, i_w, i_b;

    for (total_w = total_b = i = 0; i < nn->num_of_layers; i++) {
        total_w += nn->weights.num[i];
        total_b += nn->biases.num[i];
    }

    __ml_assert(total_w > 0, "the network has an invalid number of weights");
    __ml_assert(total_b >= 0, "the network has an invalid number of biases");

    nn->weights.total = total_w;
    nn->biases.total = total_b;

    /*
     * nn->weights.ptr has all the weights trainable
     * directly by the optimizers as a single array
     * including the biases which start after the weights
     */
    __ml_malloc_check(nn->weights.ptr, weight_t, total_w + total_b);

    if (total_b > 0) {
        nn->biases.ptr = nn->weights.ptr + total_w;
        memset(nn->biases.ptr, 0, total_b * sizeof(weight_t));
    } else {
        nn->biases.ptr = NULL;
    }

    for (i = i_w = i_b = 0; i < nn->num_of_layers; i++) {
        const int w_n = nn->weights.num[i];
        const int b_n = nn->biases.num[i];

        nn->weights.of_layer[i] = (w_n > 0) ? (nn->weights.ptr + i_w) : NULL;
        nn->biases.of_layer[i] = (b_n > 0) ? (nn->biases.ptr + i_b) : NULL;
        i_w += w_n;
        i_b += b_n;
    }

    _nn_rand_weights(nn);
}


/*
 * Helper function to allocate memory for ones array
 */
static void _nn_alloc_ones(nn_intermv_t *nni)
{
    nni->batch_size = 0;
    nni->ones_size = ONES_SIZE_DEFAULT;

    __ml_malloc_check(nni->ones, value_t, nni->ones_size);

    for (int i = 0; i < nni->ones_size; i++)
        nni->ones[i] = __ml_fpc(1.0);
}


/*
 * Helper function to allocate memory for gradients
 */
static void _nn_grad_init(nn_struct_t *nn)
{
    int i, i_w, i_b;
    const int total_w = nn->weights.total;
    const int total_b = nn->biases.total;
    grad_t *restrict gb_ptr;

    nn->grad.batch_size = 0;
    nn->grad.out = NULL;
    nn->grad.in = NULL;

    __ml_malloc_check(nn->grad.ptr, grad_t, total_w + total_b);

    gb_ptr = (total_b > 0) ? nn->grad.ptr + total_w : NULL;

    for (i = i_w = i_b = 0; i < nn->num_of_layers; i++) {
        const int w_n = nn->weights.num[i];
        const int b_n = nn->biases.num[i];

        nn->grad.weights[i] = (w_n > 0) ? (nn->grad.ptr + i_w) : NULL;
        nn->grad.biases[i] = (b_n > 0) ? (gb_ptr + i_b) : NULL;
        i_w += w_n;
        i_b += b_n;
    }
}


/*
 * Main function that creates the neural network struct
 */
nn_struct_t *_nn_create(nn_spec_t *spec)
{
    nn_struct_t *nn;
    __ml_malloc_check(nn, nn_struct_t, 1);

    _nn_measure_layers(nn, spec);
    _nn_alloc(nn);
    _nn_create_layers(nn, spec);
    _nn_create_weights(nn);
    _nn_alloc_ones(&nn->interm);
    _nn_grad_init(nn);

    for (int i = 0; i < MEM_ADDR_SIZE; i++)
        nn->mem_addr[i] = NULL;

    /* Default hyperparameter values: */
    nn->learning_rate   = __ml_fpc(0.01);
    nn->stochastic      = 1;
    nn->optimizer       = opt_create.gd();

    return nn;
}
