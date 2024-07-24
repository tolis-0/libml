#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "../../include/nn.h"
#include "nn_internal.h"


#define _nn_create_error(cond, str, ...)                            \
    if (__builtin_expect(!!(cond), 0)) {                            \
        fprintf(stderr, "\e[1;39mnn_create\e[0;39m"                 \
            " (from \e[1;39m%s:%d\e[0;39m) \e[1;31merror\e[0;39m: " \
            str "\n", file, line, ##__VA_ARGS__);                   \
        exit(EXIT_FAILURE);                                         \
    }


#define activation_template(op)                                     \
    nn->op_types[j] = op##_OP;                                      \
    nn->n_dims[j] = dims;                                           \
    nn->n_weights[j] = nn->n_biases[j] = 0;                         \
    nn->weights[j] = nn->biases[j] = NULL;                          \
    nn->reg_type[j] = NONE, nn->reg_p[j] = 0;


#define _nn_malloc_trick *
#define _nn_calloc_trick ,

#define _nn_alloc_check(var, type) _nn_alloc_check2(var, type, m, 0)
#define _nn_alloc_check2(var, type, m, k)                           \
    do {                                                            \
        nn->var = m##alloc((nn->n_layers + k)                       \
            _nn_##m##alloc_trick sizeof(type));                     \
        _nn_create_error(nn->var == NULL,                           \
            "failed to allocate memory for " #var);                 \
    } while (0)


/*  Helper function to measure number of layers in the network
    and run some initial structure checks */
void _nn_measure_layers(nn_struct_t *nn,
    nn_spec_t *spec, const char *file, int line)
{
    int i;

    for (i = 0, nn->n_layers = -1; spec[i].type != OUTPUT; i++) {
        if (spec[i].activ != LINEAR_ACTIV) nn->n_layers++;
        nn->n_layers++;
    }

    _nn_create_error(nn->n_layers <= 0, "invalid number of layers");
    _nn_create_error(spec[0].type != INPUT, "missing input on neural network");
}


/*  Helper function to allocate memory for most of the network struct fields */
void _nn_alloc(nn_struct_t *nn, const char *file, int line)
{
    _nn_alloc_check2(n_dims, int, m, 1);
    nn->n_dims++;
    _nn_alloc_check(op_types, nn_ops_t);

    _nn_alloc_check(n_weights, int);
    _nn_alloc_check(n_biases, int);
    _nn_alloc_check(weights, weight_t *);
    _nn_alloc_check(biases, weight_t *);

    _nn_alloc_check(gw, grad_t *);
    _nn_alloc_check(gb, grad_t *);

    _nn_alloc_check(reg_type, nn_reg_t);
    _nn_alloc_check(reg_p, weight_t);

    /*  nn->outputs[-1] would point to the input for ease */
    _nn_alloc_check2(outputs, value_t *, m, 1);
    nn->outputs++;

    /*  batch_outputs won't be initialized with nn_create
        so we initialize them to 0 (NULL pointers)        */
    _nn_alloc_check2(batch_outputs, value_t *, c, 1);
    nn->batch_outputs++;
    nn->k = 0;
}


/*  Helper function to create the layers in the neural network struct */
void _nn_create_layers(nn_struct_t *nn,
    nn_spec_t *spec, const char *file, int line)
{
    nn_spec_layer_t layer_type;
    int i, j, dims;

    nn->input_dims = nn->n_dims[-1] = dims = spec[0].dims;

    _nn_create_error(nn->input_dims <= 0,
        "invalid number of input dimensions: %d", nn->input_dims);

    for (j = 0, i = 1; spec[i].type != OUTPUT; j++, i++) {
        layer_type = spec[i].type;
activation_redo:
        switch (layer_type) {
            case DENSE:
                nn->op_types[j] = DENSE_OP;
                nn->n_weights[j] = dims * spec[i].dims;
                nn->n_dims[j] = dims = spec[i].dims;
                nn->n_biases[j] = spec[i].bias ? dims : 0;
                nn->reg_type[j] = spec[i].reg;
                nn->reg_p[j] = spec[i].reg_p;
                if (spec[i].activ != LINEAR_ACTIV) {
                    layer_type = spec[i].activ;
                    j++;
                    goto activation_redo;
                }
                break;
            case RELU:
                activation_template(RELU);
                break;
            case LRELU:
                activation_template(LRELU);
                break;
            case LOGISTIC:
                activation_template(LOGISTIC);
                break;
            case TANH:
                activation_template(TANH);
                break;
            case INPUT:
                _nn_create_error(1, "input layer in the middle of the network");
                break; // for implicit fallthrough warning
            default:
                _nn_create_error(1, "layer type not recognized: %d", spec[i].type);
        }
    }

    nn->output_dims = dims;
}


/*  Helper function to create the weights and biases */
void _nn_create_weights(nn_struct_t *nn, const char *file, int line)
{
    int i, total_w, total_b, i_w, i_b;

    for (total_w = total_b = i = 0; i < nn->n_layers; i++) {
        total_w += nn->n_weights[i];
        total_b += nn->n_biases[i];
    }

    _nn_create_error(total_w <= 0, "invalid number of weights");

    nn->total_weights = total_w;
    nn->total_biases = total_b;
    nn->biases_ptr = NULL;
    nn->weights_ptr = NULL;

    nn->weights_ptr = malloc((total_w + total_b) * sizeof(weight_t));
    if (total_b > 0) {
        nn->biases_ptr = nn->weights_ptr + total_w;
        memset(nn->biases_ptr, 0, total_b * sizeof(char));
    }

    _nn_create_error(nn->weights_ptr == NULL,
        "failed to allocate memory for weights, "
        "N = %d", (total_w + total_b));

    for (i = i_w = i_b = 0; i < nn->n_layers; i++) {
        const int w_n = nn->n_weights[i];
        const int b_n = nn->n_biases[i];

        nn->weights[i] = (w_n > 0) ? (nn->weights_ptr + i_w) : NULL;
        nn->biases[i] = (b_n > 0) ? (nn->biases_ptr + i_b) : NULL;
        i_w += w_n;
        i_b += b_n;
    }

    _nn_rand_weights(nn);
}


/*  Helper function to allocate memory for intermediate values */
void _nn_alloc_interm(nn_struct_t *nn, const char *file, int line)
{
    int i;

    for (i = 0; i < nn->n_layers; i++) {
        nn->outputs[i] = malloc(nn->n_dims[i] * sizeof(value_t));
        _nn_create_error(nn->outputs[i] == NULL,
            "failed to allocate memory for intermediate values, "
            "number of neurons: %d", nn->n_dims[i]);
    }

    nn->ones_n = 16;
    nn->ones = malloc(nn->ones_n * sizeof(value_t));

    _nn_create_error(nn->ones == NULL,
            "failed to allocate memory for the ones array");

    for (i = 0; i < nn->ones_n; i++) {
        nn->ones[i] = 1.0;
    }

    nn->output = NULL;
}


/*  Helper function to determine values for gradients */
void _nn_grad_vals(nn_struct_t *nn, const char *file, int line)
{
    int i, max_d, i_w, i_b;
    const int total_w = nn->total_weights;
    const int total_b = nn->total_biases;

    max_d = nn->input_dims;

    for (i = 0; i < nn->n_layers; i++) {
        max_d = (nn->n_dims[i] > max_d) ? nn->n_dims[i] : max_d;
    }

    nn->g_k = 0;
    nn->go_n = max_d;

    nn->gw_ptr = NULL;
    nn->gb_ptr = NULL;
    nn->g_out = NULL;
    nn->g_in = NULL;

    nn->gw_ptr = malloc((total_w + total_b) * sizeof(weight_t));
    if (total_b > 0) {
        nn->gb_ptr = nn->gw_ptr + total_w;
    }

    _nn_create_error(nn->gw_ptr == NULL,
        "failed to allocate memory for gw, "
        "N = %d", (total_w + total_b));

    for (i = i_w = i_b = 0; i < nn->n_layers; i++) {
        const int w_n = nn->n_weights[i];
        const int b_n = nn->n_biases[i];

        nn->gw[i] = (w_n > 0) ? (nn->gw_ptr + i_w) : NULL;
        nn->gb[i] = (b_n > 0) ? (nn->gb_ptr + i_b) : NULL;
        i_w += w_n;
        i_b += b_n;
    }
}


/*  Main function that creates the neural network struct */
nn_struct_t *_nn_create(nn_spec_t *spec, const char *file, int line)
{
    nn_struct_t *nn = malloc(sizeof(nn_struct_t));

    _nn_measure_layers(nn, spec, file, line);
    _nn_alloc(nn, file, line);
    _nn_create_layers(nn, spec, file, line);
    _nn_create_weights(nn, file, line);
    _nn_alloc_interm(nn, file, line);
    _nn_grad_vals(nn, file, line);

    /* Default values: */
    nn->learning_rate = 0.01;

    return nn;
}
