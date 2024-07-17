#include <stdlib.h>
#include <stdio.h>
#include "../../include/nn.h"

#ifndef __x86_64__
#   include <time.h>
#endif


#define _nn_create_error(cond, str, ...)                            \
    if (cond) {                                                     \
        fprintf(stderr, "\e[1;39mnn_create\e[0;39m"                 \
            " (from \e[1;39m%s:%d\e[0;39m) \e[1;31merror\e[0;39m: " \
            str "\n", file, line, ##__VA_ARGS__);                   \
        exit(1);                                                    \
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


/*  Helper function to randomize the starting weights */
void _nn_rand_weights(nn_struct_t *nn)
{
    int i, rand_2 = RAND_MAX >> 1;

    #ifdef __x86_64__
        uint32_t lo, hi;
        __asm__ __volatile__ (
            "rdtsc" : "=a" (lo), "=d" (hi)
        );
        srand(((uint64_t) hi << 32) | lo);
    #else
        srand(time(NULL));
    #endif

    for (i = 0; i < nn->total_weights; i++)
        nn->weights_ptr[i] = (rand() / (weight_t) rand_2) - 1.0;
}


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

    _nn_alloc_check(reg_type, nn_reg_t);
    _nn_alloc_check(reg_p, weight_t);

    /*  nn->outputs[-1] would point to the input for ease */
    _nn_alloc_check2(outputs, value_t *, m, 1);
    nn->outputs++;

    /*  batch_outputs won't be initialized with nn_create
        so we initialize them to 0 (NULL pointers)        */
    _nn_alloc_check2(batch_outputs, value_t *, c, 1);
    nn->batch_outputs++;
}


/*  Helper function to create the layers in the neural network struct */
void _nn_create_layers(nn_struct_t *nn,
    nn_spec_t *spec, const char *file, int line)
{
    int i, j, dims, layer_type;

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
            case LOGISTIC:
                activation_template(LOGISTIC);
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

    for (total_w = total_b = i = 0; i < nn->n_layers; i++)
        total_w += nn->n_weights[i], total_b += nn->n_biases[i];

    _nn_create_error(total_w <= 0, "invalid number of weights");

    nn->total_weights = total_w;
    nn->total_biases = total_b;
    nn->biases_ptr = NULL;
    nn->weights_ptr = NULL;

    nn->weights_ptr = malloc(total_w * sizeof(weight_t));
    if (total_b > 0) nn->biases_ptr = calloc(total_b, sizeof(weight_t));

    _nn_create_error(nn->weights_ptr == NULL,
        "failed to allocate memory for weights, "
        "N = %d", total_w);
    _nn_create_error(total_b > 0 && nn->biases_ptr == NULL,
        "failed to allocate memory for biases, "
        "N = %d", total_b);

    _nn_rand_weights(nn);

    for (i = i_w = i_b = 0; i < nn->n_layers; i++) {
        if (nn->n_weights[i] == 0) {
            nn->weights[i] = NULL;
        } else {
            nn->weights[i] = nn->weights_ptr + i_w;
            i_w += nn->n_weights[i];
        }

        if (nn->n_biases[i] == 0) {
            nn->biases[i] = NULL;
        } else {
            nn->biases[i] = nn->biases_ptr + i_b;
            i_b += nn->n_biases[i];
        }
    }
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

    nn->output = NULL;
    nn->ones = NULL;
    nn->g_w = NULL;
    nn->g_b = NULL;
    nn->g_out = NULL;
    nn->g_in = NULL;
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

    /* Default values: */
    nn->learning_rate = 0.01;

    return nn;
}
