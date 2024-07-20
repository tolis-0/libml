#include <stdlib.h>
#include "../../include/nn.h"


/*  Allocates memory for the batch intermediate values and outputs.
    Also for ones which aids in computation. */
void _nn_alloc_batch(nn_struct_t *nn, int batch_size)
{
    for (int i = 0; i < nn->n_layers; i++) {
        nn->batch_outputs[i] = malloc(
            nn->n_dims[i] * batch_size * sizeof(value_t));
    }

    nn->k = batch_size;

    nn->ones = malloc(batch_size * sizeof(value_t));
    
    for (int i = 0; i < batch_size; i++) {
        nn->ones[i] = 1.0;
    }
}


/*  Reallocates new memory for the batches */
void _nn_realloc_batch(nn_struct_t *nn, int batch_size)
{
    int old_batch_size = nn->k;

    for (int i = 0; i < nn->n_layers; i++) {
        nn->batch_outputs[i] = realloc(nn->batch_outputs[i],
            nn->n_dims[i] * batch_size * sizeof(value_t));
    }

    nn->k = batch_size;

    nn->ones = realloc(nn->ones, batch_size * sizeof(value_t));

    for (int i = old_batch_size; i < batch_size; i++) {
        nn->ones[i] = 1.0;
    }
}


/*  Free memory from the batch */
void _nn_free_batch(nn_struct_t *nn)
{
    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->batch_outputs[i]);
        nn->batch_outputs[i] = NULL;
    }

    nn->batch_outputs[-1] = NULL;

    free(nn->ones);

    nn->ones = NULL;
}


/*  Allocate memory for gradients */
void _nn_alloc_grad(nn_struct_t *nn)
{
    int max_dims, max_w, max_b, batch_size;

    max_b = max_w = 0;
    max_dims = nn->input_dims;
    batch_size = nn->k;

    for (int i = 0; i < nn->n_layers; i++) {
        max_dims = (nn->n_dims[i] > max_dims) ? nn->n_dims[i] : max_dims;
        max_b = (nn->n_biases[i] > max_b) ? nn->n_biases[i] : max_b;
        max_w = (nn->n_weights[i] > max_w) ? nn->n_weights[i] : max_w;
    }

    nn->g_w = malloc(max_w * sizeof(weight_t));
    nn->g_b = malloc(max_b * sizeof(weight_t));
    nn->g_out = malloc(batch_size * max_dims * sizeof(value_t));
    nn->g_in = malloc(batch_size * max_dims * sizeof(value_t));
}


/*  Reallocates memory for the gradients */
void _nn_realloc_grad(nn_struct_t *nn)
{
    int max_dims, batch_size;

    max_dims = nn->input_dims;
    batch_size = nn->k;

    for (int i = 0; i < nn->n_layers; i++) {
        max_dims = (nn->n_dims[i] > max_dims) ? nn->n_dims[i] : max_dims;
    }

    nn->g_out = realloc(nn->g_out,
        batch_size * max_dims * sizeof(value_t));
    nn->g_in = realloc(nn->g_in,
        batch_size * max_dims * sizeof(value_t));
}


/*  Free memory from gradients */
void _nn_free_grad(nn_struct_t *nn)
{
    free(nn->g_w);
    free(nn->g_b);
    free(nn->g_in);
    free(nn->g_out);

    nn->g_w = NULL;
    nn->g_b = NULL;
    nn->g_out = NULL;
    nn->g_in = NULL;
}
