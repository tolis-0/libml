#include <stdlib.h>
#include "../../include/nn.h"


/*  Allocates memory for the batch intermediate values and outputs */
void _nn_alloc_batch(nn_struct_t *nn, int batch_size)
{
    const int bsize = batch_size * sizeof(value_t);
    int i;

    if (nn->k == 0) {
        for (i = 0; i < nn->n_layers; i++) {
            nn->batch_outputs[i] = malloc(nn->n_dims[i] * bsize);
        }

        nn->k = batch_size;
    } else if (batch_size > nn->k) {
        for (i = 0; i < nn->n_layers; i++) {
            free(nn->batch_outputs[i]);
            nn->batch_outputs[i] = malloc(nn->n_dims[i] * bsize);
        }

        nn->k = batch_size;
    }

    if (batch_size > nn->ones_n) {
        nn->ones = realloc(nn->ones, bsize);
        for (i = nn->ones_n; i < batch_size; i++)
            nn->ones[i] = 1.0;
        nn->ones_n = batch_size;
    }
}


/*  Free memory from batch outputs */
void _nn_free_batch(nn_struct_t *nn)
{
    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->batch_outputs[i]);
        nn->batch_outputs[i] = NULL;
    }

    nn->batch_outputs[-1] = NULL;
}


/*  Allocate memory for gradients */
void _nn_alloc_grad(nn_struct_t *nn, int batch_size)
{
    const int gi_s = batch_size * nn->go_n * sizeof(value_t);

    if (nn->g_k == 0) {
        nn->g_w = malloc(nn->gw_n * sizeof(weight_t));
        nn->g_b = malloc(nn->gb_n * sizeof(weight_t));
        nn->g_out = malloc(gi_s);
        nn->g_in = malloc(gi_s);

        nn->g_k = batch_size;
    } else if (batch_size > nn->g_k) {
        free(nn->g_out);
        free(nn->g_in);

        nn->g_out = malloc(gi_s);
        nn->g_in = malloc(gi_s);

        nn->g_k = batch_size;
    }
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
