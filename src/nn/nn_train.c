#include <stdlib.h>
#include "../../include/nn.h"


/*  Helper function to allocate memory for training */
void _nn_alloc_train(nn_struct_t *nn, int batch_size)
{
    int i, max_dims, max_w, max_b;

    max_b = max_w = 0;
    max_dims = nn->input_dims;

    for (i = 0; i < nn->n_layers; i++) {
        nn->batch_outputs[i] = malloc(
            nn->n_dims[i] * batch_size * sizeof(value_t));
        max_dims = (nn->n_dims[i] > max_dims) ? nn->n_dims[i] : max_dims;
        max_b = (nn->n_biases[i] > max_b) ? nn->n_biases[i] : max_b;
        max_w = (nn->n_weights[i] > max_w) ? nn->n_weights[i] : max_w;
    }

    nn->ones = malloc(batch_size * sizeof(value_t));
    for (i = 0; i < batch_size; i++)
        nn->ones[i] = 1.0;


    nn->g_w = malloc(max_w * sizeof(weight_t));
    nn->g_b = malloc(max_b * sizeof(weight_t));
    nn->g_out = malloc(batch_size * max_dims * sizeof(value_t));
    nn->g_in = malloc(batch_size * max_dims * sizeof(value_t));
}


/*  Helper function to free memory allocated for training */
void _nn_free_train(nn_struct_t *nn)
{
    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->batch_outputs[i]);
        nn->batch_outputs[i] = NULL;
    }

    nn->batch_outputs[-1] = NULL;

    free(nn->ones);
    free(nn->g_w);
    free(nn->g_b);
    free(nn->g_in);
    free(nn->g_out);

    nn->ones = NULL;
    nn->g_w = NULL;
    nn->g_b = NULL;
    nn->g_out = NULL;
    nn->g_in = NULL;
}


/*  Trains the neural network */
void nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size,
    value_t *x, value_t *t)
{
    int i, j, batch_num;

    batch_num = set_size / batch_size;

    _nn_alloc_train(nn, batch_size);

    for (i = 0; i < epochs; i++) {
        for (j = 0; j < batch_num; j++) {
            nn->batch_outputs[-1] = x + batch_size * j;

            nn_batch_forward_pass(nn, batch_size);
            loss_diff_grad(batch_size * nn->output_dims,
                nn->output, t, nn->g_out);
            nn_batch_backward_pass(nn, batch_size);

            nn->batch_outputs[-1] = NULL;
        }
    }

    _nn_free_train(nn);
}
