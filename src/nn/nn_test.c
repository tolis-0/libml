#include <stdlib.h>
#include "../../include/nn.h"


/*  Helper function to allocate memory for testing */
void _nn_alloc_test(nn_struct_t *nn, int batch_size)
{
    for (int i = 0; i < nn->n_layers; i++) {
        nn->batch_outputs[i] = malloc(
            nn->n_dims[i] * batch_size * sizeof(value_t));
    }

    nn->ones = malloc(batch_size * sizeof(value_t));
    for (int i = 0; i < batch_size; i++)
        nn->ones[i] = 1.0;
}


/*  Helper function to free memory allocated for testing */
void _nn_free_test(nn_struct_t *nn)
{
    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->batch_outputs[i]);
        nn->batch_outputs[i] = NULL;
    }

    nn->batch_outputs[-1] = NULL;

    free(nn->ones);

    nn->ones = NULL;
}


/*  Tests the accuracy of the neural network */
double nn_test(nn_struct_t *nn, int test_size, value_t *x, value_t *t)
{
    int i, j, c = nn->output_dims, arg_max = 0, one_index = 0, correct = 0;
    value_t val, max;

    _nn_alloc_test(nn, test_size);

    nn->batch_outputs[-1] = x;
    nn_batch_forward_pass(nn, test_size);

    for (i = 0; i < test_size; i++) {
        max = 0.0;
        for (j = 0; j < c; j++) {
            val = nn->output[i*c + j];
            one_index = (t[i*c + j] > 0.5) ? j : one_index;
            max = (val > max) ? (arg_max = j, val) : max;
        }

        if (arg_max == one_index) correct++;
    }

    nn->batch_outputs[-1] = NULL;
    _nn_free_test(nn);

    return correct / (double) test_size;
}
