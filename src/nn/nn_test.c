#include <stdlib.h>
#include "../../include/nn.h"


/*  Helper function to allocate memory for testing */
void _nn_alloc_test(nn_struct_t *nn, int batch_size)
{
#undef __NN_ALLOC_GRADIENTS__
#include "source_nn_alloc_t.h"
}


/*  Helper function to free memory allocated for testing */
void _nn_free_test(nn_struct_t *nn)
{
#undef __NN_FREE_GRADIENTS__
#include "source_nn_free_t.h"
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
