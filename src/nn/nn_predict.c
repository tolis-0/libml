#include <string.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*  User function that predicts an output of the neural network */
void _nn_predict(nn_struct_t *nn, int k, const value_t *input, value_t *output,
    const char *file, int line)
{
    if (k == 1) {
        nn->outputs[-1] = (value_t *) input;
        nn_forward_pass(nn);
    } else {
        _nn_alloc_batch(nn, k, "nn_predict", file, line);

        nn->batch_outputs[-1] = (value_t *) input;
        nn_batch_forward_pass(nn, k);
    }

    memcpy(output, nn->output, k * nn->output_dims * sizeof(value_t));
}
