#include <string.h>
#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*  User function that predicts an output of the neural network */
void nn_predict(value_t *output, nn_struct_t *nn,
    const value_t *input)
{
    nn->outputs[-1] = (value_t *) input;
    nn_forward_pass(nn);

    memcpy(output, nn->output, nn->output_dims * sizeof(value_t));
}


/*  User function that predicts a batch of outputs of the neural network */
void nn_batch_predict(value_t *output, nn_struct_t *nn,
    const value_t *input, int k)
{
    _nn_alloc_batch(nn, k);

    nn->batch_outputs[-1] = (value_t *) input;
    nn_batch_forward_pass(nn, k);

    memcpy(output, nn->output, k * nn->output_dims * sizeof(value_t));
}
