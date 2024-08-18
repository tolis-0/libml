#include <string.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*
 * User function that predicts an output of the neural network
 */
void _nn_predict(nn_struct_t *nn, int k, const value_t *input, value_t *output)
{
    _nn_alloc_interm(nn, k);

    NN_INPUT(nn) = (value_t *) input;
    value_t *swap = NN_OUTPUT(nn);
    NN_OUTPUT(nn) = output;

    _nn_batch_forward_pass(nn, k);

    // TODO: need to test the swap for both NN_OUTPUT(nn) and output
    NN_OUTPUT(nn) = swap;
}
