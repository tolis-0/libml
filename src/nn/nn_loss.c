#include "../../include/nn.h" 
#include "nn_internal.h"


value_t nn_loss(nn_struct_t *nn, int k, const value_t *x, const value_t *t)
{
    _nn_alloc_batch(nn, k);

    nn->batch_outputs[-1] = (value_t *) x;
    nn_batch_forward_pass(nn, k);

    return loss_mse(k * nn->output_dims, nn->output, t);
}