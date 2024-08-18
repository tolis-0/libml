#include "../../include/nn.h"
#include "nn_internal.h"


value_t _nn_loss(nn_struct_t *nn, int k, const value_t *x, const value_t *t)
{
    _nn_alloc_interm(nn, k);

    NN_INPUT(nn) = (value_t *) x;
    _nn_batch_forward_pass(nn, k);

    return loss_mse(k * NN_OUTPUT_DIMS(nn), NN_OUTPUT(nn), t);
}
