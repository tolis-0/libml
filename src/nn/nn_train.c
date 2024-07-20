#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*  Trains the neural network */
void nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size,
    value_t *x, value_t *t)
{
    int i, j, batch_num;

    batch_num = set_size / batch_size;

    _nn_alloc_batch(nn, batch_size);
    _nn_alloc_grad(nn, batch_size);

    for (i = 0; i < epochs; i++) {
        for (j = 0; j < batch_num; j++) {
            nn->batch_outputs[-1] = x + batch_size * j;

            nn_batch_forward_pass(nn, batch_size);
            loss_diff_grad(batch_size * nn->output_dims,
                nn->output, t, nn->g_out);
            nn_batch_backward_pass(nn, batch_size);
        }
    }

    _nn_free_grad(nn);
}
