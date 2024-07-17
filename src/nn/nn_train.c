#include <stdlib.h>
#include "../../include/nn.h"


/*  Helper function to allocate memory for training */
void _nn_alloc_train(nn_struct_t *nn, int batch_size)
{
#define __NN_ALLOC_GRADIENTS__
#include "source_nn_alloc_t.h"
#undef __NN_ALLOC_GRADIENTS__
}


/*  Helper function to free memory allocated for training */
void _nn_free_train(nn_struct_t *nn)
{
#define __NN_FREE_GRADIENTS__
#include "source_nn_free_t.h"
#undef __NN_FREE_GRADIENTS__
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
