#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"
#include "../opt/opt_internal.h"


/*  Trains the neural network */
void _nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size,
    value_t *x, value_t *t, const char *file, int line)
{
    int i, j;

    const int data_size = batch_size * nn->input_dims;
    const int batch_num = set_size / batch_size;
    const int output_size = batch_size * nn->output_dims;
    const int gw_s = nn->total_weights + nn->total_biases;
    const int r = !!nn->stochastic;

    _opt_alloc_val(nn, "nn_train", file, line);
    _nn_alloc_batch(nn, batch_size, "nn_train", file, line);
    _nn_alloc_grad(nn, batch_size, "nn_train", file, line);

    for (i = 0; i < epochs; i++) {
        for (j = 0; j < batch_num; j++) {
            const int index = r ? (rand() % batch_num) : j;
            const value_t *const _t = t + output_size * index;
            nn->batch_outputs[-1] = x + data_size * index;

            nn_batch_forward_pass(nn, batch_size);
            loss_diff_grad(output_size, nn->output, _t, nn->g_out);
            nn_batch_backward_pass(nn, batch_size);

            opt_t *const p = &(nn->opt.params);
            nn->opt.call(p, gw_s, nn->learning_rate, nn->gw_ptr, nn->weights_ptr);
        }
    }

    _nn_free_batch(nn);
    _nn_free_grad(nn);
}
