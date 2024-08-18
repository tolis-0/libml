#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"
#include "../opt/opt_internal.h"


/*
 * User function that trains the neural network
 */
void _nn_train(nn_struct_t *nn, int epochs, int batch_size, int set_size, value_t *x, value_t *t)
{
    const size_t data_size = (size_t) batch_size * (size_t) NN_INPUT_DIMS(nn);
    const size_t output_size = (size_t) batch_size * (size_t) NN_OUTPUT_DIMS(nn);
    const int batch_num = set_size / batch_size;
    const int gw_s = nn->weights.total + nn->biases.total;

    _opt_alloc_val(nn);
    _nn_alloc_interm(nn, batch_size);
    _nn_alloc_grad(nn, batch_size);

    for (int i = 0; i < epochs; i++) {
        for (int j = 0; j < batch_num; j++) {
            const size_t index = nn->stochastic ? (rand() % batch_num) : j;
            const value_t *restrict const _t = &t[output_size * index];
            NN_INPUT(nn) = &x[data_size * index];

            _nn_batch_forward_pass(nn, batch_size);
            loss_diff_grad(output_size, NN_OUTPUT(nn), _t, nn->grad.out);
            _nn_batch_backward_pass(nn, batch_size);

            opt_t *restrict const p = &(nn->optimizer.params);
            nn->optimizer.call(p, gw_s, nn->learning_rate, nn->grad.ptr, nn->weights.ptr);
        }
    }

    _nn_free_interm(nn);
    _nn_free_grad(nn);
}
