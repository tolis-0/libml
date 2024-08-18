#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"

#define THRESHOLD __ml_fpc(0.5)

/*
 * Calculates the accuracy of the neural network on some data
 */
float _nn_accuracy(nn_struct_t *nn, int size, value_t *x, value_t *t)
{
    const int o_n = NN_OUTPUT_DIMS(nn);
    int i, j, correct;
    int arg_max, one_index;
    const value_t *restrict t_ptr, *restrict o_ptr;
    value_t val, max;


    _nn_alloc_interm(nn, size);

    NN_INPUT(nn) = x;
    _nn_batch_forward_pass(nn, size);

    t_ptr = t;
    o_ptr = NN_OUTPUT(nn);
    correct = 0;

    for (i = 0; i < size; i++) {
        max = -DBL_MAX;
        arg_max = -1;
        one_index = -1;

        for (j = 0; j < o_n; j++) {
            val = o_ptr[j];
            one_index = (t_ptr[j] > THRESHOLD) ? j : one_index;
            max = (val > max) ? (arg_max = j, val) : max;
        }

        correct += (arg_max == one_index);

        t_ptr += o_n;
        o_ptr += o_n;
    }

    _nn_free_interm(nn);

    return correct / (float) size;
}
