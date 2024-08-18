#include <stdio.h>
#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*
 * Does a full forward pass of the neural network with a single input
 */
void _nn_forward_pass(nn_struct_t *nn)
{
    for (int i = 0; i < nn->num_of_layers; i++) {
        const value_t *const prev = nn->interm.outputs[i-1];
        value_t *const next = nn->interm.outputs[i];
        _Bool hb = nn->biases.num[i] > 0;

        switch (nn->operation_type[i]) {
            case DENSE_OP:;
                dim_t d = {nn->num_of_dims[i-1], nn->num_of_dims[i]};
                dense_forward(d, prev, nn->weights.of_layer[i],
                    hb, nn->biases.of_layer[i], next);
                break;

            /* Using the X-Macro for cases of activation functions */
#           define X(name, NAME, _1, _2)                        \
            case NAME##_OP:                                     \
                name##_forward(nn->num_of_dims[i], prev, next); \
                break;
            ML_ACTIVATION_FUNCTIONS_DECLARATIONS
#           undef X

            case EMPTY_OP:
                _ml_throw_error("network layer %d is empty", i);
                break;
            default:
                _ml_throw_error("network layer %d is not valid", i);
        }
    }
}


/*
 * Does a full forward pass of the neural network with a batch of inputs
 */
void _nn_batch_forward_pass(nn_struct_t *nn, int batch_size)
{
    for (int i = 0; i < nn->num_of_layers; i++) {
        const int arr_size = batch_size * nn->num_of_dims[i];
        const value_t *const prev = nn->interm.outputs[i-1];
        value_t *const next = nn->interm.outputs[i];
        _Bool hb = nn->biases.num[i] > 0;

        switch (nn->operation_type[i]) {
            case DENSE_OP:;
                dim3_t d = {nn->num_of_dims[i-1], nn->num_of_dims[i], batch_size};
                batch_dense_forward(d, prev, nn->weights.of_layer[i],
                    hb, nn->biases.of_layer[i], nn->interm.ones, next);
                break;

            /* Using the X-Macro for cases of activation functions */
#           define X(name, NAME, _1, _2)                \
            case NAME##_OP:                             \
                name##_forward(arr_size, prev, next);   \
                break;
            ML_ACTIVATION_FUNCTIONS_DECLARATIONS
#           undef X

            case EMPTY_OP:
                _ml_throw_error("network layer %d is empty", i);
                break;
            default:
                _ml_throw_error("network layer %d is not valid", i);
        }
    }
}
