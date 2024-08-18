#include "../../include/nn.h"
#include "nn_internal.h"


/*
 * Using these EXPAND macros we can choose whether to use certain
 * arguments for the backwards operations of the activation layers:
 * - EXPAND_INPUT_ ## req_x     decides for the input of the layer x
 * - EXPAND_OUTPUT_ ## req_y    decides for the output of the layer y
 */
#define EXPAND_INPUT_0   /* nothing */
#define EXPAND_INPUT_1   nn->interm.outputs[i - 1],
#define EXPAND_OUTPUT_0  /* nothing */
#define EXPAND_OUTPUT_1  nn->interm.outputs[i],


/*
 * Applies backpropagation to calculate the gradients
 */
void _nn_batch_backward_pass(nn_struct_t *nn, int batch_size)
{
    value_t *swap_ptr;

    for (int i = nn->num_of_layers - 1; i >= 0; i--) {
        switch (nn->operation_type[i]) {
            case DENSE_OP:;
                dim3_t d = {nn->num_of_dims[i-1], nn->num_of_dims[i], batch_size};
                batch_dense_backward(d, nn->interm.outputs[i-1],
                    nn->weights.of_layer[i], i > 0, nn->interm.ones,
                    nn->grad.out, nn->grad.in, nn->grad.weights[i],
                    nn->biases.num[i] > 0, nn->grad.biases[i]);
                break;

            /* Using the X-Macro for cases of activation functions */
#           define X(name, NAME, req_x, req_y)                      \
            case NAME##_OP:                                         \
                name##_backward(nn->num_of_dims[i] * batch_size,    \
                    EXPAND_INPUT_ ## req_x                          \
                    EXPAND_OUTPUT_ ## req_y                         \
                    nn->grad.out, nn->grad.in);                     \
                break;
            ML_ACTIVATION_FUNCTIONS_DECLARATIONS
#           undef X

            case EMPTY_OP:
                _ml_throw_error("network layer %d is empty", i);
                break;
            default:
                _ml_throw_error("network layer %d is not valid", i);
        }

        _nn_regularization(nn->weights.num[i], &nn->regularization[i],
            nn->weights.of_layer[i], nn->grad.weights[i]);

        swap_ptr = nn->grad.out;
        nn->grad.out = nn->grad.in;
        nn->grad.in = swap_ptr;
    }
}
