#include <cblas.h>
#include "../../include/nn.h"


/*  Applies backpropagation and optimizes the neural network */
void nn_batch_backward_pass(nn_struct_t *nn, int batch_size)
{
    int i;
    value_t *swap_ptr;

    for (i = nn->n_layers - 1; i >= 0; i--) {
        switch(nn->op_types[i]) {
            case NO_OP: break;
            case DENSE_OP:;
                dim3_t d = {nn->n_dims[i-1], nn->n_dims[i], batch_size};
                batch_dense_backward(d, nn->batch_outputs[i-1], nn->weights[i],
                    i != 0, nn->ones, nn->g_out, nn->g_in, nn->g_w,
                    nn->n_biases[i] > 0, nn->g_b);

                /* TODO: MOVE TO ANOTHER FILE AND ADD REGULARIZATION */
                cblas_daxpy(nn->n_weights[i], -nn->learning_rate,
                    nn->g_w, 1, nn->weights[i], 1);

                if (nn->n_biases[i] > 0) {
                    cblas_daxpy(nn->n_biases[i], -nn->learning_rate,
                        nn->g_b, 1, nn->biases[i], 1);
                }
                /* .......................... */
                break;
            case RELU_OP:
                relu_backward(nn->n_dims[i] * batch_size,
                    nn->batch_outputs[i-1], nn->g_out, nn->g_in);
                break;
            case LRELU_OP:
                lrelu_backward(nn->n_dims[i] * batch_size,
                    nn->batch_outputs[i-1], nn->g_out, nn->g_in);
                break;
            case LOGISTIC_OP:
                logistic_backward(nn->n_dims[i] * batch_size,
                    nn->batch_outputs[i], nn->g_out, nn->g_in);
                break;
            case TANH_OP:
                tanh_backward(nn->n_dims[i] * batch_size,
                    nn->batch_outputs[i], nn->g_out, nn->g_in);
                break;
        }

        swap_ptr = nn->g_out;
        nn->g_out = nn->g_in;
        nn->g_in = swap_ptr;
    }

}
