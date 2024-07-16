#include <string.h>
#include "../../include/nn.h"


/*  Does a full forward pass of the neural network with a single input */
void nn_forward_pass(nn_struct_t *nn, value_t *input)
{
    int i;

    nn->outputs[-1] = input;

    for (i = 0; i < nn->n_layers; i++) {
        switch(nn->op_types[i]) {
            case NO_OP:
                memcpy(nn->outputs[i], nn->outputs[i-1],
                    nn->n_dims[i] * sizeof(value_t));
                break;
            case DENSE_OP:;
                dim_t d = {nn->n_dims[i-1], nn->n_dims[i]};
                dense_forward(d, nn->outputs[i-1], nn->weights[i],
                    nn->n_biases[i] > 0, nn->biases[i], nn->outputs[i]);
                break;
            case RELU_OP:
                relu_forward(nn->n_dims[i], nn->outputs[i-1], nn->outputs[i]);
                break;
            case LOGISTIC_OP:
                logistic_forward(nn->n_dims[i], nn->outputs[i-1], nn->outputs[i]);
                break;
        }
    }

    nn->output = nn->outputs[nn->n_layers - 1];
}


/*  Does a full forward pass of the neural network with a batch of inputs */
void nn_batch_forward_pass(nn_struct_t *nn, int batch_size)
{
    for (int i = 0; i < nn->n_layers; i++) {
        switch(nn->op_types[i]) {
            case NO_OP:
                memcpy(nn->batch_outputs[i], nn->batch_outputs[i-1],
                    batch_size * nn->n_dims[i] * sizeof(value_t));
                break;
            case DENSE_OP:;
                dim3_t d = {nn->n_dims[i-1], nn->n_dims[i], batch_size};
                batch_dense_forward(d, nn->batch_outputs[i-1], nn->weights[i],
                    nn->n_biases[i] > 0, nn->biases[i], nn->ones,
                    nn->batch_outputs[i]);
                break;
            case RELU_OP:
                relu_forward(nn->n_dims[i] * batch_size,
                    nn->batch_outputs[i-1], nn->batch_outputs[i]);
                break;
            case LOGISTIC_OP:
                logistic_forward(nn->n_dims[i] * batch_size,
                    nn->batch_outputs[i-1], nn->batch_outputs[i]);
                break;
        }
    }

    nn->output = nn->batch_outputs[nn->n_layers - 1];
}
