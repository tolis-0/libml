#include <string.h>
#include "../../include/nn.h"


/*  Does a full forward pass of the neural network
    with a single input (in nn->outputs[-1])        */
void nn_forward_pass(nn_struct_t *nn)
{
    const int n = nn->n_layers;

    for (int i = 0; i < n; i++) {
        const value_t *prev = nn->outputs[i-1];
        value_t *next = nn->outputs[i];
        int has_b = nn->n_biases[i] > 0;

        switch(nn->op_types[i]) {
            case NO_OP:
                memcpy(next, prev, nn->n_dims[i] * sizeof(value_t));
                break;
            case DENSE_OP:;
                dim_t d = {nn->n_dims[i-1], nn->n_dims[i]};
                dense_forward(d, prev, nn->weights[i],
                    has_b, nn->biases[i], next);
                break;
            case RELU_OP:
                relu_forward(nn->n_dims[i], prev, next);
                break;
            case LOGISTIC_OP:
                logistic_forward(nn->n_dims[i], prev, next);
                break;
        }
    }

    nn->output = nn->outputs[n - 1];
}


/*  Does a full forward pass of the neural network
    with a batch of inputs (in nn->batch_outputs[-1])   */
void nn_batch_forward_pass(nn_struct_t *nn, int batch_size)
{
    const int n = nn->n_layers;

    for (int i = 0; i < n; i++) {
        const int arr_n = batch_size * nn->n_dims[i];
        const value_t *prev = nn->batch_outputs[i-1];
        value_t *next = nn->batch_outputs[i];
        int has_b = nn->n_biases[i] > 0;

        switch(nn->op_types[i]) {
            case NO_OP:
                memcpy(next, prev, arr_n * sizeof(value_t));
                break;
            case DENSE_OP:;
                dim3_t d = {nn->n_dims[i-1], nn->n_dims[i], batch_size};
                batch_dense_forward(d, prev, nn->weights[i],
                    has_b, nn->biases[i], nn->ones, next);
                break;
            case RELU_OP:
                relu_forward(arr_n, prev, next);
                break;
            case LOGISTIC_OP:
                logistic_forward(arr_n, prev, next);
                break;
        }
    }

    nn->output = nn->batch_outputs[n - 1];
}
