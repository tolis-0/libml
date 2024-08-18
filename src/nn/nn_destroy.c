#include <stdlib.h>
#include <stdio.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*
 * Free all the memory used by the neural network struct
 */
void _nn_destroy(nn_struct_t *nn)
{
    if (nn == NULL)
        return;

    free(--nn->num_of_dims);
    free(nn->operation_type);
    free(nn->regularization);

    free(nn->weights.ptr);
    free(nn->weights.num);
    free(nn->weights.of_layer);
    free(nn->biases.num);
    free(nn->biases.of_layer);

    if (nn->interm.outputs != NULL) {
        for (int i = 0; i < nn->num_of_layers; i++)
            free(nn->interm.outputs[i]);

        free(--nn->interm.outputs);
    }

    free(nn->interm.ones);

    free(nn->grad.ptr);
    free(nn->grad.out);
    free(nn->grad.in);
    free(nn->grad.weights);
    free(nn->grad.biases);

    for (int i = 0; i < MEM_ADDR_SIZE; i++)
        free(nn->mem_addr[i]);

    free(nn);
}
