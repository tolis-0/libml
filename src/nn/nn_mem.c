#include <stdio.h>
#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"


/*
 * Allocates memory for the intermediate values
 * based on given batch size
 * TODO: use nni->ptr or nni->outputs[0] for the allocation
 */
void _nn_alloc_interm(nn_struct_t *nn, int batch_size)
{
    nn_intermv_t *restrict const nni = &nn->interm;

    if (nni->batch_size == 0) {
        for (int i = 0; i < nn->num_of_layers; i++) {
            const size_t size = batch_size * nn->num_of_dims[i];
            __ml_malloc_check(nni->outputs[i], value_t, size);
        }

        nni->batch_size = batch_size;
    } else if (batch_size > nni->batch_size) {
        for (int i = 0; i < nn->num_of_layers; i++) {
            free(nni->outputs[i]);

            const size_t size = batch_size * nn->num_of_dims[i];
            __ml_malloc_check(nni->outputs[i], value_t, size);
        }

        nni->batch_size = batch_size;
    }

    if (batch_size > nni->ones_size) {
        __ml_realloc_check(nni->ones, value_t, batch_size);

        for (int i = nni->ones_size; i < batch_size; i++)
            nni->ones[i] = __ml_fpc(1.0);

        nni->ones_size = batch_size;
    }
}


/*
 * Free memory used for intermediate values and outputs
 */
void _nn_free_interm(nn_struct_t *nn)
{
    nn_intermv_t *restrict const nni = &nn->interm;

    for (int i = 0; i < nn->num_of_layers; i++) {
        free(nni->outputs[i]);
        nni->outputs[i] = NULL;
    }

    nni->outputs[-1] = NULL;
    nni->batch_size = 0;
}


/*
 * Allocate memory for gradients based on given batch size
 */
void _nn_alloc_grad(nn_struct_t *nn, int batch_size)
{
    const int size = batch_size * nn->max_dims;
    nn_gradv_t *restrict const nng = &nn->grad;

    if (nng->batch_size == 0) {
        __ml_malloc_check(nng->out, grad_t, size);
        __ml_malloc_check(nng->in, grad_t, size);

        nng->batch_size = batch_size;
    } else if (batch_size > nng->batch_size) {
        free(nng->out);
        free(nng->in);

        __ml_malloc_check(nng->out, grad_t, size);
        __ml_malloc_check(nng->in, grad_t, size);

        nng->batch_size = batch_size;
    }
}


/*
 * Free memory used for gradients
 */
void _nn_free_grad(nn_struct_t *nn)
{
    nn_gradv_t *restrict const nng = &nn->grad;

    free(nng->out);
    free(nng->in);

    nng->out = NULL,
    nng->in = NULL;

    nng->batch_size = 0;
}
