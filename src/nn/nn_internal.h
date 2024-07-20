#ifndef _NN_INTERNAL_H
#define _NN_INTERNAL_H


/*  nn/nn_alloc.c declarations */
void _nn_alloc_batch(nn_struct_t *nn, int batch_size);
void _nn_realloc_batch(nn_struct_t *nn, int batch_size);
void _nn_free_batch(nn_struct_t *nn);
void _nn_alloc_grad(nn_struct_t *nn);
void _nn_realloc_grad(nn_struct_t *nn);
void _nn_free_grad(nn_struct_t *nn);


#endif // _NN_INTERNAL_H
