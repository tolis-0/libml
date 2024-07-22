#ifndef _NN_INTERNAL_H
#define _NN_INTERNAL_H


#define inv_sqrt2 0.70710678118


/*  nn/nn_alloc.c declarations */
void _nn_alloc_batch(nn_struct_t *nn, int batch_size);
void _nn_free_batch(nn_struct_t *nn);
void _nn_alloc_grad(nn_struct_t *nn, int batch_size);
void _nn_free_grad(nn_struct_t *nn);


/* 	nn/nn_weights.c declarations */
void _nn_rand_weights(nn_struct_t *nn);


#endif // _NN_INTERNAL_H
