#ifndef _NN_INTERNAL_H
#define _NN_INTERNAL_H


#include "../error_internal.h"


/* nn/nn_mem.c declarations */
void _nn_alloc_interm(nn_struct_t *nn, int batch_size);
void _nn_free_interm(nn_struct_t *nn);
void _nn_alloc_grad(nn_struct_t *nn, int batch_size);
void _nn_free_grad(nn_struct_t *nn);


/* nn/nn_weights.c declarations */
void _nn_rand_weights(nn_struct_t *nn);
/* nn/nn_regularization.c declarations */
void _nn_regularization(int n, nn_reg_t *reg, cwrp_t w, grp_t gw);
/* nn/nn_forward_pass.c declarations */
void _nn_forward_pass(nn_struct_t *nn);
void _nn_batch_forward_pass(nn_struct_t *nn, int batch_size);
/* nn/nn_backward_pass.c declarations */
void _nn_batch_backward_pass(nn_struct_t *nn, int batch_size);


#endif // _NN_INTERNAL_H
