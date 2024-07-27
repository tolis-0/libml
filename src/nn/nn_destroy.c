#include <stdlib.h>
#include <stdio.h>
#include "../../include/nn.h"
#include "../opt/opt_internal.h"


#define _nn_cond_free(var) if (var != NULL) free(var)

#define _nn_destroy_error(cond, str, ...)                           \
    if (__builtin_expect(!!(cond), 0)) {                            \
        fprintf(stderr, "\e[1;39mnn_destroy\e[0;39m"                \
            " (from \e[1;39m%s:%d\e[0;39m) \e[1;31merror\e[0;39m: " \
            str "\n", file, line, ##__VA_ARGS__);                   \
        exit(EXIT_FAILURE);                                         \
    }


/*  Free all the memory used by the neural network struct */
void _nn_destroy(nn_struct_t *nn, const char *file, int line)
{
    _nn_destroy_error(nn == NULL, "Neural Network is Null");

    free(--nn->n_dims);
    free(nn->op_types);

    free(nn->n_weights);
    free(nn->n_biases);
    free(nn->weights_ptr);
    free(nn->gw_ptr);
    free(nn->weights);
    free(nn->biases);

    free(nn->reg_type);
    free(nn->reg_p);

    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->outputs[i]);
        _nn_cond_free(nn->batch_outputs[i]);
    }

    free(--nn->outputs);
    free(--nn->batch_outputs);
    free(nn->ones);

    _nn_cond_free(nn->g_out);
    _nn_cond_free(nn->g_in);
    free(nn->gw);
    free(nn->gb);

    for (int i = 0; i < NN_ADDRK_SIZE; i++) {
        _nn_cond_free(nn->addr_keeper[i]);
    }

    free(nn);
}
