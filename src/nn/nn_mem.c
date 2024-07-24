#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "../../include/nn.h"


#define _nn_malloc_error(var) _nn_alloc_error(m, var)
#define _nn_realloc_error(var) _nn_alloc_error(re, var)

#define _nn_alloc_error(m, var)                                     \
    if (__builtin_expect(((var) == NULL), 0)) {                     \
        fprintf(stderr, "\e[1;39m%s\e[0;39m"                        \
            " (from \e[1;39m%s:%d\e[0;39m) \e[1;31merror\e[0;39m:"  \
            " " #m "alloc failed for " #var ", %s\n",               \
            func, file, line, strerror(errno)                       \
        );                                                          \
        exit(EXIT_FAILURE);                                         \
    }


/*  Allocates memory for the batch intermediate values and outputs */
void _nn_alloc_batch(nn_struct_t *nn, int batch_size,
    const char *func, const char *file, int line)
{
    const int bsize = batch_size * sizeof(value_t);
    int i;

    if (nn->k == 0) {
        for (i = 0; i < nn->n_layers; i++) {
            nn->batch_outputs[i] = malloc(nn->n_dims[i] * bsize);
            _nn_malloc_error(nn->batch_outputs[i]);
        }

        nn->k = batch_size;
    } else if (batch_size > nn->k) {
        for (i = 0; i < nn->n_layers; i++) {
            free(nn->batch_outputs[i]);

            nn->batch_outputs[i] = malloc(nn->n_dims[i] * bsize);
            _nn_malloc_error(nn->batch_outputs[i]);
        }

        nn->k = batch_size;
    }

    if (batch_size > nn->ones_n) {
        nn->ones = realloc(nn->ones, bsize);
        _nn_realloc_error(nn->ones);

        for (i = nn->ones_n; i < batch_size; i++)
            nn->ones[i] = 1.0;

        nn->ones_n = batch_size;
    }
}


/*  Free memory from batch outputs */
void _nn_free_batch(nn_struct_t *nn)
{
    for (int i = 0; i < nn->n_layers; i++) {
        free(nn->batch_outputs[i]);
        nn->batch_outputs[i] = NULL;
    }

    nn->batch_outputs[-1] = NULL;

    nn->k = 0;
}


/*  Allocate memory for gradients */
void _nn_alloc_grad(nn_struct_t *nn, int batch_size,
    const char *func, const char *file, int line)
{
    const int gi_s = batch_size * nn->go_n * sizeof(value_t);

    if (nn->g_k == 0) {
        nn->g_w = malloc(nn->gw_n * sizeof(weight_t));
        _nn_malloc_error(nn->g_w);

        nn->g_b = malloc(nn->gb_n * sizeof(weight_t));
        _nn_malloc_error(nn->g_b);

        nn->g_out = malloc(gi_s);
        _nn_malloc_error(nn->g_out);

        nn->g_in = malloc(gi_s);
        _nn_malloc_error(nn->g_in);

        nn->g_k = batch_size;
    } else if (batch_size > nn->g_k) {
        free(nn->g_out);
        free(nn->g_in);

        nn->g_out = malloc(gi_s);
        _nn_malloc_error(nn->g_out);

        nn->g_in = malloc(gi_s);
        _nn_malloc_error(nn->g_in);

        nn->g_k = batch_size;
    }
}


/*  Free memory from gradients */
void _nn_free_grad(nn_struct_t *nn)
{
    free(nn->g_w);
    free(nn->g_b);
    free(nn->g_in);
    free(nn->g_out);

    nn->g_w = NULL;
    nn->g_b = NULL;
    nn->g_out = NULL;
    nn->g_in = NULL;

    nn->g_k = 0;
}
