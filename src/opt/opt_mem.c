#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "../../include/nn.h"


#define _opt_malloc_error(var)                                      \
    if (__builtin_expect(((var) == NULL), 0)) {                     \
        fprintf(stderr, "\e[1;39m%s\e[0;39m"                        \
            " (from \e[1;39m%s:%d\e[0;39m) \e[1;31merror\e[0;39m:"  \
            " malloc failed for " #var ", %s\n",               \
            func, file, line, strerror(errno)                       \
        );                                                          \
        exit(EXIT_FAILURE);                                         \
    }


/*  Allocates memory for values and gradients used by optimizers */
void _opt_alloc_val(nn_struct_t *nn,
    const char *func, const char *file, int line)
{
    ml_opt_t *const opt = &(nn->opt);
    const int items = nn->total_weights + nn->total_biases;

    if (opt->mem_alloc) return;

    switch (opt->type) {
        case GD_OPT:
            break;
        case CM_OPT:
            opt->params.cm.v = calloc(items, sizeof(grad_t));
            _opt_malloc_error(opt->params.cm.v);
            break;
        case NAG_OPT:
            // TODO
            break;
        default:
            // TODO
            break;
    }

    opt->mem_alloc = 1;
}


/*  Free memory from oiptimizer values and gradients */
void _opt_free_val(nn_struct_t *nn)
{
    ml_opt_t *const opt = &(nn->opt);

    if (!opt->mem_alloc) return;

    switch (opt->type) {
        case GD_OPT:
            break;
        case CM_OPT:
            free(opt->params.cm.v);
            break;
        case NAG_OPT:
            // TODO
            break;
        default:
            // TODO
            break;
    }

    opt->mem_alloc = 0;
}
