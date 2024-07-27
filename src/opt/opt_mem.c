#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "../../include/nn.h"
#include "opt_internal.h"


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
    _opt_free_val(2, nn->addr_keeper); // addresses at indexes 0-1

    switch (opt->type) {
        case GD_OPT:
            break;
        case CM_OPT:
            opt->params.cm.v = calloc(items, sizeof(grad_t));
            _opt_malloc_error(opt->params.cm.v);
            nn->addr_keeper[0] = opt->params.cm.v;
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


/*  Free memory from previous optimizer values and gradients */
void _opt_free_val(int n, void **opt_addr)
{
    for (int i = 0; i < n; i++) {
        if (opt_addr[i] != NULL) {
            free(opt_addr[i]);
            opt_addr[i] = NULL;
        }
    }
}
