#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../../include/nn.h"
#include "opt_internal.h"



/* Allocates memory for values and gradients used by optimizers */
void _opt_alloc_val(nn_struct_t *nn)
{
    ml_opt_t *const opt = &(nn->optimizer);
    const int items = nn->weights.total + nn->biases.total;

    if (opt->mem_alloc) return;
    _opt_free_val(2, nn->mem_addr); /* addresses at indexes 0-1 */

    switch (opt->type) {
        case GD_OPT:
            break;
        case CM_OPT:
            __ml_calloc_check(opt->params.cm.v, grad_t, items);
            nn->mem_addr[0] = opt->params.cm.v;
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


/* Free memory from previous optimizer values and gradients */
void _opt_free_val(int n, void **opt_addr)
{
    for (int i = 0; i < n; i++) {
        free(opt_addr[i]);
        opt_addr[i] = NULL;
    }
}
