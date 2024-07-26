#include <cblas.h>
#include "../../include/opt.h"


/*  Apply the simple Gradient Descent optimization */
void opt_apply_gd(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w)
{
    cblas_axpy(n, -lr, grad, 1, w, 1);
}


/*  Apply the Common Momentum optimization */
void opt_apply_cm(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w)
{
    // TODO
}


/*  Apply the Nesterov Accelerated Gradient optimization */
void opt_apply_nag(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w)
{
    // TODO
}
