#include <cblas.h>
#include "../../include/opt.h"


/* Apply the simple Gradient Descent optimization */
void opt_apply_gd(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w)
{
    (void) o;
    cblas_axpy(n, -lr, grad, 1, w, 1);
}


/* Apply the Common Momentum optimization */
void opt_apply_cm(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w)
{
    opt_cm_t const opt = o->cm;

    cblas_scal(n, opt.beta, opt.v, 1);
    cblas_axpy(n, (1.0 - opt.beta), grad, 1, opt.v, 1);
    cblas_axpy(n, -lr, opt.v, 1, w, 1);
}


/* Apply the Nesterov Accelerated Gradient optimization */
void opt_apply_nag(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w)
{
    (void) o; (void) n; (void) lr; (void) grad; (void) w; // TODO
}
