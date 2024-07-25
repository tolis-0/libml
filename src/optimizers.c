#include <cblas.h>
#include "../include/nn.h"


void opt_sgd(int n, weight_t lr, cgrp_t grad, wrp_t w)
{
    cblas_axpy(n, -lr, grad, 1, w, 1);
}
