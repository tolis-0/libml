#include <cblas.h>
#include "../../include/nn.h"


static inline void _nn_l1_reg(int n, weight_t p, cwrp_t w, grp_t gw)
{
    for (int i = 0; i < n; i++)
        gw[i] += (w[i] >= 0) ? p : -p;
}


static inline void _nn_l2_reg(int n, weight_t p, cwrp_t w, grp_t gw)
{
    cblas_axpy(n, p, w, 1, gw, 1);
}


void _nn_regularization(int n, nn_reg_t type, weight_t p, cwrp_t w, grp_t gw)
{
    switch(type) {
        case NONE: break;
        case L1:
            _nn_l1_reg(n, p, w, gw);
            break;
        case L2:
            _nn_l2_reg(n, p, w, gw);
            break;
    }
}
