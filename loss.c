#include <cblas.h>
#include "nn.h"


/*	Implements grad = y - t */
void loss_diff_grad(int d, const value_t *y, const value_t *t, value_t *grad)
{
	cblas_dcopy(d, y, 1, grad, 1);
    cblas_daxpy(d, -1.0, t, 1, grad, 1);
}
