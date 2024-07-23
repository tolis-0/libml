#ifndef _NORMALIZATION_H
#define _NORMALIZATION_H

#include "core/ml_types.h"


#ifndef _STANDARD_ML_TYPES_
#define _STANDARD_ML_TYPES_
#define _STD_ML_TYPE_ double
typedef _STD_ML_TYPE_ weight_t, value_t, grad_t;
typedef int dim_t[2], dim3_t[3];
#endif // _STANDARD_ML_TYPES_


/*  normalization.c declarations */
void norm_minmax(value_t *data, int n, int m);
void norm_standard(value_t *data, int n, int m);


#endif // _NORMALIZATION_H
