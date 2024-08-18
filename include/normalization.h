#ifndef _ML_NORMALIZATION_H
#define _ML_NORMALIZATION_H

#include "core/ml_types.h"


/*  normalization.c declarations */
void norm_minmax(value_t *data, int n, int m);
void norm_standard(value_t *data, int n, int m);


#endif // _ML_NORMALIZATION_H
