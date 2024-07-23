#ifndef _NORMALIZATION_H
#define _NORMALIZATION_H

#include "core/ml_types.h"


/*  normalization.c declarations */
void norm_minmax(value_t *data, int n, int m);
void norm_standard(value_t *data, int n, int m);


#endif // _NORMALIZATION_H
