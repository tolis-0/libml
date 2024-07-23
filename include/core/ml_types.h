#ifndef _ML_TYPES
#define _ML_TYPES


#define _STD_ML_TYPE_ double


typedef _STD_ML_TYPE_ weight_t;    // type used for weights
typedef _STD_ML_TYPE_ value_t;     // type used for data values
typedef _STD_ML_TYPE_ grad_t;      // type used for gradients

typedef int dim_t[2];       // dimensions of matrix (i,j)
typedef int dim3_t[3];      // dimensions of matrix (i,j) including batch size k


#endif // _ML_TYPES
