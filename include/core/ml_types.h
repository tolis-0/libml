#ifndef _ML_TYPES
#define _ML_TYPES


/*  Available standard types */
#define _ML_TYPE_FLOAT_  1
#define _ML_TYPE_DOUBLE_ 2


/*  Used standard type */
#define _STD_ML_TYPE_ _ML_TYPE_FLOAT_


#if   _STD_ML_TYPE_ == _ML_TYPE_DOUBLE_
#   define  STD_ML_TYPE         double
#   define  __cblas_func(x)     cblas_d##x
#elif _STD_ML_TYPE_ == _ML_TYPE_FLOAT_
#   define  STD_ML_TYPE         float
#   define  __cblas_func(x)     cblas_s##x
#else
#   error "Standard type must be either float or double"
#endif


/*  Library types that indicate different behaviors */
typedef STD_ML_TYPE weight_t;   // type used for weights
typedef STD_ML_TYPE value_t;    // type used for data values
typedef STD_ML_TYPE grad_t;     // type used for gradients

typedef int dim_t[2];   // dimensions of matrix (i,j)
typedef int dim3_t[3];  // dimensions of matrix (i,j) including batch size k


/*  Concise type variations mainly for function parameters */
typedef const weight_t *restrict    cwrp_t;
typedef const value_t *restrict     cvrp_t;
typedef const grad_t *restrict      cgrp_t;

typedef weight_t *restrict           wrp_t;
typedef value_t *restrict            vrp_t;
typedef grad_t *restrict             grp_t;

typedef const dim_t     cdim_t;
typedef const dim3_t    cdim3_t;


/*  Define cblas functions corresponding to the standard type */
#define cblas_nrm2  __cblas_func(nrm2)
#define cblas_axpy  __cblas_func(axpy)
#define cblas_copy  __cblas_func(copy)
#define cblas_scal  __cblas_func(scal)
#define cblas_gemv  __cblas_func(gemv)
#define cblas_ger   __cblas_func(ger)
#define cblas_gemm  __cblas_func(gemm)


#endif // _ML_TYPES
