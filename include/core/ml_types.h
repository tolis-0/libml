#ifndef _ML_TYPES
#define _ML_TYPES

#define _ML_TYPE_FLOAT_  1
#define _ML_TYPE_DOUBLE_ 2


#define _STD_ML_TYPE_ _ML_TYPE_FLOAT_
#define STD_ML_TYPE float


typedef STD_ML_TYPE weight_t;   // type used for weights
typedef STD_ML_TYPE value_t;    // type used for data values
typedef STD_ML_TYPE grad_t;     // type used for gradients

typedef int dim_t[2];   // dimensions of matrix (i,j)
typedef int dim3_t[3];  // dimensions of matrix (i,j) including batch size k


/*  Define cblas functions corresponding to the standard type */
#if   _STD_ML_TYPE_ == _ML_TYPE_DOUBLE_
#   define __cblas_func(x) cblas_d##x
#elif _STD_ML_TYPE_ == _ML_TYPE_FLOAT_
#   define __cblas_func(x) cblas_s##x
#else
#   error "_STD_ML_TYPE_ must be either float or double"
#endif

#define cblas_nrm2  __cblas_func(nrm2)
#define cblas_axpy  __cblas_func(axpy)
#define cblas_copy  __cblas_func(copy)
#define cblas_scal  __cblas_func(scal)
#define cblas_gemv  __cblas_func(gemv)
#define cblas_ger   __cblas_func(ger)
#define cblas_gemm  __cblas_func(gemm)


#endif // _ML_TYPES
