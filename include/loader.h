#ifndef _ML_LOADER_H
#define _ML_LOADER_H

#include <stdint.h>
#include "core/ml_types.h"
#include "error.h"


/* Types in the third byte code of the magic number in mnist file format */
#define UBYTE_TYPE  0x08
#define SBYTE_TYPE  0x09
#define SHORT_TYPE  0x0B
#define INT_TYPE    0x0C
#define FLOAT_TYPE  0x0D
#define DOUBLE_TYPE 0x0E


/* mnist_alloc.c declarations */
void *_ld_mnist_alloc(const char *filename, int type, int dim, ...);

/* ld_convert.c declarations */
value_t *_ld_convert_ubyte(uint8_t *data, int n);
value_t *_ld_convert_sbyte(int8_t *data, int n);
value_t *_ld_convert_short(short *data, int n);
value_t *_ld_convert_int(int *data, int n);
value_t *_ld_convert_float(float *data, int n);
value_t *_ld_convert_double(double *data, int n);

/* ld_onehot.c declarations */
value_t *_ld_onehot_ubyte(uint8_t *data, int n, int categories);
value_t *_ld_onehot_sbyte(int8_t *data, int n, int categories);
value_t *_ld_onehot_short(short *data, int n, int categories);
value_t *_ld_onehot_int(int *data, int n, int categories);


/* Macros for loader/ functions */
#define ld_mnist_alloc(...) (__ml_error_update(ld_mnist_alloc), _ld_mnist_alloc(__VA_ARGS__))
#define ld_convert_ubyte(...) (__ml_error_update(ld_convert_ubyte), _ld_convert_ubyte(__VA_ARGS__))
#define ld_convert_sbyte(...) (__ml_error_update(ld_convert_sbyte), _ld_convert_sbyte(__VA_ARGS__))
#define ld_convert_short(...) (__ml_error_update(ld_convert_short), _ld_convert_short(__VA_ARGS__))
#define ld_convert_int(...) (__ml_error_update(ld_convert_int), _ld_convert_int(__VA_ARGS__))

#if   _STD_ML_TYPE_ == _ML_TYPE_DOUBLE_
#   define ld_convert_float(...) (__ml_error_update(ld_convert_float), _ld_convert_float(__VA_ARGS__))
#   define ld_convert_double _ld_convert_double
#elif _STD_ML_TYPE_ == _ML_TYPE_FLOAT_
#   define ld_convert_double(...) (__ml_error_update(ld_convert_double), _ld_convert_double(__VA_ARGS__))
#   define ld_convert_float _ld_convert_float
#endif

#define ld_onehot_ubyte(...) (__ml_error_update(ld_onehot_ubyte), _ld_onehot_ubyte(__VA_ARGS__))
#define ld_onehot_sbyte(...) (__ml_error_update(ld_onehot_sbyte), _ld_onehot_sbyte(__VA_ARGS__))
#define ld_onehot_short(...) (__ml_error_update(ld_onehot_short), _ld_onehot_short(__VA_ARGS__))
#define ld_onehot_int(...) (__ml_error_update(ld_onehot_int), _ld_onehot_int(__VA_ARGS__))

#endif // _ML_LOADER_H
