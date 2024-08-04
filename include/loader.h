#ifndef _LOADER_H
#define _LOADER_H

#include <stdint.h>
#include "core/ml_types.h"


/*  Types in the third byte code of the magic number in mnist file format */
#define UBYTE_TYPE  0x08
#define SBYTE_TYPE  0x09
#define SHORT_TYPE  0x0B
#define INT_TYPE    0x0C
#define FLOAT_TYPE  0x0D
#define DOUBLE_TYPE 0x0E


/*  loader.c declarations */
void *_mnist_load_alloc(const char *filename, uint8_t type, uint8_t dim,
    const char *file, int line, ...);
value_t *ml_ubyte_convert(uint8_t *data, int n);
value_t *ml_sbyte_convert(int8_t *data, int n);
value_t *ml_short_convert(short *data, int n);
value_t *ml_int_convert(int *data, int n);
value_t *ml_float_convert(float *data, int n);
value_t *ml_ubyte_onehot(uint8_t *data, int n, int categories);


/*  Macros that provide debugging info */
#define mnist_load_alloc(fn, type, dim, ...) \
    _mnist_load_alloc(fn, type, dim, __FILE__, __LINE__, ##__VA_ARGS__)


#endif // _LOADER_H
