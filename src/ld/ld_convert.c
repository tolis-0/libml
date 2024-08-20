#include <stdlib.h>
#include "../error_internal.h"
#include "../../include/loader.h"


#define define_ld_convert(name, type)               \
    value_t *_ld_convert_##name(type *data, int n)  \
    {                                               \
        value_t *new;                               \
                                                    \
        __ml_malloc_check(new, value_t, n);         \
                                                    \
        for (int i = 0; i < n; i++)                 \
            new[i] = (value_t) data[i];             \
                                                    \
        free(data);                                 \
        return new;                                 \
    }


define_ld_convert(ubyte, uint8_t)
define_ld_convert(sbyte, int8_t)
define_ld_convert(short, short)
define_ld_convert(int, int)


#if   _STD_ML_TYPE_ == _ML_TYPE_DOUBLE_
    define_ld_convert(float, float)
    value_t *_ld_convert_double(double *data, int n) {(void) n; return data;}
#elif _STD_ML_TYPE_ == _ML_TYPE_FLOAT_
    define_ld_convert(double, double)
    value_t *_ld_convert_float(float *data, int n) {(void) n; return data;}
#endif
