#include <stdlib.h>
#include "../error_internal.h"
#include "../../include/loader.h"


#define define_ld_onehot(name, type)                                \
    value_t *_ld_onehot_##name(type *data, int n, int categories)   \
    {                                                               \
        value_t *new;                                               \
                                                                    \
        __ml_calloc_check(new, value_t, n * categories);            \
                                                                    \
        for (int i = 0; i < n; i++)                                 \
            new[i*categories + data[i]] = __ml_fpc(1.0);            \
                                                                    \
        free(data);                                                 \
        return new;                                                 \
    }

define_ld_onehot(ubyte, uint8_t)
define_ld_onehot(sbyte, int8_t)
define_ld_onehot(short, short)
define_ld_onehot(int, int)
