#ifndef _ERROR_INTERNAL_H
#define _ERROR_INTERNAL_H


#include "macros_internal.h"
#include "../include/error.h"


/*
 * Assert that cond is true. Throw an error if it isn't
 */
#define __ml_assert(cond, str, ...)                 \
    do {                                            \
        if (__unlikely(!(cond)))                    \
            _ml_throw_error(str, ##__VA_ARGS__);    \
    } while (0)

void _ml_throw_error(const char *str, ...);


/*
 * Check if malloc/calloc/realloc calls are successful
 */
#define __ml_malloc_check(var, type, n)  __ml_alloc_check(var, type, n, malloc)
#define __ml_calloc_check(var, type, n)  __ml_alloc_check(var, type, n, calloc)
#define __ml_realloc_check(var, type, n) __ml_alloc_check(var, type, n, realloc)

#define __ml_malloc_call(_, A, B)      malloc((A) * (B))
#define __ml_calloc_call(_, A, B)      calloc((A) , (B))
#define __ml_realloc_call(ptr, A, B)   realloc(ptr, (A) * (B))

#define __ml_alloc_check(ptr, type, _n, token)          \
    do {                                                \
        const size_t n = _n;                            \
        const size_t type_size = sizeof(type);          \
        /* expands to malloc(n * type_size)             \
         * or         calloc(n , type_size)             \
         * or         realloc(ptr, n * type_size) */    \
        ptr = __ml_##token##_call(ptr, n, type_size);   \
        __ml_assert(ptr != NULL, #token " "             \
            "failed to allocate memory for " #ptr " "   \
            "(requested array size: %zu, "              \
            "type: " #type ", "                         \
            "size of type: %zu)", n, type_size);        \
    } while (0)


#endif // _ERROR_INTERNAL_H
