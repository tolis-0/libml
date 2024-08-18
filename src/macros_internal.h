#ifndef _MACROS_INTERNAL_H
#define _MACROS_INTERNAL_H 


#if defined(__GNUC__) || defined(__clang__)
#   define __likely(x)     __builtin_expect(!!(x), 1)
#   define __unlikely(x)   __builtin_expect(!!(x), 0)
#else
#   define __likely(x)     (x)
#   define __unlikely(x)   (x)
#endif


#endif // _MACROS_INTERNAL_H
