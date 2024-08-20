#ifndef _MACROS_INTERNAL_H
#define _MACROS_INTERNAL_H 


#if defined(__GNUC__) || defined(__clang__)
#   define __likely(x)     __builtin_expect(!!(x), 1)
#   define __unlikely(x)   __builtin_expect(!!(x), 0)

#   define __byteswap16(x) __builtin_bswap16(x)
#   define __byteswap32(x) __builtin_bswap32(x)
#   define __byteswap64(x) __builtin_bswap64(x)
#else
#   define __likely(x)     (x)
#   define __unlikely(x)   (x)

#   define __byteswap16(x)          \
        (((x) & 0x00FFU) << 8) |    \
        (((x) & 0xFF00U) >> 8)

#   define __byteswap32(x)              \
        (((x) & 0x000000FFU) << 24) |   \
        (((x) & 0x0000FF00U) <<  8) |   \
        (((x) & 0x00FF0000U) >>  8) |   \
        (((x) & 0xFF000000U) >> 24)

#   define __byteswap64(x)                      \
        (((x) & 0x00000000000000FFULL) << 56) | \
        (((x) & 0x000000000000FF00ULL) << 40) | \
        (((x) & 0x0000000000FF0000ULL) << 24) | \
        (((x) & 0x00000000FF000000ULL) <<  8) | \
        (((x) & 0x000000FF00000000ULL) >>  8) | \
        (((x) & 0x0000FF0000000000ULL) >> 24) | \
        (((x) & 0x00FF000000000000ULL) >> 40) | \
        (((x) & 0xFF00000000000000ULL) >> 56)
#endif


#endif // _MACROS_INTERNAL_H
