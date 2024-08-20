#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "../error_internal.h"
#include "../../include/loader.h"


static const char *mnist_type_str[] = {
    "ubyte",
    "sbyte",
    "short",
    "int",
    "float",
    "double",
    "unknown",
    NULL
};


static const char *mnist_th(uint8_t type)
{
    switch (type) {
        case UBYTE_TYPE:  return mnist_type_str[0];
        case SBYTE_TYPE:  return mnist_type_str[1];
        case SHORT_TYPE:  return mnist_type_str[2];
        case INT_TYPE:    return mnist_type_str[3];
        case FLOAT_TYPE:  return mnist_type_str[4];
        case DOUBLE_TYPE: return mnist_type_str[5];
        default:          return mnist_type_str[6];
    }
}


static unsigned int mnist_sh(uint8_t type)
{
    switch (type) {
        case UBYTE_TYPE:  return sizeof(uint8_t);
        case SBYTE_TYPE:  return sizeof(int8_t);
        case SHORT_TYPE:  return sizeof(short);
        case INT_TYPE:    return sizeof(int);
        case FLOAT_TYPE:  return sizeof(float);
        case DOUBLE_TYPE: return sizeof(double);
        default:          return 0;
    }
}

#define is_valid_mnist_type(metadata) (mnist_th(metadata[2]) != mnist_type_str[6])
#define check_first_bytes(metadata) (((uint16_t *) metadata)[0] == 0x0000U)

static void _ld_magicnum_check(uint8_t md[4], int items, uint8_t type, uint8_t ndims,
    const char *filename)
{
    __ml_assert(items == 4,
        "magic number read %d/4 bytes in %s", items, filename);

    __ml_assert(check_first_bytes(md),
        "magic number first 2 bytes are not zero in %s", filename);

    __ml_assert(is_valid_mnist_type(md),
        "unknown data type in %s", filename);

    __ml_assert(md[2] == type,
        "expected data type %s, not %s in %s", mnist_th(type), mnist_th(md[2]), filename);

    __ml_assert(md[3] == ndims,
        "expected dimension size %d, not %d in %s", md[3], ndims, filename);

    __ml_assert(md[3] <= 4U,
        "number of dimensions %d is too large in %s", md[3], filename);
}


static void _ld_data_fix_endianess(int type_size, int n_items, void *data)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#   define case_macro(byte, bit)                                \
        case byte:;                                             \
            uint##bit##_t *data##bit = data;                    \
            for (int i = 0; i < n_items; i++)                   \
                data##bit[i] = __byteswap##bit(data##bit[i]);   \
            break;

        switch (type_size) {
            case_macro(2, 16)
            case_macro(4, 32)
            case_macro(8, 64)
            default:;
        }
#   undef case_macro
#else
    (void) type_size;
    (void) n_items;
    (void) data;
#endif
}

/* MNIST files are normally stored in big-endian format */
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#   define __swap_if_little_endian(x) (x = __byteswap32(x))
#else
#   define __swap_if_little_endian(_) ((void) 0)
#endif


/* Helper macro that operates with va_args directly from the function */
#define _ld_dims_check(md, items, dim, ndims_var, filename)             \
    do {                                                                \
        va_list args;                                                   \
        va_start(args, ndims_var); /* this requires the                 \
                                    * actual variable name */           \
        __ml_assert(items == md[3],                                     \
            "dimensions read %d/%d in %s", items, md[3], filename);     \
                                                                        \
        for (int i = 0; i < ndims_var; i++) {                           \
            const uint32_t exp_dim = va_arg(args, uint32_t);            \
                                                                        \
            __swap_if_little_endian(dim[i]);                            \
                                                                        \
            __ml_assert(i == 0 ? dim[i] >= exp_dim : dim[i] == exp_dim, \
                "dimension %d is %d/%d in %s",                          \
                i, dim[i], exp_dim, filename);                          \
        }                                                               \
                                                                        \
        va_end(args);                                                   \
    } while (0)


/*
 * Loader function for mnist type ubyte files.
 * Allocates memory for the data that needs to be freed by the user.
 */
void *_ld_mnist_alloc(const char *filename, int type, int ndims, ...)
{
    int i, read_items, n_items, type_size;
    uint8_t *data, metadata[4];
    uint32_t dimensions[4];
    FILE *fs;

    fs = fopen(filename, "rb");
    __ml_assert(fs != NULL, "failed to open file with name %s", filename);

    /* read magic number */
    read_items = fread(metadata, sizeof(uint8_t), 4, fs);
    _ld_magicnum_check(metadata, read_items, (uint8_t) type, (uint8_t) ndims, filename);

    /* read dimension numbers */
    read_items = fread(dimensions, sizeof(uint32_t), metadata[3], fs);
    _ld_dims_check(metadata, read_items, dimensions, ndims, filename);

    for (i = 0, n_items = 1; i < metadata[3]; i++) {
        __ml_assert(dimensions[i] > 0, "dimension %d is invalid in %s", i, filename);
        n_items *= dimensions[i];
    }

    /* allocate memory for the data */
    type_size = mnist_sh(metadata[2]);
    __ml_malloc_check(data, unsigned char, n_items * type_size);

    /* read the data */
    read_items = fread(data, type_size, n_items, fs);
    __ml_assert(read_items == n_items,
        "data read %d/%d in %s", read_items, n_items, filename);

    _ld_data_fix_endianess(type_size, n_items, data);

    fclose(fs);

    return data;
}
