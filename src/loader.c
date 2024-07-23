#include <stdlib.h>
#include <stdio.h>
#include "../include/loader.h"


#define __print_error   "\e[1;31merror\e[0;39m"
#define __print_warning "\e[1;35mwarning\e[0;39m"
#define __action_error  exit(1)
#define __action_warning

#define _mnist_load_alloc_error(type, cond, str, ...)               \
    if (cond) {                                                     \
        fprintf(stderr, "\e[1mmnist_load_alloc\e[0m"                \
            " (from \e[1;39m%s:%d\e[0;39m) " __print_##type ": "    \
            str "\n", file, line, ##__VA_ARGS__);                   \
        __action_##type;                                            \
    }


const char *mnist_type_str[] = {
    "ubyte",
    "sbyte",
    "short",
    "int",
    "float",
    "double",
    "unknown",
    NULL
};


/*  MNIST type helper function that returns a string
    with the corresponding type */
const char *mnist_th(uint8_t type)
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


/*  MNIST size helper function that returns corresponding size */
unsigned int mnist_sh(uint8_t type)
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


/*  Loader function for mnist type ubyte files.
    Allocates memory for the data that needs to be freed by the user. */
void *_mnist_load_alloc(const char *filename, uint8_t type, uint8_t dim,
    const char *file, int line)
{
    int i, read_items, n_items, type_size, total_size;
    uint8_t *data = NULL, metadata[4];
    uint32_t dimensions[5];
    FILE *fs = fopen(filename, "rb");

    _mnist_load_alloc_error(error, fs == NULL,
        "failed to open file with name %s", filename);

    read_items = fread(metadata, sizeof(uint8_t), 4, fs);

    /*  Doing various checks based on the magic number */
    _mnist_load_alloc_error(error, read_items != 4,
        "magic number read %d/4 bytes in %s",
        read_items, filename);

    _mnist_load_alloc_error(error, ((uint16_t *) metadata)[0] != 0x0000,
        "magic number first 2 bytes are 0x%04X/0x0000 in %s",
        ((uint16_t *) metadata)[0], filename);

    _mnist_load_alloc_error(error, mnist_th(metadata[2]) == mnist_type_str[6],
        "expected data type is %s (0x%02X) in %s",
        mnist_type_str[6], metadata[2], filename);

    _mnist_load_alloc_error(warning, metadata[2] != type,
        "expected data type %s, not %s in %s",
        mnist_th(type), mnist_th(metadata[2]), filename);

    _mnist_load_alloc_error(warning, metadata[3] != dim,
        "expected dimension size %d, not %d in %s",
        metadata[3], dim, filename);

    _mnist_load_alloc_error(error, metadata[3] > 5U,
        "number of dimensions %d is too large in %s",
        metadata[3], filename);

    read_items = fread(dimensions, sizeof(uint32_t), metadata[3], fs);

    /*  MNIST files are normally stored in big-endian format */
    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        for (i = 0; i < metadata[3]; i++)
            dimensions[i] = __builtin_bswap32(dimensions[i]);
    #endif

    _mnist_load_alloc_error(error, read_items != metadata[3],
        "dimensions read %d/%d in %s",
        read_items, metadata[3], filename);

    for (i = 0, n_items = 1; i < metadata[3]; i++) {
        _mnist_load_alloc_error(error, dimensions[i] == 0,
            "dimension %d is 0 in %s", i, filename);
        n_items *= dimensions[i];
    }

    type_size = mnist_sh(metadata[2]);
    total_size = n_items * type_size;
    data = malloc(total_size);

    _mnist_load_alloc_error(error, data == NULL,
        "memory allocation of data failed in %s (size %d)",
            filename, total_size);

    read_items = fread(data, type_size, n_items, fs);

    #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
        switch (type_size) {
            case 2:;
                uint16_t *data16 = (uint16_t *) data;
                for (i = 0; i < n_items; i++)
                    data16[i] = __builtin_bswap16(data16[i]);
                break;
            case 4:;
                uint32_t *data32 = (uint32_t *) data;
                for (i = 0; i < n_items; i++)
                    data32[i] = __builtin_bswap32(data32[i]);
                break;
            case 8:;
                uint64_t *data64 = (uint64_t *) data;
                for (i = 0; i < n_items; i++)
                    data64[i] = __builtin_bswap64(data64[i]);
                break;
            default:;
        }
    #endif

    _mnist_load_alloc_error(error, read_items != n_items,
        "data read %d/%d in %s",
        read_items, n_items, filename);

    fclose(fs);

    return data;
}


#define define_ml_convert_f(name, type)         \
value_t *ml_##name##_convert(type *data, int n) \
{                                               \
    int i;                                      \
    value_t *new;                               \
                                                \
    new = malloc(n * sizeof(value_t));          \
                                                \
    for (i = 0; i < n; i++)                     \
        new[i] = (value_t) data[i];             \
                                                \
    free(data);                                 \
    return new;                                 \
}

define_ml_convert_f(ubyte, uint8_t);
define_ml_convert_f(sbyte, int8_t);
define_ml_convert_f(short, short);
define_ml_convert_f(int, int);

#if   _STD_ML_TYPE_ == _ML_TYPE_DOUBLE_
    define_ml_convert_f(float, float);
#elif _STD_ML_TYPE_ == _ML_TYPE_FLOAT_
    define_ml_convert_f(double, double);
#endif


value_t *ml_ubyte_onehot(uint8_t *data, int n, int categories)
{
    int i;
    value_t *new;

    new = calloc(n * categories, sizeof(value_t));

    for (i = 0; i < n; i++)
        new[i*categories + data[i]] = 1.0;

    free(data);
    return new;
}
