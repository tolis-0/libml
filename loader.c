#include <stdlib.h>
#include <stdio.h>
#include "loader.h"


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
    int i, read_bytes, data_size;
    uint8_t *data = NULL, metadata[4];
    uint32_t dimensions[5];
    FILE *fs = fopen(filename, "rb");

    _mnist_load_alloc_error(error, fs == NULL,
        "failed to open file with name %s", filename);

    read_bytes = fread(metadata, sizeof(uint8_t), 4, fs);

    /*  Doing various checks based on the magic number */
    _mnist_load_alloc_error(error, read_bytes != 4,
        "magic number read %d/4 bytes in %s",
        read_bytes, filename);

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

    read_bytes = fread(dimensions, sizeof(uint32_t), metadata[3], fs);

    _mnist_load_alloc_error(error, read_bytes != metadata[3],
        "dimensions read %d/%d bytes in %s",
        read_bytes, metadata[3], filename);

    for (i = 0, data_size = 1; i < metadata[3]; i++) {
        _mnist_load_alloc_error(error, dimensions[i] == 0,
            "dimension %d is 0 in %s", i, filename);
        data_size *= dimensions[i];
    }

    data = malloc(data_size * mnist_sh(metadata[2]));

    _mnist_load_alloc_error(error, data == NULL,
        "memory allocation of data failed in %s (size %d)",
            filename, data_size * mnist_sh(metadata[2]));

    fclose(fs);

    return data;
}
