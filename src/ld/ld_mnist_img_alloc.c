#include "../../include/loader.h"


ld_img_data_t _ld_mnist_img_alloc(const char *filename[4], const dim_t dims,
    int train_size, int test_size, int categories)
{
    uint8_t *ub_data;
    ld_img_data_t img_data;
    int d1 = dims[0], d2 = dims[1];

    ub_data = _ld_mnist_alloc(filename[0], UBYTE_TYPE, 3, train_size, d1, d2);
    img_data.train_images = _ld_convert_ubyte(ub_data, train_size*d1*d2);

    ub_data = _ld_mnist_alloc(filename[1], UBYTE_TYPE, 1, train_size);
    img_data.train_labels = _ld_onehot_ubyte(ub_data, train_size, categories);

    ub_data = _ld_mnist_alloc(filename[2], UBYTE_TYPE, 3, test_size, d1, d2);
    img_data.test_images = _ld_convert_ubyte(ub_data, test_size*d1*d2);

    ub_data = _ld_mnist_alloc(filename[3], UBYTE_TYPE, 1, test_size);
    img_data.test_labels = _ld_onehot_ubyte(ub_data, test_size, categories);

    return img_data;
}
