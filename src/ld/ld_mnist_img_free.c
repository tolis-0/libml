#include <stdlib.h>
#include "../../include/loader.h"


void ld_mnist_img_free(ld_img_data_t img_data)
{
    free(img_data.train_images);
    free(img_data.train_labels);
    free(img_data.test_images);
    free(img_data.test_labels);
}
