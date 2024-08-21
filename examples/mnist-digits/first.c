#include <stdio.h>
#include <stdlib.h>
#include <ml/loader.h>
#include <ml/normalization.h>
#include <ml/nn.h>

/*
 * To run this example do:
 * gcc -O2 first.c -o first.out -lml -lm -lopenblas
 * ./first.out
*/

int main ()
{
    value_t *train_data, *train_labels, *test_data, *test_labels;
    ld_img_data_t data;

    const char *filenames[4] = {
        "data/train-images.idx3-ubyte",     /* train set images */
        "data/train-labels.idx1-ubyte",     /* test set images */
        "data/t10k-images.idx3-ubyte",      /* train set labels */
        "data/t10k-labels.idx1-ubyte"       /* test set labels */
    };

    const dim_t dimensions = {28, 28};
    const int img_size = dimensions[0] * dimensions[1];
    const int train_size = 60000;
    const int test_size = 10000;
    const int categories = 10;

    /* Allocate memory for the data and load them from the files */
    data = _ld_mnist_img_alloc(filenames, dimensions, train_size, test_size, categories);
    train_data = data.train_images, train_labels = data.train_labels;
    test_data = data.test_images, test_labels = data.test_labels;

    /* Normalize the data with min-max */
    norm_minmax(train_data, train_size, img_size);
    norm_minmax(test_data, test_size, img_size);

    /* Define the structure of the neural network */
    nn_spec_t mlp_spec[] = {
        nnl_input(img_size),
        nnl_dense(40, 1, RELU_OP, NO_REG),
        nnl_dense(20, 1, RELU_OP, NO_REG),
        nnl_dense(10, 1, LOGISTIC_OP, NO_REG),
        NN_SPEC_END
    };

    nn_struct_t *mlp = nn_create(mlp_spec);
    mlp->learning_rate = 0.3;

    nn_train(mlp, 10, 100, train_size, train_data, train_labels);

    const float train_accuracy = nn_accuracy(mlp, train_size, train_data, train_labels);
    const float test_accuracy = nn_accuracy(mlp, test_size, test_data, test_labels);
    const double train_loss = nn_loss(mlp, train_size, train_data, train_labels);
    const double test_loss = nn_loss(mlp, test_size, test_data, test_labels);
    printf("Train Accuracy: %.2f%%\n", train_accuracy * 100.0f);
    printf("Train Loss: %.10lf\n", train_loss);
    printf("Test Accuracy: %.2f%%\n", test_accuracy * 100.0f);
    printf("Test Loss: %.10lf\n", test_loss);


    ld_mnist_img_free(data);
    nn_destroy(mlp);

    return 0;
}
