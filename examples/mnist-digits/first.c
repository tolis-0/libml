#include <stdio.h>
#include <stdlib.h>
#include <ml/loader.h>
#include <ml/normalization.h>
#include <ml/nn.h>

#define TRAIN_SIZE  60000
#define TEST_SIZE   10000


/*
 * To run this example do:
 * gcc -O2 first.c -o first.out -lml -lm -lopenblas
 * ./first.out
*/
int main ()
{
    uint8_t *ub_data;
    value_t *data, *labels, *test_data, *test_labels;

    ub_data = mnist_load_alloc("data/train-images.idx3-ubyte", UBYTE_TYPE, 3, TRAIN_SIZE, 28, 28);
    data = ml_ubyte_convert(ub_data, TRAIN_SIZE*28*28);
    norm_minmax(data, TRAIN_SIZE, 28*28);

    ub_data = mnist_load_alloc("data/train-labels.idx1-ubyte", UBYTE_TYPE, 1, TRAIN_SIZE);
    labels = ml_ubyte_onehot(ub_data, TRAIN_SIZE, 10);

    ub_data = mnist_load_alloc("data/t10k-images.idx3-ubyte", UBYTE_TYPE, 3, TEST_SIZE, 28, 28);
    test_data = ml_ubyte_convert(ub_data, TEST_SIZE*28*28);
    norm_minmax(test_data, TEST_SIZE, 28*28);

    ub_data = mnist_load_alloc("data/t10k-labels.idx1-ubyte", UBYTE_TYPE, 1, TEST_SIZE);
    test_labels = ml_ubyte_onehot(ub_data, TEST_SIZE, 10);

    nn_spec_t mlp_spec[] = {
        nnl_input(28*28),
        nnl_dense(40, 1, RELU_OP, NO_REG),
        nnl_dense(20, 1, RELU_OP, NO_REG),
        nnl_dense(10, 1, LOGISTIC_OP, NO_REG),
        NN_SPEC_END
    };

    nn_struct_t *mlp = nn_create(mlp_spec);
    mlp->learning_rate = 0.3;

    nn_train(mlp, 10, 100, TRAIN_SIZE, data, labels);

    const float train_accuracy = nn_accuracy(mlp, TRAIN_SIZE, data, labels);
    const float test_accuracy = nn_accuracy(mlp, TEST_SIZE, test_data, test_labels);
    const double train_loss = nn_loss(mlp, TRAIN_SIZE, data, labels);
    const double test_loss = nn_loss(mlp, TEST_SIZE, test_data, test_labels);
    printf("\nTrain Accuracy: %.2f%%\n", train_accuracy * 100.0);
    printf("Train Loss: %.10lf\n", train_loss);
    printf("Test Accuracy: %.2f%%\n", test_accuracy * 100.0);
    printf("Test Loss: %.10lf\n", test_loss);


    free(data);
    free(test_data);
    free(labels);
    free(test_labels);
    nn_destroy(mlp);

    return 0;
}
