#include <stdio.h>
#include <stdlib.h>
#include <ml/loader.h>
#include <ml/normalization.h>
#include <ml/nn.h>

#define TRAIN_SIZE  1000
#define TEST_SIZE   10000


/*
 *  To run this example do:
 *  gcc -O2 first.c -o first.out -lml -lm -lopenblas
 *  ./first.out
*/

void print_nn_struct(nn_struct_t* nn);

const char *nn_activ_str[] = {
    "Linear",
    "ReLU",
    "Logistic",
    NULL
};

int main ()
{
    uint8_t *ub_data;
    value_t *data, *labels, *test_data, *test_labels;

    ub_data = mnist_load_alloc("data/train-images.idx3-ubyte", UBYTE_TYPE, 3);
    data = ml_ubyte_convert(ub_data, TRAIN_SIZE*28*28);
    norm_minmax(data, TRAIN_SIZE, 28*28);

    ub_data = mnist_load_alloc("data/train-labels.idx1-ubyte", UBYTE_TYPE, 1);
    labels = ml_ubyte_onehot(ub_data, TRAIN_SIZE, 10);

    ub_data = mnist_load_alloc("data/t10k-images.idx3-ubyte", UBYTE_TYPE, 3);
    test_data = ml_ubyte_convert(ub_data, TEST_SIZE*28*28);
    norm_minmax(test_data, TEST_SIZE, 28*28);

    ub_data = mnist_load_alloc("data/t10k-labels.idx1-ubyte", UBYTE_TYPE, 1);
    test_labels = ml_ubyte_onehot(ub_data, TEST_SIZE, 10);

    nn_spec_t mlp_spec[] = {
        input_layer(28*28),
        dense_layer(40, b, lrelu),
        dense_layer(20, b, lrelu),
        dense_layer(10, b, logistic),
        output_layer()
    };

    nn_struct_t *mlp = nn_create(mlp_spec);
    print_nn_struct(mlp);

    mlp->learning_rate = 0.3;
    mlp->opt = opt_create.cm(0.5);

    nn_train(mlp, 500, 1000, TRAIN_SIZE, data, labels);

    const float train_accuracy = nn_test(mlp, TRAIN_SIZE, data, labels);
    const float test_accuracy = nn_test(mlp, TEST_SIZE, test_data, test_labels);
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


#define print_struct_helper(str, type, var) \
    printf(str ": "); \
    for (int i = 0; i < n; i++) printf(type " ", nn->var[i]); \
    putchar('\n');

void print_nn_struct(nn_struct_t* nn)
{
    int n = nn->n_layers;
    printf("Number of layers: %d\n", n);
    printf("Number of weights: %d\n", nn->total_weights);
    printf("Number of biases: %d\n", nn->total_biases);
    printf("Starting weights pointer: %p\n", nn->weights_ptr);
    printf("Starting biases pointer: %p\n", nn->biases_ptr);

    print_struct_helper("Dims", "%d", n_dims);
    print_struct_helper("Weights", "%d", n_weights);
    print_struct_helper("Biases", "%d", n_biases);
    print_struct_helper("Reguralization", "%d", reg_type);
    print_struct_helper("Parameter", "%f", reg_p);
    print_struct_helper("Weight ptrs", "%p", weights);
    print_struct_helper("Bias ptrs", "%p", biases);
}
