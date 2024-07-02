#include <stdio.h>
#include "nn.h"

void print_nn_struct(nn_struct_t* nn);

const char *nn_activ_str[] = {
    "Linear",
    "ReLU",
    "Logistic",
    NULL
};

int main ()
{
    nn_spec_t mlp_spec[] = {
        input_layer(3),
        dense_layer(8, b, relu, l2(1.0e-4)),
        dense_layer(10, b, linear, l2(1.0e-4)),
        relu_layer(),
        dense_layer(6, x, logistic),
        output_layer()
    };

    nn_struct_t *mlp = nn_create(mlp_spec);
    print_nn_struct(mlp);

    double x[3] = {8.2, -4.5, 3.7};
    nn_forward_pass(mlp, x);

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
