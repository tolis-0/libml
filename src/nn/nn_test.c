#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include "../../include/nn.h"
#include "nn_internal.h"


#define _nn_test_error(cond, str, ...)                              \
    if (__builtin_expect(!!(cond), 0)) {                            \
        fprintf(stderr, "\e[1;39mnn_test\e[0;39m"                   \
            " (from \e[1;39m%s:%d\e[0;39m) \e[1;31merror\e[0;39m: " \
            str "\n", file, line, ##__VA_ARGS__);                   \
        exit(EXIT_FAILURE);                                         \
    }


/*  Tests the accuracy of the neural network */
float _nn_test(nn_struct_t *nn, int test_size, value_t *x, value_t *t,
    const char *file, int line)
{
    const int o_n = nn->output_dims;
    int i, j, arg_max, one_index, correct;
    value_t *t_ptr, *o_ptr;
    value_t val, max;


    _nn_alloc_batch(nn, test_size, "nn_test", file, line);

    nn->batch_outputs[-1] = x;
    nn_batch_forward_pass(nn, test_size);

    t_ptr = t;
    o_ptr = nn->output;
    correct = 0;

    for (i = 0; i < test_size; i++) {
        max = -DBL_MAX;
        arg_max = -1;
        one_index = -1;

        for (j = 0; j < o_n; j++) {
            val = o_ptr[j];
            one_index = (t_ptr[j] > 0.5) ? j : one_index;
            max = (val > max) ? (arg_max = j, val) : max;
        }

        _nn_test_error(arg_max < 0, "arg_max is %d", arg_max);
        _nn_test_error(one_index < 0, "one_index is %d", one_index);

        correct += (arg_max == one_index);

        t_ptr += o_n;
        o_ptr += o_n;
    }

    _nn_free_batch(nn);

    return correct / (float) test_size;
}
