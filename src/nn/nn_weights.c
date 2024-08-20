#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "../../include/nn.h"
#include "nn_internal.h"


#define inv_sqrt2   __ml_fpc(0.70710678118) /* 1/sqrt(2) */
#define FP_0        __ml_fpc(0.0)
#define FP_1        __ml_fpc(1.0)
#define FP_2        __ml_fpc(2.0)


static unsigned _rand_seed(void)
{
    unsigned seed = (unsigned) (unsigned long long) &seed;
    seed ^= (unsigned) time(NULL);
    seed ^= (unsigned) clock();

    srand(seed);

    return seed;
}


/*
 * He initialization for ReLU activation
 */
static void _he_init(wrp_t w, int n)
{
    const weight_t stddev = _sqrt(FP_2 / n);
    const weight_t mult = FP_2 / (weight_t) RAND_MAX;
    weight_t u, v, s;

    /* Use Box–Muller transform to generate pairs of
     * gaussian random numbers */
    for (int i = 1; i < n; i += 2) {
        do {
            u = rand() * mult - FP_1;
            v = rand() * mult - FP_1;
            s = u * u + v * v;
        } while (s >= FP_1 || s == FP_0);

        s = _sqrt(-FP_2 * _log(s) / s) * stddev;

        w[i]   = u * s;
        w[i-1] = v * s;
    }

    /* In case n is odd, the previous loop did not
     * handle the final weight */
    if (n & 1) {
        const size_t x = rand() % (n - 1);
        const size_t y = rand() % (n - 1);
        const weight_t swap = (w[x] + w[y]) * inv_sqrt2;
        w[n-1] = w[x];
        w[x] = swap;
    }
}


/*
 * Glorot/Xavier initialization for the rest
 */
static void _glorot_init(wrp_t w, int n, int sum)
{
    const weight_t range = _sqrt(__ml_fpc(6.0) / sum);
    const weight_t mult = (__ml_fpc(2.0) * range) / (weight_t) RAND_MAX;

    for (int i = 0; i < n; i++)
        w[i] = rand() * mult - range;
}


/*
 * Function to randomize the starting weights of a network
 */
void _nn_rand_weights(nn_struct_t *nn)
{
    nn->seed = _rand_seed();

    for (int i = 0; i < nn->num_of_layers; i++) {
        weight_t *restrict const w = nn->weights.of_layer[i];
        const int n = nn->weights.num[i];

        if (w == NULL) continue;
        if ((i < n - 1) && (nn->operation_type[i] == RELU_OP)) {
            _he_init(w, n);
        } else {
            _glorot_init(w, n, nn->num_of_dims[i-1] + nn->num_of_dims[i]);
        }
    }
}
