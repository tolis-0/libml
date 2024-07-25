#include <stdlib.h>
#include <math.h>
#include "../../include/nn.h"
#include "nn_internal.h"


#ifdef __x86_64__
#   include <stdint.h>
#   define __set_seed()                         \
        do {                                    \
            uint32_t lo, hi;                    \
            __asm__ __volatile__ (              \
                "rdtsc" : "=a" (lo), "=d" (hi)  \
            );                                  \
            srand(((uint64_t) hi << 32) | lo);  \
        } while (0)
#else
#   include <time.h>
#   define __set_seed() srand(time(NULL))
#endif


/*  He initialization function for ReLU */
void _he_init(weight_t *w, int n)
{
    const weight_t stddev = _sqrt(2.0 / n);
    const weight_t mult = 2.0 / (weight_t) RAND_MAX;
    weight_t u, v, s;

    for (int i = 0; i < n; i += 2) {
        do {
            u = rand() * mult - 1.0;
            v = rand() * mult - 1.0;
            s = u * u + v * v;
        } while (s >= 1.0 || s == 0.0);

        s = _sqrt(-2.0 * _log(s) / s);

        w[i]   = u * s * stddev;
        w[i+1] = v * s * stddev;
    }

    /*  In case n is odd, the previous loop did not
        handle the final weight */
    if (n & 1) {
        const int x = rand() % (n-1);
        const int y = rand() % (n-1);
        const weight_t swap = (w[x] + w[y]) * inv_sqrt2;
        w[n-1] = w[x];
        w[x] = swap;
    }
}


void _glorot_init(weight_t *w, int n, int sum)
{
    const weight_t range = _sqrt(6.0 / sum);
    const weight_t mult = (2.0 * range) / (weight_t) RAND_MAX;

    for (int i = 0; i < n; i++) {
        w[i] = rand() * mult - range;
    }
}


/*  Function to randomize the starting weights of a network */
void _nn_rand_weights(nn_struct_t *nn)
{
    __set_seed();

    for (int i = 0; i < nn->n_layers; i++) {
        weight_t *const w = nn->weights[i];
        const int n = nn->n_weights[i];

        if (w == NULL) continue;
        if ((i < n - 1) && (nn->op_types[i] == RELU_OP)) {
            _he_init(w, n);
        } else {
            _glorot_init(w, n, nn->n_dims[i-1] + nn->n_dims[i]);
        }
    }
}
