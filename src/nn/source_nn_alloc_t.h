/*  Setting default values for nn and batch_outputs */
#ifndef __NN__
#   define __NN__ nn
#   define __NN__DO_UNDEF
#endif

#ifndef __BATCH_OUTPUTS__
#   define __BATCH_OUTPUTS__ batch_outputs
#   define __BATCH_OUTPUTS__DO_UNDEF
#endif


/*  Defining the function body to allocate memory
    for batch outputs, the ones array and optionally gradients */
{
    int i;
#ifdef __NN_ALLOC_GRADIENTS__
    int max_dims, max_w, max_b;

    max_b = max_w = 0;
    max_dims = __NN__->input_dims;
#endif

    /*  TODO: add checks for malloc */
    for (i = 0; i < __NN__->n_layers; i++) {
        __NN__->__BATCH_OUTPUTS__[i] = malloc(
            __NN__->n_dims[i] * batch_size * sizeof(value_t));
#ifdef __NN_ALLOC_GRADIENTS__
        max_dims = (__NN__->n_dims[i] > max_dims) ? __NN__->n_dims[i] : max_dims;
        max_b = (__NN__->n_biases[i] > max_b) ? __NN__->n_biases[i] : max_b;
        max_w = (__NN__->n_weights[i] > max_w) ? __NN__->n_weights[i] : max_w;
#endif
    }

    __NN__->ones = malloc(batch_size * sizeof(value_t));
    for (i = 0; i < batch_size; i++)
        __NN__->ones[i] = 1.0;

#ifdef __NN_ALLOC_GRADIENTS__
    __NN__->g_w = malloc(max_w * sizeof(weight_t));
    __NN__->g_b = malloc(max_b * sizeof(weight_t));
    __NN__->g_out = malloc(batch_size * max_dims * sizeof(value_t));
    __NN__->g_in = malloc(batch_size * max_dims * sizeof(value_t));
#endif
}


/*  Removing default values */
#ifdef __NN__DO_UNDEF
#   undef __NN__
#endif

#ifdef __BATCH_OUTPUTS__DO_UNDEF
#   undef __BATCH_OUTPUTS__
#endif
