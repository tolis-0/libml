/*  Setting default values for nn and batch_outputs */
#ifndef __NN__
#   define __NN__ nn
#   define __NN__DO_UNDEF
#endif

#ifndef __BATCH_OUTPUTS__
#   define __BATCH_OUTPUTS__ batch_outputs
#   define __BATCH_OUTPUTS__DO_UNDEF
#endif


/*  Defining the function body to free memory from
    batch outputs, the ones array and optionally gradients */
{
    for (int i = 0; i < __NN__->n_layers; i++) {
        free(__NN__->__BATCH_OUTPUTS__[i]);
        __NN__->__BATCH_OUTPUTS__[i] = NULL;
    }

    /*  stores the batch input but is not allocated */
    __NN__->__BATCH_OUTPUTS__[-1] = NULL;

    free(__NN__->ones);
    __NN__->ones = NULL;

#ifdef __NN_FREE_GRADIENTS__
    free(__NN__->g_w);
    free(__NN__->g_b);
    free(__NN__->g_in);
    free(__NN__->g_out);

    __NN__->g_w = NULL;
    __NN__->g_b = NULL;
    __NN__->g_out = NULL;
    __NN__->g_in = NULL;
#endif
}


/*  Removing default values */
#ifdef __NN__DO_UNDEF
#   undef __NN__
#endif

#ifdef __BATCH_OUTPUTS__DO_UNDEF
#   undef __BATCH_OUTPUTS__
#endif
