#include "../../include/opt.h"


/*  Struct that calls opt_create functions */
const opt_create_t opt_create = {
    .gd     = opt_create_gd,
    .cm     = opt_create_cm,
    .nag    = opt_create_nag
};


/*  Initialize a GD optimizer */
ml_opt_t opt_create_gd(void)
{
    const ml_opt_t opt = {
        .call = opt_apply_gd,
        .type = GD_OPT,
        .mem_alloc = 0
    };

    return opt;
}


/*  Initialize a CM optimizer */
ml_opt_t opt_create_cm(weight_t beta)
{
    const opt_t u = {.cm = {.beta = beta}};
    const ml_opt_t opt = {
        .call = opt_apply_cm,
        .type = CM_OPT,
        .params = u,
        .mem_alloc = 0
    };

    return opt;
}


/*  Initialize a NAG optimizer */
ml_opt_t opt_create_nag(weight_t beta)
{
    const opt_t u = {.cm = {.beta = beta}};
    const ml_opt_t opt = {
        .call = opt_apply_nag,
        .type = NAG_OPT,
        .params = u,
        .mem_alloc = 0
    };

    return opt;
}
