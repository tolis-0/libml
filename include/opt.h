#ifndef _OPT_H
#define _OPT_H


#include "core/ml_types.h"


/*  Enum for recognizing types of structs */
typedef enum {
    GD_OPT,
    CM_OPT,
    NAG_OPT
} enum_opt_t;


/*  Struct for Classical Momentum optimizer */
typedef struct {
    value_t     beta;
    grad_t     *v;
} opt_cm_t;


/*  Struct for Nesterov Accelerated Gradient optimizer */
typedef struct {
    value_t     beta;
    grad_t     *v;
    grad_t     *gwv;
} opt_nag_t;


/*  Union for different Optimizer parameter structs */
typedef union {
    opt_cm_t    cm;
    opt_nag_t   nag;
} opt_t;


/*  Type of functions that apply optimization */
typedef void (*opt_func_t)(opt_t *, int, weight_t, cgrp_t, wrp_t);


/*  General struct for optimizers */
typedef struct {
    opt_func_t  call;
    enum_opt_t  type;
    opt_t       params;
    int         mem_alloc;
} ml_opt_t;


/*  Definition of struct that calls opt_create functions
    and declarations as extern */
typedef struct {
    ml_opt_t (*gd)();
    ml_opt_t (*cm)(weight_t beta);
    ml_opt_t (*nag)(weight_t beta);
} opt_create_t;

extern const opt_create_t opt_create;


/*  opt/opt_create.c declarations */
ml_opt_t opt_create_gd();
ml_opt_t opt_create_cm(weight_t beta);
ml_opt_t opt_create_nag(weight_t beta);


/*  opt/opt_apply.c declarations */
void opt_apply_gd(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w);
void opt_apply_cm(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w);
void opt_apply_nag(opt_t *o, int n, weight_t lr, cgrp_t grad, wrp_t w);


#endif // _OPT_H
