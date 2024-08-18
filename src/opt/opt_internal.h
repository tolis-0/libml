#ifndef _OPT_INTERNAL_H
#define _OPT_INTERNAL_H

#include "../error_internal.h"


/*  opt/opt_mem.c declarations */
void _opt_alloc_val(nn_struct_t *nn);
void _opt_free_val(int n, void **opt_addr);


#endif // _OPT_INTERNAL_H
