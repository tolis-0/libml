#ifndef _OPT_INTERNAL_H
#define _OPT_INTERNAL_H


/*  opt/opt_mem.c declarations */
void _opt_alloc_val(nn_struct_t *nn,
    const char *func, const char *file, int line);
void _opt_free_val(nn_struct_t *nn);


#endif // _OPT_INTERNAL_H
