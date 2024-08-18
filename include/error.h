#ifndef _ML_ERROR_H
#define _ML_ERROR_H


/*
 * If an error occurs we have saved:
 * 1) Name of the function the user called
 * 2) File in which the function was called
 * 3) Line in which the function was called
 */
typedef struct {
    const char *func;
    const char *file;
    int line;
} ml_error_info_t;

extern _Thread_local ml_error_info_t ml_error_info;

/*
 * Macro that updates the error info struct.
 * Inject this into user function arguments like:
 * - #define <func>(args...) (__ml_error_update(<func>), _<func>(args))
 */
#define __ml_error_update(_func)        \
    (                                   \
        ml_error_info.func = #_func,    \
        ml_error_info.file = __FILE__,  \
        ml_error_info.line = __LINE__   \
    )

#endif // _ML_ERROR_H
