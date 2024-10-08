#ifndef _TEST_H
#define _TEST_H

#include <stddef.h>
#include "../include/core/ml_types.h"


/* Helper string macros for printing test info */
#define __title(str) puts("Testing function \e[1;39m" str "\e[0;39m:");
#define __test_passed "\e[32mPassed\e[39m"
#define __test_failed "\e[31mFailed\e[39m"


/* Macros for stringizing code */
#define __to_str_exp(x) __to_str(x) // allows x to expand first
#define __to_str(x) #x


/* Helper macros to count elements of an array */
#define __arr_count(x) (sizeof(x) / sizeof((x)[0]))
#define __arr_count_null(x) (               \
    (((x) != NULL) ? sizeof(x)      : 0) /  \
    (((x) != NULL) ? sizeof((x)[0]) : 1))


/* Macro to check equality between numbers */
#define __are_equal(x, y, e)                                    \
    _Generic((x),                                               \
        float:  ((((x) > (y)) ? (x) - (y) : (y) - (x)) < (e)),  \
        double: ((((x) > (y)) ? (x) - (y) : (y) - (x)) < (e)),  \
        default: ((x) == (y))                                   \
    )


/* Macros for initializing variables and arrays */
#define __nan_val __builtin_nan("") // invalid number
#define __nan_array(size)       \
    {[0 ... (size)-1] = __nan_val}  // placeholder for output array
#define __val_array(size, val)  \
    {[0 ... (size)-1] = (val)}      // uniform array


/* Assert macros */
#define __assert_null(arr)      \
    assert((arr) == NULL)
#define __assert_not_null(arr)  \
    assert((arr) != NULL)
#define __assert_cond_null(cond, arr)   \
    ((cond) ? __assert_null(arr) : __assert_not_null(arr))
#define __assert_size(arr, n) \
    assert(__arr_count(arr) == (size_t) (n))
#define __assert_size_null(arr, n) \
    assert(__arr_count_null(arr) == (size_t) (n))
#define __Static_assert_size(arr, n)    \
    _Static_assert(__arr_count(arr) == (size_t) (n), "array size mismatch")
#define __Static_assert_size_null(arr, n)   \
    _Static_assert(__arr_count_null(arr) == (size_t) (n), "array size mismatch")
#define __Static_assert_cond_size(cond, arr, n)             \
    ((cond) ? _Static_assert(__arr_count_null(arr) == (size_t) (n),  \
        "array size mismatch") : (void) 0)


/* Macros to help print different types of values */
#if   _STD_ML_TYPE_ == _ML_TYPE_DOUBLE_
#   define __print_lf  "%.10lf"
#elif _STD_ML_TYPE_ == _ML_TYPE_FLOAT_
#   define __print_lf  "%.7f"
#endif
#define __print_d   "%d"


/*
 * Macro to compare results with expected values
 * type: is either lf (float/double) or d (int)
 * name: text to print that recognizes that specific test
 * n: number of elements to compare
 * y: array that gets compared with respective exp_y array
 * e: absolute error for floating-point values
 */
#define __exp_check(type, name, n, y, e)                    \
    do {                                                    \
        const int _n = (int) n;                             \
        int i, correct;                                     \
                                                            \
        for (i = correct = 0; i < _n; i++) {                \
            if (__are_equal(y[i], exp_##y[i], (e))) {       \
                correct++;                                  \
            } else {                                        \
                printf("At %d: expected " __print_##type    \
                    ", got " __print_##type "\n",           \
                    i, exp_##y[i], y[i]);                   \
            }                                               \
        }                                                   \
        if (correct == _n) {                                \
            printf(name " %d/%d " __test_passed "\n",       \
                correct, _n);                               \
        } else {                                            \
            printf(name " %d/%d " __test_failed "\n",       \
                correct, _n);                               \
        }                                                   \
    } while (0)


/* Handling different types for __exp_check */
#define __exp_check_f __exp_check_lf
#define __exp_check_lf(name, n, y, e) \
    __exp_check(lf, name, n, y, e)
#define __exp_check_d(name, n, y) \
    __exp_check(d, name, n, y, 1)


#endif // _TEST_H
