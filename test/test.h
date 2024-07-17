#ifndef _TEST_H
#define _TEST_H


/*  Helper string macros for printing test info */
#define __title(str) puts("Testing function \e[1;39m" str "\e[0;39m:");
#define __test_passed "\e[32mPassed\e[39m"
#define __test_failed "\e[31mFailed\e[39m"


/*  Macros for stringizing code */
#define __to_str_exp(x) __to_str(x) // allows x to expand first
#define __to_str(x) #x


/*  Helper macros to count elements of an array */
#define __arr_count(x) sizeof(x) / sizeof((x)[0])
#define __arr_count_null(x)                 \
    ((x != NULL) ? sizeof(x)      : 0) /    \
    ((x != NULL) ? sizeof((x)[0]) : 1)


/*  Macros to check equality between numbers */
#define __are_equal(type, x, y, e) (__are_equal_##type(x, y, e))
#define __are_equal_lf(x, y, e) ((((x) > (y)) ? (x) - (y) : (y) - (x)) < (e))
#define __are_equal_d(x, y, e) ((x) == (y))


/*  Macros for initializing variables and arrays */
#define __nan_val __builtin_nan("")                         // invalid number
#define __nan_array(size) {[0 ... (size)-1] = __nan_val}    // output array
#define __val_array(size, val) {[0 ... (size)-1] = (val)}   // single value array


/*  Macros to help print different types of values */
#define __print_lf  "%.10lf"
#define __print_d   "%d"


/*  Macro to compare results with expected values
    type: is either lf (double) or d (int)
    name: text to print that recognizes that specific test
    n: number of elements to compare
    y: array that gets compared with respective exp_y array
    e: absolute error for floating-point values             */
#define __exp_check(type, name, n, y, e)                    \
    do {                                                    \
        int i, correct = 0;                                 \
        for (i = 0; i < n; i++) {                           \
            if (__are_equal(type, y[i], exp_##y[i], (e))) { \
                correct++;                                  \
            } else {                                        \
                printf("At %d: expected " __print_##type    \
                    ", got " __print_##type "\n",           \
                    i, exp_##y[i], y[i]);                   \
            }                                               \
        }                                                   \
        if (correct == n) {                                 \
            printf(name " %d/%d " __test_passed "\n",       \
                correct, n);                                \
        } else {                                            \
            printf(name " %d/%d " __test_failed "\n",       \
                correct, n);                                \
        }                                                   \
    } while (0)


/*  Handling different types for __exp_check */
#define __exp_check_lf(name, n, y, e) \
    __exp_check(lf, name, n, y, e)
#define __exp_check_d(name, n, y) \
    __exp_check(d, name, n, y, 0)



#endif // _TEST_H
