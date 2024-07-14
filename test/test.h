#ifndef _TEST_H
#define _TEST_H


/*  Helper string macros for printing test info */
#define __title(str) puts("Testing function \e[1;39m" str "\e[0;39m:");
#define __test_passed "\e[32mPassed\e[39m"
#define __test_failed "\e[31mFailed\e[39m"


/*  Macro to check equality between double values */
#define __are_equal(x, y, e) ((((x) > (y)) ? (x) - (y) : (y) - (x)) < (e))


/*  Macro to compare results with expected values */
#define __exp_check(name, n, y, exp_y, error)               \
    do {                                                    \
        int i, correct = 0;                                 \
        for (i = 0; i < n; i++) {                           \
            if (__are_equal(y[i], exp_y[i], (error))) {     \
                correct++;                                  \
            } else {                                        \
                printf("At %d: expected %.10lf, got "       \
                    "%.10lf\n", i, exp_y[i], y[i]);         \
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


#endif // _TEST_H
