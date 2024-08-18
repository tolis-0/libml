#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include "error_internal.h"


_Thread_local ml_error_info_t ml_error_info = {.func = NULL, .file = NULL, .line = 0};


#if defined(__unix__) || defined(__linux__)
#   include <unistd.h>

    _Bool _ml_use_ansi_escape(void) {
        if (!isatty(STDERR_FILENO))
            return 0;

        return 1;
    }
#elif defined(_WIN32)
#   include <windows.h>
    // TODO
    _Bool _ml_use_ansi_escape(void) {return 0;}
#else
    _Bool _ml_use_ansi_escape(void) {return 0;}
#endif


/*
 * General function to print an error message in stderr and exit
 */
void _ml_throw_error(const char *str, ...)
{
    const char *const func = ml_error_info.func;
    const char *const file = ml_error_info.file;
    const int line         = ml_error_info.line;
    
    va_list args;
    va_start(args, str);

    if (_ml_use_ansi_escape()) {
        fprintf(stderr, "\e[1;39m%s\e[0;39m (from \e[1;39m%s:%d\e[0;39m) "
            "\e[1;31merror\e[0;39m: ", func, file, line);
    } else {
        fprintf(stderr, "%s (from %s:%d) error: ", func, file, line);
    }

    vfprintf(stderr, str, args);
    fputc('\n', stderr);

    va_end(args);
    exit(EXIT_FAILURE);
}
