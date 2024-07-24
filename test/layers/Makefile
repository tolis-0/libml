CC = gcc
CFLAGS = -g3 -Wall -Wextra -fsanitize=address -fsanitize=undefined -fstack-protector
LIBFLAGS = -lopenblas

SOURCEF = $(wildcard *.c)
EXEF = $(patsubst %.c, %.out, $(SOURCEF))
TESTF = ../../src/layers/dense.c

all: $(EXEF)

%.out: %.c $(TESTF)
	$(CC) $(CFLAGS) -o $@ $< $(TESTF) $(LIBFLAGS)

clean:
	find . -type f -executable -delete

.PHONY: clean all
