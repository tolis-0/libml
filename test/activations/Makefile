CC = gcc
CFLAGS = -g3 -Wall -Wextra -fsanitize=address -fsanitize=undefined
LIBFLAGS = -lopenblas -lm

SOURCEF = $(wildcard *.c)
EXEF = $(patsubst %.c, %.out, $(SOURCEF))
TESTDIR = ../../src/layers/activations

all: $(EXEF)

%_forward.out: %_forward.c $(TESTDIR)/%.c
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

%_backward.out: %_backward.c $(TESTDIR)/%.c
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

clean:
	find . -type f -executable -delete

run: $(EXEF)
	@for test in $(EXEF); do \
		./$$test; \
		echo "\n"; \
	done
	@echo "Done"

.PHONY: all clean run
