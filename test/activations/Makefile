CC = gcc
CFLAGS = -g3 -Wall -Wextra -fsanitize=address -fsanitize=undefined -fstack-protector
LIBFLAGS = -lopenblas -lm

SOURCEF = $(wildcard *.c)
EXEF = $(patsubst %.c, %.out, $(SOURCEF))
TESTF = ../../src/activations.c

all: $(EXEF)

%.out: %.c $(TESTF)
	$(CC) $(CFLAGS) -o $@ $< $(TESTF) $(LIBFLAGS)

clean:
	find . -type f -executable -delete

run: $(EXEF)
	@for test in $(EXEF); do \
		./$$test; \
		echo "\n"; \
	done
	@echo "Done"

.PHONY: all clean run
