CC = gcc
CFLAGS = -g3 -Wall -Wextra -fsanitize=address \
	-fsanitize=undefined -fstack-protector

DIR = ../../src

EXEF = $(patsubst %.c, %.out, $(wildcard *.c))


NN_CREATE_SOURCES = \
	$(DIR)/nn/nn_create.c \
	$(DIR)/nn/nn_destroy.c
NN_FORWARD_PASS_SOURCES = \
	$(DIR)/nn/nn_create.c \
	$(DIR)/nn/nn_destroy.c \
	$(DIR)/nn/nn_forward_pass.c \
	$(DIR)/activations.c \
	$(DIR)/layers.c \
	$(DIR)/nn/nn_mem.c


all: $(EXEF)

nn_create.out: nn_create.c $(NN_CREATE_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^

nn_forward_pass.out: nn_forward_pass.c $(NN_FORWARD_PASS_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lopenblas

nn_batch_forward_pass.out: nn_batch_forward_pass.c $(NN_FORWARD_PASS_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ -lm -lopenblas


clean:
	find . -type f -executable -delete

.PHONY: clean all
