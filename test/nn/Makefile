CC = gcc
CFLAGS = -g3 -Wall -Wextra -fsanitize=address \
	-fsanitize=undefined -fstack-protector
LIBFLAGS = -lm -lopenblas

DIR = ../../src

EXEF = $(patsubst %.c, %.out, $(wildcard *.c))


NN_CREATE_SOURCES = \
	$(DIR)/nn/nn_create.c \
	$(DIR)/nn/nn_destroy.c \
	$(DIR)/nn/nn_weights.c \
	$(DIR)/error.c \
	$(DIR)/opt/opt_create.c \
	$(DIR)/opt/opt_apply.c
NN_FORWARD_PASS_SOURCES = $(NN_CREATE_SOURCES) \
	$(DIR)/nn/nn_forward_pass.c \
	$(DIR)/nn/nn_mem.c \
	$(wildcard $(DIR)/layers/*.c $(DIR)/layers/**/*.c)
NN_BACKWARD_PASS_SOURCES = $(NN_FORWARD_PASS_SOURCES) \
	$(DIR)/nn/nn_backward_pass.c \
	$(DIR)/nn/nn_regularization.c \
	$(DIR)/loss.c
NN_TRAIN_SOURCES = $(NN_BACKWARD_PASS_SOURCES) \
	$(DIR)/nn/nn_train.c \
	$(DIR)/opt/opt_mem.c


all: $(EXEF)

nn_create.out: nn_create.c $(NN_CREATE_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

nn_forward_pass.out: nn_forward_pass.c $(NN_FORWARD_PASS_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

nn_batch_forward_pass.out: nn_batch_forward_pass.c $(NN_FORWARD_PASS_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

nn_batch_backward_pass.out: nn_batch_backward_pass.c $(NN_BACKWARD_PASS_SOURCES)
	$(CC) $(CFLAGS) -o $@ $^ $(LIBFLAGS)

nn_train.out: nn_train.c $(NN_TRAIN_SOURCES)
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
