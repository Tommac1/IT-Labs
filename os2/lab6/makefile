CC=gcc
CFLAGS=-g -Wall -pedantic
TARGETS=main hack

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -lcrypt
