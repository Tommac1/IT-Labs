CC=gcc
CFLAGS=-Wall -g 

TARGETS=main

all: $(TARGETS)

%: %.o
	gcc -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<
