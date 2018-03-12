CC=gcc
CFLAGS=-g -Wall -pedantic

TARGETS=main

all : $(TARGETS)

% : %.o 
	${CC} -o % $<

%.o : %.c
	$(CC) $(CFLAGS) -c $<

