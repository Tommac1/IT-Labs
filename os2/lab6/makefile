CC=gcc
CFLAGS=-g -Wall -pedantic -fsanitize=address
TARGETS=main hack

all: $(TARGETS)

%: %.c
	$(CC) $(CFLAGS) -o $@ $< -lcrypt -pthread

clean:
	rm -rfv $(TARGETS)
	rm -rfv *.o
