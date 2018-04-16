TARGETS=main
CC=gcc
CFLAGS=-Wall -g -pedantic


all: $(TARGETS)

%: %.o
	$(CC) $(CFLAGS) -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rfv *.o
	rm -rfv $(TARGETS)
