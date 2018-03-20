CC=gcc
CFLAGS=-Wall -g -pedantic
TARGETS=main


all: $(TARGETS)

%: %.o
	$(CC) -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rf $(TARGETS)
	rm -rf *.o

