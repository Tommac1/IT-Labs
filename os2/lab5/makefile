CC=gcc
CFLAGS=-Wall -g -pedantic
TARGETS=main


all: $(TARGETS)

%: %.o
	$(CC) -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -rfv $(TARGETS)
	rm -rf *.o

