
all: main libwho.so libwho.o

main: libwho.so main.c
	gcc main.c -o $@ -ldl

libwho.so: libwho.o
	gcc -shared -fPIC libwho.o -o $@ -nostartfiles

libwho.o: libwho.c
	gcc -Wall -c -fPIC libwho.c
 
clean:
	rm -f *.o *.so 
