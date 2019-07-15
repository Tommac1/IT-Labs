CC=gcc
CFLAGS=-Wall -g
LIBS=-lm

PSNR_SRC=src/psnr.c src/libbmp.c
PSNR_TARGET=psnr

ENTR_SRC=src/entrophy.c
ENTR_TARGET=entrophy

all:
	$(CC) $(CFLAGS)	-o $(PSNR_TARGET) $(PSNR_SRC) $(LIBS)
	$(CC) $(CFLAGS)	-o $(ENTR_TARGET) $(ENTR_SRC) $(LIBS)

clean:
	rm $(TARGET) $(ENTR_TARGET)
