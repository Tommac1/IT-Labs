#!/bin/bash

KISS_SRC="kissfft/kiss_fft.c kissfft/kiss_fftr.c"

# compile, run, make plots and display
gcc -Wall -g -o lab02 lab02.c $KISS_SRC -L/usr/local/lib -lm && ./lab02 && ./make_plot.sh && feh plot*.png

