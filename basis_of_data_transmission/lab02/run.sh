#!/bin/bash

KISS_SRC="kissfft/kiss_fft.c kissfft/kiss_fftr.c"

# compile, run, make plots and display
gcc -Wall -g -o 02 02.c $KISS_SRC -L/usr/local/lib -lfftw3 -lm
&& ./02
&& ./make_plot.sh
&& feh plot*.png

