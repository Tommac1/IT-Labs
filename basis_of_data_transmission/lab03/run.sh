#!/bin/bash


# compile, run, make plots and display
gcc -Wall -g -o lab03 lab03.c -L/usr/local/lib -lm && ./lab03 && ./make_plot.sh && feh plot*.png

