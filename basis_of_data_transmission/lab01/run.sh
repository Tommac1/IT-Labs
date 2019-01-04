#!/bin/bash

# compile, run, make plots and display
gcc -Wall -g -o lab01 lab01.c -lm && ./lab01 && ./make_plot.sh && feh plot*.png

