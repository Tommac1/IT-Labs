#!/bin/bash

# compile
gcc -Wall -g -o 01 01.c -lm

# run
if [ $? == 0 ]
then
    ./01
fi

#make plots
if [ $? == 0 ]
then
    ./make_plot.sh
fi

# show plots
if [ $? == 0 ]
then
    feh plot*.png
fi
