#!/bin/bash


gcc -I /usr/include/cuda -lcuda main.c -o main -lm
RET1=$?
nvcc -arch=sm_20 -cubin kernel.ptx -Wno-deprecated-gpu-targets
RET2=$?

if [[ 0 -eq $RET1 ]] && [[ 0 -eq $RET2 ]]; then
    ./main
fi
