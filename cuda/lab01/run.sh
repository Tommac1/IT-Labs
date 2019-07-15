#!/bin/bash

N_VALS="1 4 16 256 1024 2048"
EPS_EXP="3 6 9 11 12 13"

echo "EXCERCISE #1 ========================="

for N in $N_VALS; do
    nvcc -ccbin clang-3.8 ex1.cu -o ex1 -DN=$N -Wno-deprecated-gpu-targets
#    ./ex1
    nvprof ./ex1
done

echo "EXCERCISE #2 ========================="

nvcc -ccbin clang-3.8 ex2.cu -o ex2 -Wno-deprecated-gpu-targets
for EPS in $EPS_EXP; do
#    ./ex2 $EPS
    nvprof ./ex2 $EPS
done

exit 0
