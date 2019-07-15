#!/bin/bash

if [[ -n $1 ]]; then
    nvcc -ccbin clang-3.8 "$1.cu" -o "$1" -Wno-deprecated-gpu-targets --run \
            -lm -ftz=true -prec-div=false -prec-sqrt=false
else
    echo "usage: $0 ex"
fi

# ./ex2
