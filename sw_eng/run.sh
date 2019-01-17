#!/bin/bash

if [[ $1 == "-clean" ]]; then
    make clean
fi

# compile and run
make && ./main
