#!/bin/bash

nvcc -ccbin clang-3.8 ex2.cu libbmp.c -o ex2 -Wno-deprecated-gpu-targets

./ex2

eog illidan* &>/dev/null
