#!/bin/bash

TARGETS="main.cpp Database.cpp Administrator.cpp User.cpp"
CPP_FLAGS="-Wall -std=c++17 -pedantic -g -pthread"
JSON_INCLUDE_PATH="/home/anon/Repositories/json/include"
NLOHMANN_PATH="/home/anon/Repositories/json/include/nlohmann"

# compile and run
g++ -o main ${TARGETS} \
    -I${NLOHMANN_PATH} \
    -I${JSON_INCLUDE_PATH} \
    ${CPP_FLAGS} \
    && ./main
