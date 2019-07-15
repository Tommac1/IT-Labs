CUDA_CC=nvcc
CC=gcc

CUDA_FLAGS=-arch=sm_20 -Wno-deprecated-gpu-targets -ccbin=clang-3.8
C_FLAGS=-Wall -Wextra -I/usr/include -lcuda -lcudart -g -lm

EX1_TARGET=ex1
EX1_CUBIN_TARGET=ex1_kernel.cubin
EX1_CUBIN_SRC=ex1_kernel.cu
EX1_SRC=ex1.c
EX1_HEDAER=ex1.h

EX2_TARGET=ex2
EX2_CUBIN_TARGET=ex2_kernel.cubin
EX2_CUBIN_SRC=ex2_kernel.cu
EX2_SRC=ex2.c
EX2_HEDAER=ex2.h

all: $(EX1_TARGET) $(EX2_TARGET)


$(EX1_TARGET): $(EX1_SRC) $(EX1_CUBIN_TARGET) $(EX1_HEADER)
	$(CC) $(C_FLAGS) -o $@ $<

$(EX1_CUBIN_TARGET): $(EX1_CUBIN_SRC)
	$(CUDA_CC) $(CUDA_FLAGS) --cubin $<

$(EX2_TARGET): $(EX2_SRC) $(EX2_CUBIN_TARGET) $(EX1_HEADER)
	$(CC) $(C_FLAGS) -o $@ $<

$(EX2_CUBIN_TARGET): $(EX2_CUBIN_SRC)
	$(CUDA_CC) $(CUDA_FLAGS) --cubin $<


.PHONY: clean
clean:
	rm -f $(EX1_TARGET) $(EX2_TARGET) $(EX1_CUBIN_TARGET) $(EX2_CUBIN_TARGET)
