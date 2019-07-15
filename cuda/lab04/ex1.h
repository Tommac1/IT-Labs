#ifndef EX1_H
#define EX1_H

#define DIM (2048)
#define N 16
#define BLOCK_SIZE 128

#define CUDA_CALL(X)                                            \
        do {                                                    \
            cudaError_t cuda_error__ = (X);                     \
            if (cuda_error__) {                                 \
                printf("CUDA error: " #X                        \
                       " returned \"%s\" (%d) (line: %d)\n",    \
                        cudaGetErrorString(cuda_error__),       \
                        cuda_error__,  __LINE__);               \
            }                                                   \
        } while (0)                                             \


#endif
