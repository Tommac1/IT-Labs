#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>

int main(int argc, char *argv[])
{
    int numCPU = sysconf(_SC_NPROCESSORS_ONLN);
    printf("numCPU %d\n", numCPU);

    numCPU = sysconf(_SC_NPROCESSORS_CONF);
    printf("numCPU %d\n", numCPU);

    return 0;
}

