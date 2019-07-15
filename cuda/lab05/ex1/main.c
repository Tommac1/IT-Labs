#include	<stdio.h>
#include	<assert.h> 
#include    <math.h>
#include	<cuda.h>

#define N 50
#define BLK_SZ 128
#define CALL(x)	{int r=x;\
		if(r!=0){fprintf(stderr,"%s returned %d in line %d -- exiting.\n",#x,r,__LINE__);\
		exit(0);}} 

#define ALIGN_UP(offset, alignment)  (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

float    HostVect[N];

float fun(float x, int i) {
	return pow(x, (double) i);
}

int main(int argc, char *argv[]) 
{
    float tab[N] = { 0 };
	int i;
	float x;
	int blocks = N / BLK_SZ;
    if(N % BLK_SZ) blocks++;

    for (i = 0; i < N; i++) {
        HostVect[i] = (i + 1.0) / 10.0;
        tab[i] = (i + 1.0) / 10.0;
    }

	CUdevice	hDevice;
	CUcontext	hContext;
	CUmodule	hModule;
	CUfunction	hFunction;

	CALL( cuInit(0) );
	CALL( cuDeviceGet(&hDevice, 0) ); 	
	CALL( cuCtxCreate(&hContext, 0, hDevice) );
	CALL( cuModuleLoad(&hModule, "kernel.cubin") );
	CALL( cuModuleGetFunction(&hFunction, hModule, "KERN") );

	CUdeviceptr DevVect;

	CALL( cuMemAlloc(&DevVect, sizeof(HostVect)) );

	CALL( cuMemcpyHtoD(DevVect, HostVect, sizeof(HostVect)) );

	CALL( cuFuncSetBlockShape(hFunction, BLK_SZ, 1, 1) );

	int  offset = 0;
	void *ptr;

	ptr = (void*)(size_t)N;
	ALIGN_UP(offset, __alignof(ptr));
	CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
	offset += sizeof(ptr);

	ptr = (void*)(size_t)DevVect;
	ALIGN_UP(offset, __alignof(ptr));
	CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
	offset += sizeof(ptr);

	CALL( cuParamSetSize(hFunction, offset) );

	CALL( cuLaunchGrid(hFunction, blocks, 1) );

	CALL( cuMemcpyDtoH((void *) HostVect, DevVect, sizeof(HostVect)) );

	CALL( cuMemFree(DevVect) );

	for (i = 0; i < N; i++)
        printf("arg: %f  gpu: %f  cpu: %f\n", (float)i, HostVect[i], fun(tab[i], i));
	
	puts("done");
	return 0;
}

