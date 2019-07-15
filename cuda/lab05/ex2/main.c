#include	<stdio.h>
#include    <string.h>
#include	<assert.h> 
#include    <math.h>
#include	<cuda.h>

#define N 50
#define BLK_SZ 128
#define CALL(x)	{int r=x;\
		if(r!=0){fprintf(stderr,"%s returned %d in line %d -- exiting.\n",#x,r,__LINE__);\
		exit(0);}} 

#define ALIGN_UP(offset, alignment)  (offset) = ((offset) + (alignment) - 1) & ~((alignment) - 1)

float HostVect[N];
float gpu_result[1];
float cpu_result[1];
float x_val[1] = { 1.1 };

void calc_cpu()
{
    float tmp = 0.0;

    for (int i = 0; i < N; i++) {
        tmp = pow(*x_val, i);
        cpu_result[0] += (tmp * HostVect[i]);
    }
}

int main(int argc, char *argv[]) 
{
	int i;
	float x;
	int blocks = N / BLK_SZ;
    if(N % BLK_SZ) blocks++;

    for (i = 0; i < N; i++) {
        HostVect[i] = (i + 1.0) / 10.0;
    }

    calc_cpu();

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
    CUdeviceptr dXval;

	CALL( cuMemAlloc(&DevVect, sizeof(HostVect)) );
    CALL( cuMemAlloc(&dXval, sizeof(x_val)) );

	CALL( cuMemcpyHtoD(DevVect, HostVect, sizeof(HostVect)) );
    CALL( cuMemcpyHtoD(dXval, x_val, sizeof(x_val)) );

	CALL( cuFuncSetBlockShape(hFunction, BLK_SZ, 1, 1) );

	int  offset = 0;
	void *ptr;

	ptr = (void*)(size_t) (N - 1);
	ALIGN_UP(offset, __alignof(ptr));
	CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
	offset += sizeof(ptr);

	ptr = (void*)(size_t) dXval;
	ALIGN_UP(offset, __alignof(ptr));
	CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
	offset += sizeof(ptr);

	ptr = (void*)(size_t) DevVect;
	ALIGN_UP(offset, __alignof(ptr));
	CALL( cuParamSetv(hFunction, offset, &ptr, sizeof(ptr)) );
	offset += sizeof(ptr);

	CALL( cuParamSetSize(hFunction, offset) );

	CALL( cuLaunchGrid(hFunction, blocks, 1) );

	CALL( cuMemcpyDtoH((void *) HostVect, DevVect, sizeof(HostVect)) );
    CALL( cuMemcpyDtoH((void *) x_val, dXval, sizeof(x_val)) );

	CALL( cuMemFree(DevVect) );
    CALL( cuMemFree(dXval) );

    gpu_result[0] = x_val[0];

    printf("cpu: %.5f\ngpu: %.5f\n", cpu_result[0], gpu_result[0]);
	
	puts("done");
	return 0;
}

