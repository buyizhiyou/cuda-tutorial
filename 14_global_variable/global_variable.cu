#include <cuda_runtime.h>
#include <stdio.h>

// __constant__ float cons;
__device__ float devData;
__global__ void checkGlobalVariable(){
    printf("blockIdx:%d,threadIdx:%d,Device:the value of the global variable is %f\n",blockIdx.x,threadIdx.x,devData);
    // __syncthreads();
    devData +=2.0;
}

int main(){
    float value=3.1f;
    cudaMemcpyToSymbol(devData,&value,sizeof(float));
    checkGlobalVariable<<<4,32>>>();
    cudaMemcpyFromSymbol(&value,devData,sizeof(float));
    printf("host:the value changed by the kernel to %f\n",value);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}
