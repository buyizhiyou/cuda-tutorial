#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/myhelp.h"

__global__ void warmup(float* c){
    //warmup部分是提前启动一次GPU，因为第一次启动GPU时会比第二次速度慢一些
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    if((tid/warpSize)%2==0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    c[tid]=a+b;
}

__global__ void mathKernel1(float* c){
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    float a=0.0;
    float b=0.0;
    if(tid%2==0){
        a = 100.0f;
    }
    else{
        b = 200.0f;
    }
    c[tid]=a+b;
}

__global__ void mathKernel2(float* c){
    int tid = blockIdx.x*blockDim.x+threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    if((tid/warpSize)%2==0){
        a=100.0f;
    }
    else{
        b = 200.0f;
    }
    c[tid]=a+b;
}

__global__ void mathKernel3(float* c){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    float a = 0.0;
    float b = 0.0;
    bool ipred = (tid%2==0);
    if(ipred){
        a=100.0f;
    }
    else{
        b=200.0f;
    }
    c[tid]=a+b;
}

int main(int argc,char ** argv){
    int dev=0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("%s using device %d:%s\n",argv[0],dev,deviceProp.name);

    int size = 2<<5;
    int blockSize = 2<<5;
    if(argc>1) blockSize=atoi(argv[1]);
    if(argc>2) size=atoi(argv[2]);
    printf("Data size:%d\n",size);

    //set up execution configuration
    dim3 block(blockSize,1);
    dim3 grid((size-1)/block.x+1,1);

    //allocate memory
    float * C_dev;
    size_t nBytes = size*sizeof(float);
    float* C_host=(float*)malloc(nBytes);
    cudaMalloc((float**)&C_dev,nBytes);

    //run a warmup kernel to remove overhead
    double start,elapes;
    cudaDeviceSynchronize();
    start = cpuSecond();
    warmup<<<grid,block>>>(C_dev);
    cudaDeviceSynchronize();
    elapes=cpuSecond()-start;
    printf("warmup <<<%d,%d>>> elapsed %lf sec\n",grid.x,block.x,elapes);

    //run kernel
    start=cpuSecond();
    mathKernel1<<<grid,block>>>(C_dev);
    cudaDeviceSynchronize();
    elapes=cpuSecond()-start;
    printf("mathKernel1<<<%d,%4d>>>elapsed %lf sec\n",grid.x,block.x,elapes);
    cudaMemcpy(C_host,C_dev,nBytes,cudaMemcpyDeviceToDevice);

    start=cpuSecond();
    mathKernel2<<<grid,block>>>(C_dev);
    cudaDeviceSynchronize();
    elapes=cpuSecond()-start;
    printf("mathKernel2<<<%d,%4d>>>elapsed %lf sec\n",grid.x,block.x,elapes);

    start=cpuSecond();
    mathKernel3<<<grid,block>>>(C_dev);
    cudaDeviceSynchronize();
    elapes=cpuSecond()-start;
    printf("mathKernel<<<%d,%4d>>>elapsed %lf sec\n",grid.x,block.x,elapes);

    cudaFree(C_dev);
    free(C_host);
    cudaDeviceReset();
    return EXIT_SUCCESS;
}