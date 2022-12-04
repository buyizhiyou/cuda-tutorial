#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/myhelp.h"

void sumArrays(float* a,float* b,float* res,const int size){
    for(int i=0;i<size;i++){
        res[i]=a[i]+b[i];
    }
}

__global__ void sumArraysGPU(float* a,float* b,float* res){
    int i=threadIdx.x;
    res[i]=a[i]+b[i];
}


int main(int argc,char** argv){
    // int dev=0;
    // cudaSetDevice(dev);
    initDevice(0);

    int n=32;
    printf("Vector size:%d\n",n);
    int nByte=sizeof(float)*n;
    float* a_h=(float*)malloc(nByte);
    float* b_h=(float*)malloc(nByte);
    float* res_h=(float*)malloc(nByte);
    float* res_from_gpu_h=(float*)malloc(nByte);
    memset(res_h,0,nByte);
    memset(res_from_gpu_h,0,nByte);



    float* a_d,* b_d,* res_d;
    CHECK(cudaMalloc((float**)&a_d,nByte));
    CHECK(cudaMalloc((float**)&b_d,nByte));
    CHECK(cudaMalloc((float**)&res_d,nByte));

    initialData(a_h,n);
    initialData(b_h,n);

    
    CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

    dim3 block(n);
    dim3 grid(n/block.x);
    sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d);
    printf("Execution configuration<<<%d,%d>>>",block.x,grid.x);

    CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
    sumArrays(a_h,b_h,res_h,n);

    checkResult(res_h,res_from_gpu_h,n);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);

    return  0;

}



