#include <cuda_runtime.h>
#include <stdio.h>
#include "../include/myhelp.h"


void sumArrays(float* a,float* b,float* res,int offset,const int size){
    for(int i=0,k=offset;k<size;i++,k++){
        res[i]=a[k]+b[k];
    }
}


__global__ void sumArraysGPU(float* a,float* b,float* res,int offset,int n){
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    int k= i+offset;
    if(k+3*blockDim.x<n){
        res[i]=a[k]+b[k];
        res[i+blockDim.x]=a[k+blockDim.x]+b[k+blockDim.x];
        res[i+blockDim.x*2]=a[k+blockDim.x*2]+b[k+blockDim.x*2];
        res[i+blockDim.x*3]=a[k+blockDim.x*3]+b[k+blockDim.x*3];
    }
}


int main(int argc,char** argv){
    int dev=0;
    cudaSetDevice(dev);
    int block_x=512;
    int nElem=1<<18;
    int offset=0;
    if(argc==2)
        offset=atoi(argv[1]);
    else if(argc==3){
        offset=atoi(argv[1]);
        block_x=atoi(argv[2]);
    }

    int nBytes=sizeof(float)*nElem;
    float* a_h=(float*)malloc(nBytes);
    float* b_h=(float*)malloc(nBytes);
    float* res_h=(float*)malloc(nBytes);
    float* res_from_gpu_h=(float*)malloc(nBytes);
    memset(res_h,0,nBytes);
    memset(res_from_gpu_h,0,nBytes);

    float* a_d,* b_d,* res_d;
    CHECK(cudaMalloc((float**)&a_d,nBytes));
    CHECK(cudaMalloc((float**)&b_d,nBytes));
    CHECK(cudaMalloc((float**)&res_d,nBytes));
    initialData(a_h,nElem);
    initialData(b_h,nElem);

    CHECK(cudaMemcpy(a_d,a_h,nBytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(b_d,b_h,nBytes,cudaMemcpyHostToDevice));

    dim3 block(block_x);
    dim3 grid(nElem/block_x);
    double iStart,iElaps;
    iStart=cpuSecond();
    sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d,offset,nElem);
    cudaDeviceSynchronize();
    iElaps=cpuSecond()-iStart;
    CHECK(cudaMemcpy(res_from_gpu_h,res_d,nBytes,cudaMemcpyDeviceToHost));
    printf("Execution config<<<%d,%d>>> time elapsed %f sec --offset:%d",grid.x,block.x,iElaps,offset);

    sumArrays(a_h,b_h,res_h,offset,nElem);
    checkResult(res_h,res_from_gpu_h,nElem-4*block_x);
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(res_d);

    free(a_h);
    free(b_h);
    free(res_h);
    free(res_from_gpu_h);
    
    return 0;
}