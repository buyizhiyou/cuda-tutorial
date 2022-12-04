#include <stdio.h>

__global__ void hello_world(void)
{ // The qualifier __global__ tells the compiler that the function will be called from the CPU and executed on the GPU
    printf("GPU:Hello world!\n");
}

int main(int argc, char **argv)
{
    printf("CPU:Hello world!\n");
    hello_world<<<1, 10>>>();
    cudaDeviceReset();
    return 0;
} 