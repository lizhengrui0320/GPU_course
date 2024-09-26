// hello_world.cu
#include <iostream>
using namespace std;
__global__ void helloFromGPU() {
    printf("Hello, World from GPU!\n");
}

int main() {
    // Launch kernel
    helloFromGPU<<<2, 2>>>();
    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    return 0;
}
