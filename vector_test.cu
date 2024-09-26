#include <stdio.h>  
#include <cuda_runtime.h>  
  
// 假设我们有一个简单的CUDA核函数  
__global__ void simpleKernel(int *d_array, int arraySize) {  
    int index = threadIdx.x + blockIdx.x * blockDim.x;  
    if (index < arraySize) {  
        d_array[index] = index * index; // 计算平方  
    }  
}  
  
int main() {  
    const int arraySize = 10;  
    int h_array[arraySize] = {0,1,2,3,4,5,6,7,8,9}; // 主机上的数组  
    int *d_array; // 设备上的数组  
  
    // 分配设备内存  
    cudaMalloc((void**)&d_array, arraySize * sizeof(int));  
  
    // 设置核函数参数  
    int blocks = (arraySize + 255) / 256; // 计算所需的块数  
    int threads = min(256, arraySize); // 计算每块的线程数  
    simpleKernel<<<blocks, threads>>>(d_array, arraySize);  
  
    // 等待GPU上的操作完成  
    cudaDeviceSynchronize();  
  
    // 将结果从设备复制回主机  
    cudaMemcpy(h_array, d_array, arraySize * sizeof(int), cudaMemcpyDeviceToHost);  
  
    // 再次同步，确保数据已复制到主机  
    cudaDeviceSynchronize();  
  
    // 现在，我们可以安全地访问h_array中的数据  
    for (int i = 0; i < arraySize; i++) {  
        printf("%d ", h_array[i]);  
    }  
    printf("\n");  
  
    // 释放设备内存  
    cudaFree(d_array);  
  
    return 0;  
}
