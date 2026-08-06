#include <cstddef>
#include <cmath>
#include <wb.h>

//@@ Insert code to implement vector addition here
__global__
void vecAddKernel(float *input1, float *input2, float *output, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input1[i] + input2[i];
    }
}

int main(int argc, char **argv) {
    wbArg_t args;
    int inputLength;
    float *hostInput1;
    float *hostInput2;
    float *hostOutput;
  
    args = wbArg_read(argc, argv);
  
    hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
    hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
    hostOutput = (float *)malloc(inputLength * sizeof(float));

    //@@ Allocate GPU memory here
    float *deviceInput1;
    float *deviceInput2;
    float *deviceOutput;
    size_t n = inputLength * sizeof(float);
    cudaMalloc(static_cast<void **>(&deviceInput1), n);
    cudaMalloc(static_cast<void **>(&deviceInput2), n);
    cudaMalloc(static_cast<void **>(&deviceOutput), n);

    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput1, hostInput1, n, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceInput2, hostInput2, n, cudaMemcpyHostToDevice);

    //@@ Initialize the grid and block dimensions here
    const size_t numThreadsPerBlock = 256;
    const size_t numBlocks = std::ceil(1.0 * n / numThreadsPerBlock);
    dim3 DimGrid(numBlocks, 1, 1);
    dim3 DimBlock(numThreadsPerBlock, 1, 1);

    //@@ Launch the GPU Kernel here
    vecAddKernel<<<DimGrid, DimBlock>>>(deviceInput1, deviceInput2, deviceOutput, n);
    cudaDeviceSynchronize();

    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, n, cudaMemcpyDeviceToHost);

    //@@ Free the GPU memory here
    cudaFree(deviceInput1);
    cudaFree(deviceInput2);
    cudaFree(deviceOutput);

    free(hostInput1);
    free(hostInput2);
    free(hostOutput);

    return 0;
}
