#include <cmath>
#include <cstddef>
#include <fstream>
#include <stdexcept>

#include "helper.h"

__global__
void vecAddKernel(float *input1, float *input2, float *output,
                             std::size_t n) {
    std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = input1[i] + input2[i];
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " input1.txt input2.txt expected.txt\n";
        return 1;
    }

    try {
        //@@ Read inputs and expected into host
        const std::vector<float> hostInput1 = readVectorFromFile(argv[1]);
        const std::vector<float> hostInput2 = readVectorFromFile(argv[2]);
        const std::vector<float> expectedOutput = readVectorFromFile(argv[3]);

        const std::size_t inputLength = hostInput1.size();
        if (inputLength == 0) {
            throw std::runtime_error("Input vectors must not be empty.");
        }
        if (hostInput2.size() != inputLength ||  expectedOutput.size() != inputLength) {
            throw std::runtime_error("All three files must contain the same number of values.");
        }

        const std::size_t totalSize = inputLength * sizeof(float);
        std::vector<float> hostOutput(inputLength);

        float *deviceInput1;
        float *deviceInput2;
        float *deviceOutput;

        //@@ Allocate GPU memory here
        cudaMalloc(static_cast<void **>(&deviceInput1), totalSize);
        cudaMalloc(static_cast<void **>(&deviceInput2), totalSize);
        cudaMalloc(static_cast<void **>(&deviceOutput), totalSize);
        
        //@@ Copy memory to the GPU here
        cudaMemcpy(deviceInput1, hostInput1.data(), totalSize, cudaMemcpyHostToDevice);
        cudaMemcpy(deviceInput2, hostInput2.data(), totalSize, cudaMemcpyHostToDevice);

        //@@ Initialize the grid and block dimensions here
        const std::size_t threadsPerBlock = 256;
        const std::size_t blocks = static_cast<std::size_t>(std::ceil(1.0 * inputLength / threadsPerBlock));
        dim3 gridDim(blocks, 1, 1);
        dim3 blockDim(threadsPerBlock, 1, 1);

        //@@ Launch the GPU Kernel here
        vecAddKernel<<<gridDim, blockDim>>>(deviceInput1, deviceInput2, deviceOutput, inputLength);
        
        cudaDeviceSynchronize();

        //@@ Copy the GPU memory back to the CPU here
        cudaMemcpy(hostOutput.data(), deviceOutput, totalSize, cudaMemcpyDeviceToHost);

        //@@ Free the GPU memory here
        cudaFree(deviceInput1);
        cudaFree(deviceInput2);
        cudaFree(deviceOutput);

        return compare(hostOutput, expectedOutput);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
