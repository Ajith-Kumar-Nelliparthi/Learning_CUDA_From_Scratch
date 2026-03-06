#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void textureMemoryKernel(cudaTextureObject_t textObj, float* output, int width, int height) {
    int x = blockIdx.y * blockDim.y + threadIdx.y;
    int y = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < width && y < height) {
        float u = (float)x / (float)width;
        float v = (float)y / (float)height;

        float val = tex2D<float>(textObj, u, v);
        output[y * width + x] = val;
    }
}

int main() {
    const int width = 8;
    const int height = 8;
    size_t size = width * height * sizeof(float);

    float h_data[width * height];
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            h_data[y * width + x] = (float)(x + y);
        }
    }

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* d_array;
    cudaMallocArray(&d_array, &channelDesc, width, height);

    cudaMemcpyToArray(d_array, 0, 0, h_data, size, cudaMemcpyHostToDevice);

    // Resource descriptor
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_array;

    // Texture descriptor
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;   // Enable interpolation
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;               

    // Create texture object
    cudaTextureObject_t texObj = 0;
    cudaCreateTextureObject(&texObj, &resDesc, &texDesc, nullptr);

    // Allocate output buffer
    float* d_output;
    cudaMalloc(&d_output, size);

    // Launch kernel
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,
              (height + block.y - 1) / block.y);
    textureMemoryKernel<<<grid, block>>>(texObj, d_output, width, height);

    // Copy back results
    float h_output[width * height];
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Interpolated output:\n";
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            std::cout << h_output[y * width + x] << " ";
        }
        std::cout << "\n";
    }

    // Cleanup
    cudaDestroyTextureObject(texObj);
    cudaFreeArray(d_array);
    cudaFree(d_output);

    return 0;
}