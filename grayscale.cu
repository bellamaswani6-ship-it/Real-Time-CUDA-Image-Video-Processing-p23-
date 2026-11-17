#include <cuda_runtime.h>
#include <iostream>
#include "grayscale.h"

using namespace std;

unsigned char *d_input = nullptr;
unsigned char *d_output = nullptr;

int imgWidth = 0;
int imgHeight = 0;

__global__ void grayscaleKernel()
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < width && y < height)
    {
        int index = (y * width + x) * 3;

        unsigned char b = input[index];
        unsigned char g = input[index + 1];
        unsigned char r = input[index + 2];

        unsigned char gray =0.299*r+0.587*g+0.114*b

        output[y * width + x] = gray;
    }
}
bool cudaInit(int width, int height)
{
    imgWidth = width;
    imgHeight = height;

    size_t inputSize = width * height * 3;
    size_t outputSize = width * height;

    cudaError_t err;

    err = cudaMalloc(&d_input, inputSize);

    if(err != cudaSuccess)
    {
        cerr << "Input allocation failed: "
             << cudaGetErrorString(err) << endl;
        return false;
    }

    err = cudaMalloc(&d_output, outputSize);

    if(err != cudaSuccess)
    {
        cerr << "Output allocation failed: "<< cudaGetErrorString(err) << endl;
        cudaFree(d_input);
        d_input = nullptr;
        return false;
    }

    cout << "CUDA memory allocated successfully!" << endl;

    return true;
}
bool cudaProcess(const unsigned char* input,unsigned char* output,int width,int height)
{
    if(!input || !output ||
       width != imgWidth || height != imgHeight)
        return false;
    size_t inputSize = width * height * 3;
    size_t outputSize = width * height;
    cudaError_t err;
    err = cudaMemcpy(d_input,input,inputSize,cudaMemcpyHostToDevice);
    if(err != cudaSuccess)
        return false;
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x,(height + block.y - 1) / block.y);
    grayscaleKernel<<<grid, block>>>();

    err = cudaGetLastError();

    if(err != cudaSuccess)
    {
        cerr << "Kernel launch failed: "
             << cudaGetErrorString(err) << endl;
        return false;
    }
    err = cudaMemcpy(output,d_output,outputSize,cudaMemcpyDeviceToHost);

    if(err != cudaSuccess)
    {
        cerr << "Output copy failed: "<< cudaGetErrorString(err) << endl;
        return false;
    }

    return true;
}

void cudaCleanup()
{
    if(d_input)
        cudaFree(d_input);

    if(d_output)
        cudaFree(d_output);

    d_input = nullptr;
    d_output = nullptr;

    cout << "CUDA memory released!" << endl;
}
