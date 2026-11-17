#include<iostream>
#include<cuda_runtime.h>
#include "invert.h"
unsigned char* d_input=nullptr,*d_output=nullptr;
int img_width,img_height;
__global__ void invert()
{
  int x=blockIdx.x*blockDim.x+threadIdx.x;
 int y=blockIdx.y*blockDim.y+threadIdx.y;
 if(x<img_width && y<img_height)
 {
int index=(y*width+x)*3;
unsigned char b=input[index];
unsigned char g=input[index+1];
unsigned char r=input[index+2];
output[index]=255-b;
output[index+1]=255-g;
output[index+2]=255-r;
}
}
bool cudainit(int width,int height)
{
img_width=width;
img_height=height;
cudaError_t error;
int input_size=width*height*3;
int output_size=width*height*3;
error=cudaMalloc(&d_input,input_size);
if(error!=cudaSuccess)
{
cerr<<"Error in Allocating Memory for input in GPU"<<endl;
return false;
}
error=cudaMalloc(&d_output,output_size);
if(error!=cudaSuccess)
{
cerr<<"Error in Allocating Memory for output in GPU"<<endl;
return false;
}
return true;
}
bool cudaprocess(unsigned char* input,unsigned char* output,int width,int height)
{
if(!input||!output||width!=img_width||height!=img_height)
return false;
int input_size=width*height*3;
int output_size=width*height*3;
cudaError_t error;
error=cudaMemcpy(d_input,input,input_size,cudaMemcpyHostToDevice);
if(error!=cudaSuccess)
{
cerr<<"Error in copying from Host to Device "<<endl;
return false;
}
dim3 block(16,16);
dim3 grid((width+block.x -1)/block.x,(height+block.y -1)/block.y);
invert<<<grid,block>>>();
error=cudaGetLastError();
if(error!=cudaSuccess)
{
cerr<<"Kernel Launch Failed!!!"<<endl;
return false;
}
cudaDeviceSynchronize();
error=cudaMemcpy(output,d_output,output_size,cudaMemcpyDeviceToHost);
if(error!=cudaSuccess)
{
cerr<<"Copying From GPU to CPU is Failed!!!"<<endl;
return false;
}
return true;
}
void cudaclean()
{
if(d_input)
cudaFree(d_input);
if(d_output)
cudaFree(d_output);
d_input=nullptr;
d_output=nullptr;
}



