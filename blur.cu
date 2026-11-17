#include<iostream>
#include<cuda_runtime.h>
#include "blur.h"
using namespace std;
int img_width,img_height,input_size,output_size;
unsigned char *d_input,*d_output;
__global__ void box_blur()
{
	int b_sum,g_sum,r_sum,count;
	int x=blockIdx.x*blockDim.x+threadIdx.x;
	int y=blockIdx.y*blockDim.y+threadIdx.y;
	if(x<width && y<height)
	{
		b_sum=0;
		g_sum=0;
		r_sum=0;
		count=0;
		int dx,dy,nx,ny;
		for(dy=-3;dy<=3;dy++)
		{
			for(dx=-3;dx<=3;dx++)
			{
				nx=x+dx;
				ny=y+dy;
				if((nx>=0 && nx< width)&&(ny>=0 && ny<height))
				{
					int pixel_index=ny*width+nx;
					int index=pixel_index*3;
					unsigned char b=input[index];
					unsigned char g=input[index+1];
					unsigned char r=input[index+2];
					b_sum+=b;
					g_sum+=g;
					r_sum+=r;
					count++;
				}
			}
		}
		int output_index=(y*width+x)*3;
		output[output_index]=b_sum/count;
		output[output_index+1]=g_sum/count;
		output[output_index+2]=r_sum/count;
	}
}
bool cudainit(int width,int height)
{
	img_width=width;
	img_height=height;
	input_size=width*height*3;
	output_size=width*height*3;
	cudaError_t error;
	error=cudaMalloc(&d_input,input_size);
	if(error!=cudaSuccess)
	{
		cerr<<"Memory for input  is not allocated successfully in GPU!!!"<<endl;
		return false;
	}
	error=cudaMalloc(&d_output,output_size);
	if(error!=cudaSuccess)
	{
		cerr<<"Memory for output is not allocated successfully in GPU!!!"<<endl;
		return false;
	}
	return true;
}
bool cudaprocess(unsigned char*input,unsigned char* output,int width,int height)
{
	if(!input||!output||width!=img_width||height!=img_height)
		return false;
	cudaError_t error;
	error=cudaMemcpy(d_input,input,input_size,cudaMemcpyHostToDevice);
	if(error!=cudaSuccess)
	{
		cerr<<"Error in copying the input from host to device "<<endl;
		return false;
	}
	dim3 block(16,16);
	dim3 grid((width+block.x-1)/block.x,(height+block.y-1)/block.y);
	box_blur<<<grid,block>>>();
	error=cudaGetLastError();
	if(error!=cudaSuccess)
	{
		cerr<<"Kernel launch failed: " << cudaGetErrorString(error) << endl;
		return false;
	}
	cudaDeviceSynchronize();
	error=cudaMemcpy(output,d_output,output_size,cudaMemcpyDeviceToHost);
	if(error!=cudaSuccess)
	{
		cerr<<"Error in copying from device to host "<<endl;
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
	cout<<"Cuda Memory released successfully "<<endl;
}
