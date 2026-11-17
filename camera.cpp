#include<iostream>
#include<opencv2/opencv.hpp>
#include "invert.h"
using namespace std;
using namespace cv;
int main()
{
	VideoCapture cap(0,CAP_V4L2);
	Mat frame,invert;
	if(!cap.isopened())
	{
		cerr<<"camera is failed"<<endl;
		return -1;
	}
	bool state=false;
	cout<<"camera is opened successfully!!!"<<endl;
	while(true)
	{
		if(!cap.read(frame))
                {
		cerr<<"frames are not reading"<<endl;
		break;
		}
		if(!frame.iscontinous())
			frame=frame.clone();

                   if(!state)
	           {
		      blur.create(frame.rows,frame.cols,CV_8UC3);
                       if(!cudainit(frame.cols,frame.rows)
                         {
			 cerr<<"GPU Memory Allocation is Failed!!!"<<endl;
			 return -1;
			 }
	           state=true;
		   }
		   if(!cudaprocess(frame.data,invert.data,frame.cols,frame.rows))
		   {
		   cerr<<"cuda launch failed"<<endl;
		   cudaclean();
		   }
		   imshow("orginal image",frame);
		   imshow("inversion",invert);
		   int key=pollKey();
		   if(key==27)
		   break;
	}
	if(state)
	cudaclean();
	cap.release();
	destroyAllWindows();
	return 0;
	}
	
