#pragma once
#include<opencv2\core\core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<iostream>
#include<fstream>
#include<array>
#include<vector>
#include <amp.h>
#include<amp_math.h>
#include<ctime>
using namespace std;
using namespace cv;
typedef float** Kernel;


/*
	------------卷积核对照表-------------
	所有卷积核文件的命名以"ij.txt"为准
	i代表第i层



*/
Mat generateConvKernel(int scale, Kernel def);//生成一个自定义的卷积核
Mat maxPooling(Mat ingredient, int grid, int step);//进行一次最大池化
void compareCharacter(Mat m1, Mat m2,int factor);



