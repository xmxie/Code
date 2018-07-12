#include"FaceID.h"

Mat* CalIntegralDiagrams(Mat* samples){
	Mat* intergralMat = new Mat[SAMPLE_NUM];
	for (int num = 0; num < SAMPLE_NUM; num++){
		intergralMat[num] = samples[num].clone();
		intergralMat[num].at<uchar>(0, 0) = samples[num].at<uchar>(0, 0);
		for (int j = 1; j < MAP_ROWS; j++)//求出第一列的值
			intergralMat[num].at<uchar>(j, 0) = intergralMat[num].at<uchar>(j-1, 0)
			+ samples[num].at<uchar>(j, 0);
		for (int i = 1; i < MAP_ROWS; i++)//求出第一行的值
			intergralMat[num].at<uchar>(0, i) = intergralMat[num].at<uchar>(0, i-1) 
			+ samples[num].at<uchar>(0, i);
		for (int j = 1; j < MAP_ROWS; j++) {
			for (int i = 1; i < MAP_COLS; i++) {
				intergralMat[num].at<uchar>(j, i) = intergralMat[num].at<uchar>(j, i - 1)
					+ intergralMat[num].at<uchar>(j - 1, i) + samples[num].at<uchar>(j, i)
					- intergralMat[num].at<uchar>(j - 1, i - 1);
			}
		}
	}
	return intergralMat;
}