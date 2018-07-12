#include"FaceID.h"

Mat* CalIntegralDiagrams(Mat* samples){
	Mat* intergralMat = new Mat[SAMPLE_NUM];
	for (int num = 0; num < SAMPLE_NUM; num++){
		intergralMat[num] = samples[num].clone();
		for (int j = 0; j < MAP_ROWS; j++) {
			for (int i = 0; i < MAP_COLS; i++) {

			}
		}
	}
	return intergralMat;
}