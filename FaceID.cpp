#include"FaceID.h"

Mat generateConvKernel(int scale, Kernel def) {
	Mat kernel = Mat_<float>(scale, scale);
	for (int i = 0; i < scale; i++)
		for (int j = 0; j < scale; j++)
			kernel.ptr<float>(i)[j] = def[i][j];
	return kernel;
}
Mat maxPooling(Mat ingredient, int grid, int step) {
	Mat pooled = Mat((ingredient.rows - grid) / step + 1, (ingredient.cols - grid) / step + 1, CV_8UC3);
	for (int i = 0; i < pooled.rows; i++)
		for (int j = 0; j < pooled.cols; j++)
			for (int k = 0; k < 3; k++) {
				int max = ingredient.at<Vec3b>(step*i, step*j)[k];
				for (int ii = 0; ii < grid; ii++)
					for (int jj = 0; jj < grid; jj++)
						if (ingredient.at<Vec3b>(step*i + ii, step*j + jj)[k] > max)
							max = ingredient.at<Vec3b>(step*i + ii, step*j + jj)[k];
				pooled.at<Vec3b>(i, j)[k] = max;
			}
	return pooled;
}
void compareCharacter(Mat m1, Mat m2,int factor) {
	for (int i = 0; i < m1.rows; i++)
		for (int j = 0; j < m1.cols; j++)
			for (int k = 0; k < 3; k++) {
				if (m1.at<Vec3b>(i, j)[k] >= m2.at<Vec3b>(i, j)[k]) {
					m1.at<Vec3b>(i, j)[k] = factor;
					m2.at<Vec3b>(i, j)[k] =0;
				}
				else {
					m1.at<Vec3b>(i, j)[k] =0;
					m2.at<Vec3b>(i, j)[k] = factor;
				}
			}
}