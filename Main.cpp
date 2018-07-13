#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
int main() {
	Mat test = imread("E:/C++/test.jpg");
	Mat change;
	namedWindow("test", WINDOW_NORMAL);
	resize(test, change, Size(test.cols * 2, test.rows * 2), 0, 0);
	imshow("test", test);
	imshow("change", change);
	waitKey(0);
}
