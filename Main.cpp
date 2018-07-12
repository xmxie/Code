#include"FaceID.h"

int main() {
	/*//以下是定义二维数组用以创建卷积核
	Kernel vertical_edge_kernel = new float*[3];
	Kernel horizontal_edge_kernel = new float*[3];
	for (int i = 0; i < 3; i++) {
		vertical_edge_kernel[i] = new float[3]{ 1,0,-1 };//这个是竖直边界检测器
		horizontal_edge_kernel[i] = new float[3]{(float)(1-i),(float)(1 - i),(float)(1 - i) };//这个是水平边界检测器
	}
	Point anchor(-1, -1);//锚点。。。直接全设为（-1，-1）即可
	Mat vertical_kernel = generateConvKernel(3, vertical_edge_kernel);//生成的竖直边界检测器
	Mat horizontal_kernel = generateConvKernel(3, horizontal_edge_kernel);//生成的水平边界检测器
	Mat img = imread("Code/1.jpg");//读取图片
	namedWindow("Eason");//创建三个窗口
	//namedWindow("Verticaled Eason");
	//namedWindow("Horizontaled Eason");
	imshow("Eason", img);
	waitKey(0);
	getchar();

	Mat vertical_out;
	Mat horizontal_out;
	Mat pool_out;
	filter2D(img, vertical_out, -1, vertical_kernel, anchor, 0);//竖直卷积
	namedWindow("竖直检测");
	imshow("竖直检测", vertical_out);
	filter2D(img,horizontal_out, -1, horizontal_kernel, anchor, 0);//水平卷积
	namedWindow("水平检测");
	imshow("水平检测", horizontal_out);
	compareCharacter(vertical_out, horizontal_out, 100);
	namedWindow("比较后竖直检测");
	imshow("比较后竖直检测", vertical_out);
	namedWindow("比较后水平检测");
	imshow("比较后水平检测", vertical_out);
	//pool_out = maxPooling(vertical_out, 2, 2);//池化
	//filter2D(img, vertical_out, -1, vertical_kernel, anchor, 0);
	//pool_out = maxPooling(vertical_out, 2, 2);

	//imshow("Pooled Eason", pool_out);
	//imshow("Horizontaled Eason",horizontal_out);
	//imshow("Verticaled Eason", vertical_out);
	//imshow("Eason", img);
	waitKey(0);
	getchar();
	*/
	Mat img = imread("Code/1.jpg");
	Mat gray,mean, dev,After;
	cvtColor(img, gray, CV_RGB2GRAY);
	meanStdDev(img, mean, dev);
	double Mean, Dev;
	Mean = mean.at<double>(0,0);
	Dev = dev.at<double>(0,0);
	cout << "Mean:" << Mean << endl;
	cout << "Dev:" << Dev << endl;
	namedWindow("Fuck");
	for (int i = 0; i < gray.rows; i++)
		for (int j = 0; j < gray.cols; j++)
			gray.at<float>(i, j) = ((double)gray.at<uchar>(i, j) - Mean) / Dev;
	imshow("Fuck", gray);
	waitKey(0);
	cin.get();
}
