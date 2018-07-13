#include"FaceID.h"

int main() {
	/*string samplePathName;//样本路径名
	string sampleWeightPathName;//样本权重路径名
	string* classifierPathName;//各级强分类器中的弱分类器权重路径名数组
	Mat* samples;//样本数组
	Mat*integralDiagrams;//积分图数组
	Mat sampleWeights;//样本权重矩阵
	bool* results;

	//sampleWeights = LoadSampleWeights(sampleWeightPathName);//加载样本权重
	samples = GetSamples(samplePathName,results);//获取样本
	integralDiagrams = CalIntegralDiagrams(samples);//计算样本积分图
	Train(samples, integralDiagrams);//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重*/
	Mat test(200, 200, CV_8UC3, Scalar(100,100,10));
	Vec3b testimg = test.at<Vec3b>(1, 1);
	cout << testimg << endl;
	test.at<Vec3b>(1, 1) = (0, 0, 0);
	testimg = test.at<Vec3b>(1, 1);
	rectangle(test, Rect(0, 0, 50, 50), 10, -1, 2,0);
	cout << testimg << endl;
	imshow("test", test);
	waitKey(0);
}
