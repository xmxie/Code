#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
using namespace cv;
using namespace std;
int main() {
	/*
	GetSamples();//获取样本
	CalIntegralDiagrams();//计算样本积分图
	InitialSomeVariable();
	GenerateFeatures();
	*/
	/*
	ifstream fin(classifierPathName.c_str());
	int cur = 0;
	while (cur<20)
		fin >> Factor[0][cur++];
	for (int i = 0; i < cur; i++)
		DrawRectangle(Factor[0][i], samples[i]);
	//DrawRectangle(Factor[0][5], samples[0]);
	*/
	LoadClassifier();
	//Sample* sample = LoadAImage("Code/pos/0002_j_01204.jpg");
	Sample* sample = LoadAImage("Code/neg/000009.jpg");
	CalSampleAllFeatureValues(sample);
	PredictResult();
	cout << P << endl;
	int NO = 0;
	for (int i = 0; i < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; i++) {
		if (predictResult[i]) {
			cout << "NO:" << NO++ << endl;
			cout << "Threshold: " << weakFeatures[i].threshold*weakFeatures[i].p << endl;
			cout << "CalValue: " << sampleFeatureValue[i] * weakFeatures[i].p << endl;
			cout << endl;
			DrawRectangle(weakFeatures[i], *sample);
		}
	}
	cin.get();
	//Train();//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重
}
