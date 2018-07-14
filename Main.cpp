#include"FaceID.h"
#include"Global.h"
using namespace cv;
using namespace std;
int main() {
	/*
	GetSamples();//获取样本
	CalIntegralDiagrams();//计算样本积分图
	InitialSomeVariable();//初始化一部分变量
	GenerateFeatures();//生成特征
	Train();//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重

	*/
	LoadClassifier();
	Sample* bigSample = LoadAImage("Code/pos/0002_j_01204.jpg");
	//Sample* bigSample = LoadAImage("Code/neg/000009.jpg");
	Sample* smallSample;
	CalOneSampleIntegralDiagram(smallSample);
	CalSampleAllFeatureValues(smallSample);
	PredictResult();
	cout << P << endl;
	int NO = 0;
	if(P>0.5)
		for (int i = 0; i < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; i++) 
			if (predictResult[i]) {
				int Delta = weakFeatures[i].p*(sampleFeatureValue[i] - weakFeatures[i].threshold);
				Delta = Delta / (weakFeatures[i].xSize*weakFeatures[i].ySize);
				cout << "NO:" << NO++ << endl;
				cout << "Threshold: " << weakFeatures[i].threshold*weakFeatures[i].p << endl;
				cout << "CalValue: " << sampleFeatureValue[i] * weakFeatures[i].p << endl;
				cout << "Delta/Square: " << Delta << endl;
				cout << endl;
				Rotate(weakFeatures[i], *bigSample);
			}
	cin.get();
}
