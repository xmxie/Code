#include"FaceID.h"

int weakClassifierNum[HARD_CLASSIFIER_STAGES] = {};
int minSquare[MODEL_NUM] = {};
int s[MODEL_NUM];
int t[MODEL_NUM];
void CalIntegralDiagrams(Sample* samples){
	for (int num = 0; num < SAMPLE_NUM; num++){
		samples[num].integralDiagram = samples[num].img.clone();
		samples[num].integralDiagram.at<uchar>(0, 0) = samples[num].img.at<uchar>(0, 0);
		for (int j = 1; j < MAP_ROWS; j++)//求出第一列的值
			samples[num].integralDiagram.at<uchar>(j, 0) = 
			samples[num].integralDiagram.at<uchar>(j-1, 0)
			+ samples[num].img.at<uchar>(j, 0);
		for (int i = 1; i < MAP_ROWS; i++)//求出第一行的值
			samples[num].integralDiagram.at<uchar>(0, i) = 
			samples[num].integralDiagram.at<uchar>(0, i-1)
			+ samples[num].img.at<uchar>(0, i);
		for (int j = 1; j < MAP_ROWS; j++) {
			for (int i = 1; i < MAP_COLS; i++) {
				samples[num].integralDiagram.at<uchar>(j, i) =
					samples[num].integralDiagram.at<uchar>(j, i - 1)
					+ samples[num].integralDiagram.at<uchar>(j - 1, i)
					+ samples[num].img.at<uchar>(j, i)
					- samples[num].integralDiagram.at<uchar>(j - 1, i - 1);
			}
		}
	}
}
void Train(Sample* samples) {
	Feature* Features = new Feature[FEATURE_NUM];
	ER_Number* ERtable = new ER_Number[FEATURE_NUM];
	int featureNum=0;//初始化特征数为0
	double curTP = (double)__TP / SAMPLE_NUM;
	double curTN = (double)__TN / SAMPLE_NUM;
	ofstream Fout("classifiers.txt");

	for (int model = 0; model < MODEL_NUM; model++) {
		Feature CurBestFeature;
		printf("开始处理模型%d\n", model);
		for (int factor = 1;; factor++) {
			//判断跳出条件 小于最小面积或者超过图片面积
			int xSize = factor * s[model];//计算窗口长
			int ySize = factor * t[model];//计算窗口高
			int Square = xSize * ySize;//计算窗口面积
			if (Square<minSquare[model] || xSize>MAP_COLS || ySize > MAP_ROWS) {//面积过小 或者长高超限就跳出循环
				printf("重置放大因子\n");
				break;
			}
			printf("放大因子为%d\n", factor);
			for (int Y = 0; Y <= MAP_ROWS - ySize; Y++)
				for (int X = 0; X <= MAP_COLS - xSize; X++) {
					Features[featureNum].factor = factor;
					Features[featureNum].model = model;
					Features[featureNum].xSize = xSize;
					Features[featureNum].ySize = ySize;
					Features[featureNum].X = X;
					Features[featureNum].Y = Y;

					Key_Value* keyValues = CalFeatureValue(samples, Features[featureNum]);
					sort(keyValues, keyValues + SAMPLE_NUM, [](Key_Value kv1, Key_Value kv2) {return kv1.value > kv2.value; });
					double minWrong = SAMPLE_NUM;
					double __SP = 0;
					double __SN = 0;
					double wrong10, wrong01;
					for (int i = 0; i < SAMPLE_NUM; i++) {
						if (keyValues[i].key)
							__SP+=keyValues[i].weight;
						else __SN += keyValues[i].weight;
						wrong10 = __SN + curTP - __SP;
						wrong01 = __SP + curTN - __SN;
						if (wrong10 < minWrong&&wrong10 < wrong01) {
							minWrong = wrong10;
							Features[featureNum].p = 1;
							Features[featureNum].threshold = keyValues[i].value;
						}
						else if (wrong01 < minWrong&&wrong01 < wrong10) {
							minWrong = wrong01;
							Features[featureNum].p = -1;
							Features[featureNum].threshold = keyValues[i].value;
						}
					}
					ERtable[featureNum].errorRate = Features[featureNum].eRate = (double)minWrong / SAMPLE_NUM;
					ERtable[featureNum].Number = featureNum;
					featureNum++;
					delete[] keyValues;
				}
		}
		sort(ERtable, ERtable + featureNum, [](ER_Number& ern1, ER_Number& ern2) {return ern1.Number < ern2.Number; });
		cout << Features[ERtable[0].Number];
		CurBestFeature=StoreClassifier(Fout, Features, ERtable);
		UpdateSampleWeight(samples,);
	}
}
Sample* GetSamples(string& posPathName, string& negPathName) {
	ifstream fin; 
	Sample* imageSet = new Sample[__TP+__TN];
	string imagePath;
	fin.open(posPathName);
	if (!fin.is_open())
	{
		cout << "File is not exit" << endl;
		abort();
	}
	for (int i = 0; i < __TP ; i++)
	{
		fin >> imagePath >> imageSet[i].result;
		imageSet[i].img = imread(imagePath, 0);
		imageSet[i].weight = 1.0 / SAMPLE_NUM;
	}
	fin.close();
	fin.open(negPathName);
	if (!fin.is_open())
	{
		cout << "File is not exit" << endl;
		abort();
	}
	for (int j = __TP ; j < __TP + __TN; j++)
	{
		fin >> imagePath >> imageSet[j].result;
		imageSet[j].img = imread(imagePath,0);
		imageSet[j].weight = 1.0 / SAMPLE_NUM;
	}
	fin.close();
	return imageSet;
}
Key_Value* CalFeatureValue(Sample* samples, Feature& feature) {
	Key_Value* keyValues = new Key_Value[SAMPLE_NUM];
	switch (feature.model)
	{
	case 0: {
		uchar X_Y, X_YF, X_YFF, XF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + feature.factor) : 0;
			X_YFF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + 2 * feature.factor) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y) : 0;
			keyValues[i].value = X_Y + 2 * samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + feature.factor) + X_YFF - XF_Y
				- 2 * X_YF - samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + 2 * feature.factor);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 1: {
		uchar X_Y, X_YF, XF_Y, XFF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + feature.factor) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y) : 0;
			XFF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + 2 * feature.factor, feature.Y) : 0;
			keyValues[i].value = X_YF + 2 * XF_Y + samples[i].integralDiagram.at<uchar>(feature.X + 2 * feature.factor, feature.Y + feature.factor)
				- X_Y - 2 * samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + feature.factor) - XFF_Y;
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 2: {
		uchar X_Y, X_YF, XF_Y, X_YFF, X_YFFF;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + feature.factor) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y) : 0;
			X_YFF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + 2 * feature.factor) : 0;
			X_YFFF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + 3 * feature.factor) : 0;
			keyValues[i].value = 3 * X_YF + XF_Y + X_YFFF + 3 * samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + 2 * feature.factor)
				- X_Y - 3 * X_YFF - samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + 3 * feature.factor) - 3 * samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + feature.factor);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 3: {
		uchar X_Y, X_YF, XF_Y, XFF_Y, XFFF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + feature.factor) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y) : 0;
			XFF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + 2 * feature.factor, feature.Y) : 0;
			XFFF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + 3 * feature.factor, feature.Y) : 0;
			keyValues[i].value = 3 * XF_Y + X_YF + XFFF_Y + 3 * samples[i].integralDiagram.at<uchar>(feature.X + 2 * feature.factor, feature.Y + feature.factor)
				- X_Y - 3 * XFF_Y - samples[i].integralDiagram.at<uchar>(feature.X + 3 * feature.factor, feature.Y + feature.factor) - 3 * samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + feature.factor);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 4: {
		uchar X_Y, X_YF, XF_Y, XFF_Y, X_YFF;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + feature.factor) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y) : 0;
			XFF_Y = feature.Y ? samples[i].integralDiagram.at<uchar>(feature.X + 2 * feature.factor, feature.Y) : 0;
			X_YFF = feature.X ? samples[i].integralDiagram.at<uchar>(feature.X, feature.Y + 2 * feature.factor) : 0;
			keyValues[i].value = 2 * XF_Y + 2 * X_YF + 2 * samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + 2 * feature.factor) + 2 * samples[i].integralDiagram.at<uchar>(feature.X + 2 * feature.factor, feature.Y + feature.factor)
				- X_Y - X_YFF - XFF_Y - 4 * samples[i].integralDiagram.at<uchar>(feature.X + feature.factor, feature.Y + feature.factor) - samples[i].integralDiagram.at<uchar>(feature.X + 2 * feature.factor, feature.Y + 2 * feature.factor);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	default:
		break;
	}
	return keyValues;
}
ostream& operator<<(ostream& os, Feature& feature) {
	os << "模型: " << feature.model<<endl
		<<"放大倍数: "<<feature.factor<<endl
		<<"左上角位置: "<<"("<<feature.X<<","<<feature.Y<<")"<<endl
		<<"错误率: "<<feature.eRate<<endl
		<<"阈值："<<feature.threshold<<endl
		<<"符号："<<feature.p<<endl;
	return os;
}
Feature& StoreClassifier(ofstream& fout, Feature* allFeatures, ER_Number* ERtable) {

}
void UpdateSampleWeight(Sample* samples,Feature& bestFeature) {
	Key_Value* keyValues = CalFeatureValue(samples, bestFeature);
	double Alpha = log((1 - bestFeature.eRate) / bestFeature.eRate) / 2;
	for (int i = 0; i < SAMPLE_NUM; i++) {
		bool predictIsTrue = ((bestFeature.p*keyValues[i].value >= bestFeature.p*bestFeature.threshold)==keyValues[i].key);
		if(predictIsTrue)

	}
}


