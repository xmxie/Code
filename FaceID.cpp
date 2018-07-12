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
	int featureNum=0;//初始化特征数为0

	for (int model = 0; model < MODEL_NUM; model++) 
		for (int factor = 1;; factor++) {
			//判断跳出条件 小于最小面积或者超过图片面积
			int xSize = factor * s[model];//计算窗口长
			int ySize = factor * t[model];//计算窗口高
			int Square = xSize * ySize;//计算窗口面积
			if (Square<minSquare[model] || xSize>MAP_COLS||ySize>MAP_ROWS)//面积过小 或者长高超限就跳出循环
				break;

			for(int Y = 1; Y<=MAP_ROWS-ySize; Y++)
				for (int X = 1; X <= MAP_COLS - xSize; X++) {
					Features[featureNum].factor = factor;
					Features[featureNum].model = model;
					Features[featureNum].xSize = xSize;
					Features[featureNum].ySize = ySize;
					Features[featureNum].X = X;
					Features[featureNum].Y = Y;

					Key_Value keyValues[SAMPLE_NUM];
					switch (model)
					{
					case 0: {
						for (int i = 0; i < SAMPLE_NUM; i++) {
							keyValues[i].value = samples[i].integralDiagram.at<uchar>(X,Y)+ samples[i].integralDiagram.at<uchar>(X+factor, Y+factor)
								- samples[i].integralDiagram.at<uchar>(X,Y+factor)- samples[i].integralDiagram.at<uchar>(X+factor,Y);
							keyValues[i].key=samples[i].result;
						}
					}break;
					case 1: {

					}break;
					case 2: {

					}break;
					case 3: {

					}break;
					case 4: {

					}break;
					default:
						break;
					}
					//vector.sort();
					int min = SAMPLE_NUM;
					int __SP = 0;
					int __SN = 0;
					for (int i = 0; i < SAMPLE_NUM; i++) {
						//if( [i] isPerson)
						//	__SP++
						//else
						//	__SN++

					}
						
				}
		}

}
Sample* GetSamples(string& pathName) {
	ifstream fin;
	fin.open(pathName);
	if (!fin.is_open())
	{
		cout << "File is not exit" << endl;
		abort();
	}
	Sample* imageSet = new Sample[SAMPLE_NUM];
	string imagePath;
	for (int i = 0; i < SAMPLE_NUM; i++)
	{
		fin >> imagePath >> imageSet[i].result;
		imageSet[i].img = imread(imagePath, 0);
	}
	return imageSet;
}
