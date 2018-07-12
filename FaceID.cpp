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
	//Model[0-4]
	//
	Feature* Features = new Feature[FEATURE_NUM];
	int featureNum=0;
	for (int model = 0; model < MODEL_NUM; model++) 
		for (int factor = 1;; factor++) {
			//判断跳出条件 小于最小面积或者超过图片面积
			int xSize = factor * s[model];
			int ySize = factor * t[model];
			int Square = xSize * ySize;
			if (Square<minSquare[model] || xSize>MAP_COLS||ySize>MAP_ROWS)
				break;
			for(int Y = 0; Y<=MAP_ROWS-ySize; Y++)
				for (int X = 0; X <= MAP_COLS - xSize; X++) {
					Features[featureNum].factor = factor;
					Features[featureNum].model = model;
					Features[featureNum].xSize = xSize;
					Features[featureNum].ySize = ySize;
					Features[featureNum].X = X;
					Features[featureNum].Y = Y;
					
					switch (model)
					{
					case 0: {
						for (int i = 0; i < SAMPLE_NUM; i++) {
							//Key_Value[i].Value=Samples[i].
							//Key_Value[i].Key=Samples[i].result;
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

Mat* GetSamples(string& pathName, bool*& results) {
	int imageNumber = 0;
	string line;
	ifstream fin;
	fin.open(pathName);
	if (!fin.is_open())
	{
		cout << "File is not exit" << endl;
		abort();
	}
	while (fin >> line)
		imageNumber++;
	fin.close();
	Mat *imageSet = new Mat[imageNumber];
	fin.open(pathName);
	string imagePath;
	for (int i = 0; i < imageNumber; i++)
	{
		fin >> imagePath;

		imageSet[i] = imread(imagePath, 0);
	}
}
