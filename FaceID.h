#pragma once
#include<opencv2\core\core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<iostream>
#include<fstream>
#include<array>
#include<vector>
#include <amp.h>
#include<amp_math.h>
#include <algorithm>
#include<ctime>
#include<stdlib.h> 
#include<string>
#include<iomanip>
using namespace std;
using namespace cv;
#define Version_20
#define USE

#ifdef Version_20
#define MAP_ROWS 20
#define MAP_COLS 20
#endif // Version_20
#ifdef Version_100
#define MAP_ROWS 100
#define MAP_COLS 100
#endif // Version_100
#define SAMPLE_NUM 2000
#define HARD_CLASSIFIER_STAGES 1
#define MAX_WEAK_CLASSIFIER_NUM_PER_HARD 20
#define MODEL_NUM 5
#define FEATURE_NUM 1000000
#define __TP 1000
#define __TN 1000

typedef struct {
	//以下是不随迭代变化的量
	int model;//哪个大类
	int factor;//缩放因子
	int xSize;//x方向大小
	int ySize;//y方向大小
	int X;//左上角的X坐标
	int Y;//左上角的Y坐标
	int Number;//在所有特征中的编号

	//以下是在每次迭代中值不同的量
	double eRate;//该特征在阈值下的错误率
	int threshold;//该特征的当前阈值
	int p;//指示不等号的方向
} Feature;
typedef struct {
	Mat img;
	Mat integralDiagram;
	bool result;
	double weight;
} Sample;
typedef struct {
	bool key;
	int value;
	double weight;
} Key_Value;
typedef struct {
	double errorRate;
	int Number;
} ER_Number;

/* 
	------------特征模板标记及其外貌-------------
	0:	(s,t)=(1,2)
		---------
		|*******|	
		---------
		|		|
		---------	

	1:	(s,t)=(2,1)
		---------
		|	|***|
		|	|***|
		---------
	
	2:	(s,t)=(1,3)
		---------
		|		|
		---------
		|*******|	
		---------
		|		|
		---------	

	3.	(s,t)=(3,1)
		-------------
		|	|***|	|
		|	|***|	|
		-------------

	4.	(s,t)=(2,2)
		---------
		|	|***|
		---------
		|***|	|
		---------
*/
ostream& operator<<(ostream& os, Feature& feature);
#ifdef TRAIN
void GetSamples();//读入样本图
void Train();//训练
Key_Value* CalFeatureValue(Feature& feature);
void CalIntegralDiagrams();//计算样本的积分图 并返回一个矩阵
void GenerateFeatures();
void CalFeatureMinErrorRate();
Feature& StoreClassifier(ofstream& fout, int& curWeakClassifierNum, int stage);
void UpdateSampleWeight(Feature& bestFeature);
void InitialSomeVariable();
void DrawRectangle(Feature& feature, Sample& image);
void Rotate0(Feature feature, Sample &sample);
void Rotate1(Feature feature, Sample &sample);
void Rotate2(Feature feature, Sample &sample);
void Rotate3(Feature feature, Sample &sample);
void Rotate4(Feature feature, Sample &sample);
ofstream& operator<<(ofstream& fout, Feature& feature);
#endif // TRAIN


#ifdef USE
ifstream& operator>>(ifstream& fin, Feature& feature);
void LoadClassifier();
Sample* LoadAImage(string imagePathName);
void CalOneSampleIntegralDiagram(Sample* sample);
void DrawRectangle(Feature& feature, Sample& image);
int CalSampleOneFeatureValue(Sample* sample, Feature& feature);
void CalSampleAllFeatureValues(Sample* sample);
void PredictResult();
#endif // 


