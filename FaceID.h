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
using namespace std;
using namespace cv;
#define MAP_ROWS 20
#define MAP_COLS 20
#define SAMPLE_NUM 200
#define HARD_CLASSIFIER_STAGES 1
#define MAX_WEAK_CLASSIFIER_NUM_PER_HARD 20
#define MODEL_NUM 5
#define FEATURE_NUM 200000
#define __TP 100
#define __TN 100
typedef struct {
	int model;//哪个大类
	int factor;//缩放因子
	int xSize;
	int ySize;
	int X;
	int Y;
	double eRate;
	int threshold;
	int p;
	int Number;
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
}Key_Value;

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

Sample* GetSamples();//读入样本图
void Train();//训练
Key_Value* CalFeatureValue(Feature& feature);
void CalIntegralDiagrams();//计算样本的积分图 并返回一个矩阵
void GenerateFeatures();
ostream& operator<<(ostream& os, Feature& feature);
ofstream& operator<<(ofstream& fout, Feature& feature);
void CalFeatureMinErrorRate();
Feature& StoreClassifier(int& curWeakClassifierNum,int stage);
void UpdateSampleWeight(Feature& bestFeature);


