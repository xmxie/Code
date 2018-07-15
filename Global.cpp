#include"FaceID.h"
int weakClassifierNum[HARD_CLASSIFIER_STAGES] = { 30 };
int minSquare[MODEL_NUM] = { 3,3,4,4,5 };
//int minSquare[MODEL_NUM] = {  };
int s[MODEL_NUM] = { 1,2,1,3,2 };
int t[MODEL_NUM] = { 2,1,3,1,2 };
string classifierPathName = "Code/classifiers.txt";
#ifdef TRAIN
Feature** Factor;//各级强分类器中各弱分类器的当时副本
Sample* samples;//样本数组
Feature* Features;//特征数组
int featureNum=0;//全部特征数
ER_Number* ERtable;
double curTP=(double)__TP / SAMPLE_NUM;
double curTN=(double)__TN / SAMPLE_NUM;
#endif // !TRAIN
#ifdef USE
double weakFactors[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
Feature weakFeatures[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
int sampleFeatureValue[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
bool predictResult[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
double P=0;
#endif // USE


#ifdef Version_100
string posPathName = "Code/pos_100/pos.txt";
string negPathName = "Code/neg_100/neg.txt";
#endif // Version_100
#ifdef Version_20
string posPathName = "Code/pos/pos.txt";
string negPathName = "Code/neg/neg.txt";
#endif







