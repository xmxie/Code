#pragma once
#include"FaceID.h"

extern int weakClassifierNum[HARD_CLASSIFIER_STAGES];
extern int minSquare[MODEL_NUM];
extern int s[MODEL_NUM];
extern int t[MODEL_NUM];
extern string classifierPathName;
#ifdef TRAIN
extern Feature** Factor;
extern Sample* samples;
extern Feature* Features;
extern int featureNum;
extern ER_Number* ERtable;
extern double curTP;
extern double curTN;
#endif // TRAIN
#ifdef USE
extern double weakFactors[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
extern Feature weakFeatures[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
extern int sampleFeatureValue[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
extern bool predictResult[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
extern double P;
#endif // USE

extern string posPathName;
extern string negPathName;





