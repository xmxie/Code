#pragma once
#include"FaceID.h"

extern Feature** Factor;
extern int weakClassifierNum[HARD_CLASSIFIER_STAGES];
extern int minSquare[MODEL_NUM];
extern int s[MODEL_NUM];
extern int t[MODEL_NUM];
extern Sample* samples;
extern Feature* Features;
extern int featureNum;
extern ER_Number* ERtable;
extern double curTP;
extern double curTN;
extern string posPathName;
extern string negPathName;




