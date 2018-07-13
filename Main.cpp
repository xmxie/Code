#include"FaceID.h"

int main() {
	string posPathName; //正样本路径名
	string negPathName;//负样本路径名
	string* classifierPathName;//各级强分类器中的弱分类器权重路径名数组
	Sample* samples;//样本数组

	samples = GetSamples(posPathName, negPathName);//获取样本
	CalIntegralDiagrams(samples);//计算样本积分图
	Train(samples);//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重
}
