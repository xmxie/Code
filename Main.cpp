#include"FaceID.h"
#include"Global.h"

int main() {
	GetSamples();//获取样本
	CalIntegralDiagrams();//计算样本积分图
	InitialSomeVariable();
	GenerateFeatures();
	ifstream fin(classifierPathName.c_str());
	int cur = 0;
	while (cur<20)
		fin >> Factor[0][cur++];
	//for (int i = 0; i < cur; i++)
		//DrawRectangle(Factor[0][i], samples[0]);
	DrawRectangle(Factor[0][5], samples[0]);

	//Train();//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重
}
