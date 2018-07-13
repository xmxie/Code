#include"FaceID.h"
#include"Global.h"

using namespace std;
int main() {
	samples = GetSamples();//获取样本
	CalIntegralDiagrams();//计算样本积分图
	Train();//训练样本 期间会更新样本权重以及各强分类器中的弱分类器的权重
}
