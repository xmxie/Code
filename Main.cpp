#include"FaceID.h"

int main() {
<<<<<<< HEAD
	/*string samplePathName;//����·����
	string sampleWeightPathName;//����Ȩ��·����
	string* classifierPathName;//����ǿ�������е���������Ȩ��·��������
	Mat* samples;//��������
	Mat*integralDiagrams;//����ͼ����
	Mat sampleWeights;//����Ȩ�ؾ���
	bool* results;

	//sampleWeights = LoadSampleWeights(sampleWeightPathName);//��������Ȩ��
	samples = GetSamples(samplePathName,results);//��ȡ����
	integralDiagrams = CalIntegralDiagrams(samples);//������������ͼ
	Train(samples, integralDiagrams);//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��*/
	Mat test(200, 200, CV_8UC3, Scalar(100,100,10));
	Vec3b testimg = test.at<Vec3b>(1, 1);
	cout << testimg << endl;
	test.at<Vec3b>(1, 1) = (0, 0, 0);
	testimg = test.at<Vec3b>(1, 1);
	rectangle(test, Rect(0, 0, 50, 50), 10, -1, 2,0);
	cout << testimg << endl;
	imshow("test", test);
	waitKey(0);
=======
	string posPathName="Code/pos/pos.txt"; //������·����
	string negPathName="Code/neg/neg.txt";//������·����
	//string* classifierPathName;//����ǿ�������е���������Ȩ��·��������
	Sample* samples;//��������

	samples = GetSamples(posPathName, negPathName);//��ȡ����
	CalIntegralDiagrams(samples);//������������ͼ
	Train(samples);//ѵ������ �ڼ���������Ȩ���Լ���ǿ�������е�����������Ȩ��
>>>>>>> 05599edc04428af42d56f47e1dc7682201123859
}
