#include"FaceID.h"
#include"Global.h"
#ifdef TRAIN
void CalIntegralDiagrams() {
	for (int num = 0; num < SAMPLE_NUM; num++) {
		//samples[num].integralDiagram = samples[num].img.clone();
		samples[num].img.convertTo(samples[num].integralDiagram, CV_32SC1);
		samples[num].integralDiagram.at<int>(0, 0) = samples[num].img.at<uchar>(0, 0);
		for (int j = 1; j < MAP_ROWS; j++)//求出第一列的值
			samples[num].integralDiagram.at<int>(j, 0) =
			samples[num].integralDiagram.at<int>(j - 1, 0)
			+ samples[num].img.at<uchar>(j, 0);
		for (int i = 1; i < MAP_ROWS; i++)//求出第一行的值
			samples[num].integralDiagram.at<int>(0, i) =
			samples[num].integralDiagram.at<int>(0, i - 1)
			+ samples[num].img.at<uchar>(0, i);
		for (int j = 1; j < MAP_ROWS; j++) {
			for (int i = 1; i < MAP_COLS; i++) {
				samples[num].integralDiagram.at<int>(j, i) =
					samples[num].integralDiagram.at<int>(j, i - 1)
					+ samples[num].integralDiagram.at<int>(j - 1, i)
					+ samples[num].img.at<uchar>(j, i)
					- samples[num].integralDiagram.at<int>(j - 1, i - 1);
			}
		}
	}
}
void GenerateFeatures() {
	int xSize, ySize, Square;
	for (int model = 0; model < MODEL_NUM; model++) {
		for (int factor = 1;; factor++) {
			//判断跳出条件 小于最小面积或者超过图片面积
			xSize = factor * s[model];//计算窗口长
			ySize = factor * t[model];//计算窗口高
			Square = xSize * ySize;//计算窗口面积
			if (xSize > MAP_COLS || ySize > MAP_ROWS) {//面积过小 或者长高超限就跳出循环
				break;
			}
			else if (Square < minSquare[model])
				continue;
			for (int Y = 0; Y <= MAP_ROWS - ySize; Y++)
				for (int X = 0; X <= MAP_COLS - xSize; X++) {
					Features[featureNum].factor = factor;
					Features[featureNum].model = model;
					Features[featureNum].xSize = xSize;
					Features[featureNum].ySize = ySize;
					Features[featureNum].X = X;
					Features[featureNum].Y = Y;
					Features[featureNum].Number = featureNum;
					featureNum++;
				}
		}
	}

}
void CalFeatureMinErrorRate() {
	Key_Value* keyValues;
	double minWrong;
	double __SP, __SN, wrong10, wrong01;
	int test = 0;
	for (int fIndex = 0; fIndex < featureNum; fIndex++) {
		keyValues = CalFeatureValue(Features[fIndex]);
		sort(keyValues, keyValues + SAMPLE_NUM, [](Key_Value kv1, Key_Value kv2) {return kv1.value > kv2.value; });
		minWrong = 1;
		__SP = __SN = 0;
		for (int sIndex = 0; sIndex < SAMPLE_NUM; sIndex++) {
			if (keyValues[sIndex].key)
				__SP += keyValues[sIndex].weight;
			else __SN += keyValues[sIndex].weight;
			wrong10 = __SN + curTP - __SP;
			wrong01 = __SP + curTN - __SN;
			if (wrong10 < 0)
				wrong10 = 0;
			if (wrong01 < 0)
				wrong01 = 0;
			if (wrong10 < minWrong&&wrong10 < wrong01) {
				minWrong = wrong10;
				Features[fIndex].p = 1;
				Features[fIndex].threshold = keyValues[sIndex].value;
			}
			else if (wrong01 < minWrong&&wrong01 < wrong10) {
				minWrong = wrong01;
				Features[fIndex].p = -1;
				Features[fIndex].threshold = keyValues[sIndex].value;
			}
		}
		ERtable[fIndex].errorRate = Features[fIndex].eRate = minWrong;
		ERtable[fIndex].Number = fIndex;
		delete[] keyValues;
	}
}
void InitialSomeVariable() {
	Features = new Feature[FEATURE_NUM];
	ERtable = new ER_Number[FEATURE_NUM];
	Factor = new Feature*[HARD_CLASSIFIER_STAGES];
	for (int i = 0; i < HARD_CLASSIFIER_STAGES; i++)
		Factor[i] = new Feature[MAX_WEAK_CLASSIFIER_NUM_PER_HARD];
}
void Train() {
	ofstream fout(classifierPathName.c_str());
	for (int stage = 0; stage < HARD_CLASSIFIER_STAGES; stage++) {
		for (int curWeakClassifierNum = 0; curWeakClassifierNum < weakClassifierNum[stage]; curWeakClassifierNum++) {
			CalFeatureMinErrorRate();
			sort(ERtable, ERtable + featureNum, [](ER_Number& ern1, ER_Number& ern2) {return ern1.errorRate < ern2.errorRate; });
			Feature& CurBestFeature = StoreClassifier(fout, curWeakClassifierNum, stage);
			UpdateSampleWeight(CurBestFeature);
		}
	}
}
void GetSamples() {
	ifstream fin;
	samples = new Sample[__TP + __TN];
	string imagePath;
	fin.open(posPathName);
	if (!fin.is_open()){
		cout << "File is not exit" << endl;
		abort();
	}
	for (int i = 0; i < __TP; i++){
		fin >> imagePath >> samples[i].result;
		samples[i].img = imread(imagePath, 0);
		samples[i].weight = 1.0 / SAMPLE_NUM;
	}
	fin.close();
	fin.open(negPathName);
	if (!fin.is_open()){
		cout << "File is not exit" << endl;
		abort();
	}
	for (int j = __TP; j < __TP + __TN; j++){
		fin >> imagePath >> samples[j].result;
		samples[j].img = imread(imagePath, 0);
		samples[j].weight = 1.0 / SAMPLE_NUM;
	}
	fin.close();
}
Key_Value* CalFeatureValue(Feature& feature) {
	Key_Value* keyValues = new Key_Value[SAMPLE_NUM];
	switch (feature.model){
	case 0: {
		int  X_Y, X_YF, X_YFF, XF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			X_YFF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + 2 * feature.factor - 1) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			keyValues[i].value = X_Y + 2 * samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1) + X_YFF - XF_Y
				- 2 * X_YF - samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 2 * feature.factor - 1);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 1: {
		int  X_Y, X_YF, XF_Y, XFF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			XFF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y) : 0;
			keyValues[i].value = X_YF + 2 * XF_Y + samples[i].integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + feature.factor - 1)
				- X_Y - 2 * samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1) - XFF_Y;
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 2: {
		int  X_Y, X_YF, XF_Y, X_YFF, X_YFFF;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			X_YFF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + 2 * feature.factor - 1) : 0;
			X_YFFF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + 3 * feature.factor - 1) : 0;
			keyValues[i].value = 3 * X_YF + XF_Y + X_YFFF + 3 * samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 2 * feature.factor - 1)
				- X_Y - 3 * X_YFF - samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 3 * feature.factor - 1) - 3 * samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 3: {
		int  X_Y, X_YF, XF_Y, XFF_Y, XFFF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			XFF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y) : 0;
			XFFF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + 3 * feature.factor - 1, feature.Y) : 0;
			keyValues[i].value = 3 * XF_Y + X_YF + XFFF_Y + 3 * samples[i].integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + feature.factor - 1)
				- X_Y - 3 * XFF_Y - samples[i].integralDiagram.at<int >(feature.X + 3 * feature.factor - 1, feature.Y + feature.factor - 1) - 3 * samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	case 4: {
		int  X_Y, X_YF, XF_Y, XFF_Y, X_YFF;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? samples[i].integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			XFF_Y = feature.Y ? samples[i].integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y) : 0;
			X_YFF = feature.X ? samples[i].integralDiagram.at<int >(feature.X, feature.Y + 2 * feature.factor - 1) : 0;
			keyValues[i].value = 2 * XF_Y + 2 * X_YF + 2 * samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 2 * feature.factor - 1) + 2 * samples[i].integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + feature.factor - 1)
				- X_Y - X_YFF - XFF_Y - 4 * samples[i].integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1) - samples[i].integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + 2 * feature.factor - 1);
			keyValues[i].key = samples[i].result;
			keyValues[i].weight = samples[i].weight;
		}
	}break;
	default:
		break;
	}
	return keyValues;
}
ostream& operator<<(ostream& os, Feature& feature) {
	os << "模型: " << feature.model << endl
		<< "放大倍数: " << feature.factor << endl
		<< "左上角位置: " << "(" << feature.X << "," << feature.Y << ")" << endl
		<< "错误率: " << feature.eRate << endl
		<< "权重系数: " << log((1 - feature.eRate) / feature.eRate) / 2
		<< "阈值：" << feature.threshold << endl;
	//<< "符号：" << feature.p << endl
	//<< "编号：" << feature.Number << endl;
	return os;
}
ofstream& operator<<(ofstream& fout, Feature& feature) {
	fout << left << setw(3) << feature.model
		<< setw(3) << feature.factor
		<< setw(5) << feature.X
		<< setw(5) << feature.Y
		<< setw(12) << feature.eRate
		<< setw(12) << log((1 - feature.eRate) / feature.eRate) / 2
		<< setw(5) << feature.threshold
		<< setw(3) << feature.p
		<< endl;
	return fout;
}
ifstream& operator>>(ifstream& fin, Feature& feature) {
	double tmp;
	fin >> feature.model
		>> feature.factor
		>> feature.X
		>> feature.Y
		>> feature.eRate
		>> tmp
		>> feature.threshold
		>> feature.p;
	feature.xSize = s[feature.model] * feature.factor;
	feature.ySize = t[feature.model] * feature.factor;
	return fin;
}
Feature& StoreClassifier(ofstream& fout, int& curWeakClassifierNum, int stage) {
	int index;
	for (int i = 0; i < featureNum; i++) {
		bool ok = true;
		for (int j = 0; j < curWeakClassifierNum; j++)
			if (ERtable[i].Number == Factor[stage][j].Number) {
				ok = false;
				break;
			}
		if (ok) {
			index = ERtable[i].Number;
			Factor[stage][curWeakClassifierNum] = Features[index];
			break;
		}
	}
	//cout << "第"<< curWeakClassifierNum +1<<"个"<<endl<<Features[index]<<endl;
	fout << Features[index];
	return Features[index];
}
void UpdateSampleWeight(Feature& bestFeature) {
	Key_Value* keyValues = CalFeatureValue(bestFeature);
	double Alpha = log((1 - bestFeature.eRate) / bestFeature.eRate) / 2;
	double sum = 0;
	bool predictIsTrue;
	curTP = curTN = 0;
	for (int i = 0; i < SAMPLE_NUM; i++) {
		predictIsTrue = ((bestFeature.p*keyValues[i].value >= bestFeature.p*bestFeature.threshold) == keyValues[i].key);
		if (predictIsTrue)
			samples[i].weight = samples[i].weight*exp(-Alpha);
		else
			samples[i].weight = samples[i].weight*exp(Alpha);
		sum += samples[i].weight;
	}
	for (int i = 0; i < SAMPLE_NUM; i++) {
		samples[i].weight = samples[i].weight / sum;
		if (samples[i].result)
			curTP += samples[i].weight;
		else
			curTN += samples[i].weight;
	}
	delete[] keyValues;
}
#endif // TRAIN
#ifdef USE
ifstream& operator>>(ifstream& fin, Feature& feature) {
	double tmp;
	fin >> feature.model
		>> feature.factor
		>> feature.X
		>> feature.Y
		>> feature.eRate
		>> tmp
		>> feature.threshold
		>> feature.p;
	feature.xSize = s[feature.model] * feature.factor;
	feature.ySize = t[feature.model] * feature.factor;
	return fin;
}
Sample* LoadAImage(string imagePathName) {
	Sample* sample = new Sample;
	sample->img = imread(imagePathName,0);
	return sample;
}
void DrawRectangle(Feature &feature, Sample &sample) {
	Sample image;
	image.img = sample.img.clone();

	switch (feature.model) {
	case(0): {
		for (int count = 1; count <= 2; count++)
			rectangle(image.img, Rect(feature.X, feature.Y, feature.xSize, feature.factor * count), Scalar(0, 0, 0), 1, 1, 0);
		break; }
	case(1): {
		for (int count = 1; count <= 2; count++)
			rectangle(image.img, Rect(feature.X, feature.Y, feature.factor * count, feature.ySize), Scalar(0, 0, 0), 1, 1, 0);
		break; }
	case(2): {
		for (int count = 1; count <= 3; count++)
			rectangle(image.img, Rect(feature.X, feature.Y, feature.xSize, feature.factor * count), Scalar(0, 0, 0), 1, 1, 0);
		break; }
	case(3): {
		for (int count = 1; count <= 3; count++)
			rectangle(image.img, Rect(feature.X, feature.Y, feature.factor * count, feature.ySize), Scalar(0, 0, 0), 1, 1, 0);
		break; }
	case(4): {
		for (int count = 1; count <= 2; count++)
			rectangle(image.img, Rect(feature.X, feature.Y, feature.factor* count, feature.factor * count), Scalar(0, 0, 0), 1, 1, 0);
		break; }
	}
	static int name = 0;

	imwrite("drawnimage/" + to_string(name) + ".jpg", image.img);
	name += 1;
}
void Rotate(Feature& feature, Sample& sample) {
	switch (feature.model)
	{
	case 0:Rotate0(feature, sample);
		break;
	case 1:Rotate1(feature, sample);
		break;
	case 2:Rotate2(feature, sample);
		break;
	case 3:Rotate3(feature, sample);
		break;
	case 4:Rotate4(feature, sample);
		break;
	default:
		break;
	}
}
void Rotate0(Feature feature, Sample &_sample)
{
	Sample sample;
	sample.img = _sample.img.clone();
	int times = (sample.img.rows / MAP_ROWS);//输出图：参数图
	Mat img(sample.img);
		const double angle = 180;
		const double scale = 1;
		Mat ROI, Rotatemat, Rotateimg;
		Point2f center;
	ROI = img(Rect(times*feature.X, times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第一部分
	ROI = img(Rect(times*feature.X, times*(feature.Y + feature.factor), times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第二部分
	static int name = 1;
	imwrite("Rotateimage/model0" + to_string(name) + ".jpg", sample.img);
	name += 1;
}
void Rotate1(Feature feature, Sample &_sample){
	Sample sample;
	sample.img = _sample.img.clone();
	int times = (sample.img.rows / MAP_ROWS);//输出图：参数图
	Mat img(sample.img);
		const double angle = 180;
		const double scale = 1;
		Mat ROI, Rotatemat, Rotateimg;
		Point2f center;
	ROI = img(Rect(times*feature.X, times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第一部分
	ROI = img(Rect(times*(feature.X + feature.factor), times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第二部分
	static int name = 101;
	imwrite("Rotateimage/model1" + to_string(name) + ".jpg", sample.img);
	name += 1;
}
void Rotate2(Feature feature, Sample &_sample){
	Sample sample;
	sample.img = _sample.img.clone();
	int times = (sample.img.rows / MAP_ROWS);//输出图：参数图
	Mat img(sample.img);
		const double angle = 180;
		const double scale = 1;
		Mat ROI, Rotatemat, Rotateimg;
		Point2f center;
	ROI = img(Rect(times*feature.X, times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第一部分
	ROI = img(Rect(times*feature.X, times*(feature.Y + feature.factor), times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第二部分
	ROI = img(Rect(times*feature.X, times*(feature.Y + 2*feature.factor), times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第三部分
	static int name = 201;
	imwrite("Rotateimage/model2" + to_string(name) + ".jpg", sample.img);
	name += 1;
}
void Rotate3(Feature feature, Sample &_sample){
	Sample sample;
	sample.img = _sample.img.clone();
	int times = (sample.img.rows / MAP_ROWS);//输出图：参数图
	Mat img(sample.img);
	const double angle = 180;
	const double scale = 1;
	Mat ROI, Rotatemat, Rotateimg;
	Point2f center;
	ROI = img(Rect(times*feature.X, times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第一部分
	ROI = img(Rect(times*(feature.X + feature.factor), times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第二部分
	ROI = img(Rect(times*(feature.X + 2*feature.factor), times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第三部分
	static int name = 301;
	imwrite("Rotateimage/model3" + to_string(name) + ".jpg", sample.img);
	name += 1;
}
void Rotate4(Feature feature, Sample &_sample){
	Sample sample;
	sample.img = _sample.img.clone();
	int times = (sample.img.rows / MAP_ROWS);//输出图：参数图
	Mat img(sample.img);
	const double angle = 180;
	const double scale = 1;
	Mat ROI, Rotatemat, Rotateimg;
	Point2f center;
	ROI = img(Rect(times*feature.X, times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第一部分
	ROI = img(Rect(times*(feature.X + feature.factor), times*feature.Y, times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第二部分
	ROI = img(Rect(times*(feature.X +feature.factor), times*(feature.Y+feature.factor), times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第三部分
	ROI = img(Rect(times*feature.X, times*(feature.Y + feature.factor), times*feature.factor, times*feature.factor));
	center = Point2f(ROI.cols / 2, ROI.rows / 2);
	Rotatemat = getRotationMatrix2D(center, angle, scale);
	Rotateimg;
	warpAffine(ROI, Rotateimg, Rotatemat, ROI.size());
	Rotateimg.copyTo(ROI);//旋转第四部分
	static int name = 401;
	imwrite("Rotateimage/model4" + to_string(name) + ".jpg", sample.img);
	name += 1;
}
Sample* Compress(Sample *origin) {
	Sample *less = new Sample;
	resize(origin->img, less->img, Size(20, 20));
	return less;
}
void LoadClassifier() {
	ifstream fin(classifierPathName.c_str());
	double sum = 0;
	for (int i = 0; i < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; i++) {
		fin >> weakFeatures[i];
		sum += (weakFactors[i] = log((1 - weakFeatures[i].eRate) / weakFeatures[i].eRate) / 2);
	}
	for (int i = 0; i < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; i++)
		weakFactors[i] = weakFactors[i] / sum;
}
void CalOneSampleIntegralDiagram(Sample* sample) {
	sample->img.convertTo(sample->integralDiagram, CV_32SC1);
	sample->integralDiagram.at<int>(0, 0) = sample->img.at<uchar>(0, 0);
	for (int j = 1; j < MAP_ROWS; j++)//求出第一列的值
		sample->integralDiagram.at<int>(j, 0) =
		sample->integralDiagram.at<int>(j - 1, 0)
		+ sample->img.at<uchar>(j, 0);
	for (int i = 1; i < MAP_ROWS; i++)//求出第一行的值
		sample->integralDiagram.at<int>(0, i) =
		sample->integralDiagram.at<int>(0, i - 1)
		+ sample->img.at<uchar>(0, i);
	for (int j = 1; j < MAP_ROWS; j++) {
		for (int i = 1; i < MAP_COLS; i++) {
			sample->integralDiagram.at<int>(j, i) =
				sample->integralDiagram.at<int>(j, i - 1)
				+ sample->integralDiagram.at<int>(j - 1, i)
				+ sample->img.at<uchar>(j, i)
				- sample->integralDiagram.at<int>(j - 1, i - 1);
		}
	}
}
int CalSampleOneFeatureValue(Sample* sample,Feature& feature) {
	int featureValue;
	switch (feature.model)
	{
	case 0: {
		int  X_Y, X_YF, X_YFF, XF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? sample->integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			X_YFF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + 2 * feature.factor - 1) : 0;
			XF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			featureValue = X_Y + 2 * sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1) + X_YFF - XF_Y
				- 2 * X_YF - sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 2 * feature.factor - 1);
		}
	}break;
	case 1: {
		int  X_Y, X_YF, XF_Y, XFF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? sample->integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			XFF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y) : 0;
			featureValue = X_YF + 2 * XF_Y + sample->integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + feature.factor - 1)
				- X_Y - 2 * sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1) - XFF_Y;
		}
	}break;
	case 2: {
		int  X_Y, X_YF, XF_Y, X_YFF, X_YFFF;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? sample->integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			X_YFF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + 2 * feature.factor - 1) : 0;
			X_YFFF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + 3 * feature.factor - 1) : 0;
			featureValue = 3 * X_YF + XF_Y + X_YFFF + 3 * sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 2 * feature.factor - 1)
				- X_Y - 3 * X_YFF - sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 3 * feature.factor - 1) - 3 * sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1);
		}
	}break;
	case 3: {
		int  X_Y, X_YF, XF_Y, XFF_Y, XFFF_Y;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? sample->integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			XFF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y) : 0;
			XFFF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + 3 * feature.factor - 1, feature.Y) : 0;
			featureValue = 3 * XF_Y + X_YF + XFFF_Y + 3 * sample->integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + feature.factor - 1)
				- X_Y - 3 * XFF_Y - sample->integralDiagram.at<int >(feature.X + 3 * feature.factor - 1, feature.Y + feature.factor - 1) - 3 * sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1);
		}
	}break;
	case 4: {
		int  X_Y, X_YF, XF_Y, XFF_Y, X_YFF;
		for (int i = 0; i < SAMPLE_NUM; i++) {
			X_Y = feature.X + feature.Y ? sample->integralDiagram.at<int >(feature.X, feature.Y) : 0;
			X_YF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + feature.factor - 1) : 0;
			XF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y) : 0;
			XFF_Y = feature.Y ? sample->integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y) : 0;
			X_YFF = feature.X ? sample->integralDiagram.at<int >(feature.X, feature.Y + 2 * feature.factor - 1) : 0;
			featureValue = 2 * XF_Y + 2 * X_YF + 2 * sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + 2 * feature.factor - 1) + 2 * sample->integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + feature.factor - 1)
				- X_Y - X_YFF - XFF_Y - 4 * sample->integralDiagram.at<int >(feature.X + feature.factor - 1, feature.Y + feature.factor - 1) - sample->integralDiagram.at<int >(feature.X + 2 * feature.factor - 1, feature.Y + 2 * feature.factor - 1);
		}
	}break;
	default:
		break;
	}
	return featureValue;
}
void CalSampleAllFeatureValues(Sample* sample) {
	for (int featureNO = 0; featureNO < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; featureNO++)
		sampleFeatureValue[featureNO] = CalSampleOneFeatureValue(sample, weakFeatures[featureNO]);
}
void PredictResult() {
	for (int featureNO = 0; featureNO < MAX_WEAK_CLASSIFIER_NUM_PER_HARD; featureNO++)
		if (predictResult[featureNO] = (sampleFeatureValue[featureNO] * weakFeatures[featureNO].p >= weakFeatures[featureNO].threshold*weakFeatures[featureNO].p))
			P += weakFactors[featureNO];
}
#endif // USE


