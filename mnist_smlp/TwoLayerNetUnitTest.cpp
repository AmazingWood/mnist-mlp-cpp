#include <iostream>
#include "TrainEntry.h"

int mainPredictTest() {
	TwoLayerNet a(2, 2, 2);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> testData(5, 2);
	testData << 1, 2,
		3, 4,
		5, 6,
		7, 8,
		9, 0;
	auto b = a.predict(testData);
	std::cout << b << std::endl;
	//target data
	/*
		[[0.5004938  0.4995062 ]
		 [0.50049404 0.49950596]
		 [0.50049428 0.49950572]
		 [0.50049451 0.49950549]
		 [0.5005016  0.4994984 ]]
	*/
	return 0;
}
//loss test
int mainLossTest() {
	TwoLayerNet a(2, 2, 2);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> testData(5, 2),testResult(5,2);
	testData << 0.56f, 0.33f,
			0.12f, 0.14f,
			0.99f, 0.1f,
			0.23f, 0.97f,
			0.10f, 0.0f;
	testResult << 0, 1,
				1, 0,
				1, 0,
				0, 1,
				0, 1;
	auto b = a.lossFunc(testData, testResult);
	//0.6922956620919839
	std::cout << b << std::endl;
	return 0;
}
//accuracyFunc test
int mainAccuracyTest() {
	TwoLayerNet a(2, 2, 2);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> testData(5, 2), testResult(5, 2);
	testData << 0.56f, 0.33f,
		0.12f, 0.14f,
		0.99f, 0.1f,
		0.23f, 0.97f,
		0.10f, 0.0f;
	testResult << 0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1;
	auto b = a.accuracyFunc(testData, testResult);
	//0.4 or 0.6
	std::cout << b << std::endl;
	return 0;
}
//GradientFunc test
int mainGradientFuncTest() {
	TwoLayerNet a(2, 2, 2);
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> testData(5, 2), testResult(5, 2);
	testData << 0.56f, 0.33f,
		0.12f, 0.14f,
		0.99f, 0.1f,
		0.23f, 0.97f,
		0.10f, 0.0f;
	testResult << 0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1;
	auto [w1,b1,w2,b2] = a.gradientFunc(testData, testResult);
	/*
	w2
	[[ 0.0509037  -0.0509037 ]
	[ 0.05100288 -0.05100288]]

	b2
	[ 0.1001147 -0.1001147] or [ 0.0984075 -0.0984075]

	w1 is random, just like this kind of shit is fine.
	[[-2.67886256e-05  3.75430029e-06]
	 [ 1.30525463e-04 -1.82944097e-05]]

	b1 is random, just like this kind of shit is fine.
	[ 0.00031724 -0.00041602]
	*/
	std::cout << w1 << std::endl << std::endl;
	std::cout << b1 << std::endl << std::endl;
	std::cout << w2 << std::endl << std::endl;
	std::cout << b2 << std::endl << std::endl;
	return 0;
}