#pragma once
#include "Header.h"
#include "MatrixFunctions.h"

using MatrixXFDynamic = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class TwoLayerNet {
public:
	TwoLayerNet(int inputSize, int hiddenSize, int outputSize, float weightInitStd = 0.01);
	//x shape :batchsize*inputsize
	const MatrixXFDynamic predict(MatrixXFDynamic const &x);
	const float lossFunc(MatrixXFDynamic const &x, MatrixXFDynamic const &t);
	const float accuracyFunc(MatrixXFDynamic const &x, MatrixXFDynamic const &t);
	const std::tuple<MatrixXFDynamic, Eigen::RowVectorXf, MatrixXFDynamic, Eigen::RowVectorXf> gradientFunc(MatrixXFDynamic const &x, MatrixXFDynamic const &t);
	void updateNetParams(float const &learningRate, MatrixXFDynamic const &w1G, MatrixXFDynamic const &w2G, Eigen::RowVectorXf const &b1G, Eigen::RowVectorXf const &b2G);
private:
	float weightInitStd;
	int inputSize, hiddenSize, outputSize;
	MatrixXFDynamic w1, w2;
	Eigen::RowVectorXf b1, b2;
};