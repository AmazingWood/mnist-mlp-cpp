#include "TwoLayerNet.h"

TwoLayerNet::TwoLayerNet(int inputSize, int hiddenSize, int outputSize, float weightInitStd)
	:inputSize(inputSize),
	hiddenSize(hiddenSize),
	outputSize(outputSize),
	weightInitStd(weightInitStd)
{
	b1 = Eigen::RowVectorXf::Zero(this->hiddenSize);
	b2 = Eigen::RowVectorXf::Zero(this->outputSize);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::normal_distribution<> dis(-1, 1);
	w1 = weightInitStd * MatrixXFDynamic::Zero(inputSize, hiddenSize).unaryExpr([&](float dummy) {return dis(gen); });
	w2 = weightInitStd * MatrixXFDynamic::Zero(hiddenSize, outputSize).unaryExpr([&](float dummy) {return dis(gen); });
}

const MatrixXFDynamic TwoLayerNet::predict(MatrixXFDynamic const &x)
{
	auto a1 = (x * w1).rowwise() + b1;
	auto z1 = MFUNC::sigmoidFunc(a1);
	auto a2 = (z1 * w2).rowwise() + b2;
	auto y = MFUNC::softmaxFunc(a2);
	return y;
}

const float TwoLayerNet::lossFunc(MatrixXFDynamic const &x, MatrixXFDynamic const &t) {
	auto y = predict(x);
	return MFUNC::crossEntropyError(y, t);
}

const float TwoLayerNet::accuracyFunc(MatrixXFDynamic const &x, MatrixXFDynamic const &t)
{
	auto y = predict(x);
	auto yArgMax = MFUNC::argMax(MFUNC::ROW,y);
	auto tArgMax = MFUNC::argMax(MFUNC::ROW,t);
	auto accuracy = (tArgMax.array() == yArgMax.array()).cast<float>().sum() / tArgMax.size();
	return accuracy;
}

const std::tuple<MatrixXFDynamic, Eigen::RowVectorXf, MatrixXFDynamic, Eigen::RowVectorXf> TwoLayerNet::gradientFunc(MatrixXFDynamic const &x, MatrixXFDynamic const &t)
{
	int batchNum = static_cast<int>(x.rows());

	//forward
	auto a1 = (x * w1).rowwise() + b1;
	auto z1 = MFUNC::sigmoidFunc(a1);
	auto a2 = (z1 * w2).rowwise() + b2;
	auto y = MFUNC::softmaxFunc(a2);

	//backward
	auto dY = (y - t).array() / batchNum;
	z1.transposeInPlace();

	MatrixXFDynamic w2Grads = z1 * dY.matrix();
	Eigen::RowVectorXf b2Grads = dY.colwise().sum();

	auto da1 = dY.matrix() * w2.transpose();
	//here,u cannot use auto instead of matrix, because the fuckin gcc will change dz1's type and make ur fuckin martix to zero!
	MatrixXFDynamic dz1 = MFUNC::sigmoidFuncGrad(a1).array() * da1.array();
	MatrixXFDynamic w1Grads = x.transpose() * dz1;
	Eigen::RowVectorXf b1Grads = dz1.colwise().sum();
	
	return std::make_tuple(w1Grads, b1Grads, w2Grads, b2Grads);
}

void TwoLayerNet::updateNetParams(float const &learningRate, MatrixXFDynamic const &w1G, MatrixXFDynamic const &w2G, Eigen::RowVectorXf const &b1G, Eigen::RowVectorXf const &b2G)
{
	this->w1.array() -= (learningRate * w1G.array());
	this->w2.array() -= (learningRate * w2G.array());
	this->b1.array() -= (learningRate * b1G.array());
	this->b2.array() -= (learningRate * b2G.array());

}
