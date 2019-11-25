#pragma once
#include "load_mnist.h"

namespace MFUNC {
	using MiddleWareData =
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

	enum AXIS {
		ROW,
		COL
	};

	//Sigmoid Func - unit test complete
	MiddleWareData sigmoidFunc(MiddleWareData const &x);

	//Sigmoid Grad Func - unit test complete
	MiddleWareData sigmoidFuncGrad(MiddleWareData const &x);

	//Softmax Func - unit test complete
	MiddleWareData softmaxFunc(MiddleWareData const &x);

	//Cross Entropy Error - unit test complete
	float crossEntropyError(MiddleWareData const &y, MiddleWareData const &t);

	//Argmax - unit test complete
	Eigen::RowVectorXi argMax(AXIS const &a, MiddleWareData const &x);

	class RandomChoice {
		std::random_device rd;
		std::set<size_t> idcesSet;
		std::mt19937 randGen;

	public:
		RandomChoice();
		std::set<size_t> operator()(size_t maxSize, size_t choiceNum);
	};
}