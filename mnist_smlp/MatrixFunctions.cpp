#include "MatrixFunctions.h"

namespace MFUNC {

	MiddleWareData MFUNC::sigmoidFunc(MiddleWareData const &x)
	{
		return  1 / (1 + (0 - x.array()).exp());
	}

	MiddleWareData sigmoidFuncGrad(MiddleWareData const &x)
	{
		auto sigmoidFuncVal = sigmoidFunc(x).array();
		return  (1 - sigmoidFuncVal)*sigmoidFuncVal;
	}

	MiddleWareData softmaxFunc(MiddleWareData const &x)
	{
		if (x.rows() > 1) {
			MiddleWareData xTransposed = x.transpose().rowwise() - x.transpose().colwise().maxCoeff();
			MiddleWareData y = (xTransposed.array().exp().rowwise() / xTransposed.array().exp().colwise().sum()).transpose();
			return y;
		}
		else if (x.rows() == 1) {
			auto y = x.array() - x.maxCoeff();
			return y.exp() / y.array().exp().sum();
		}
		throw std::out_of_range("Unknown Params Dimension!");
	}

	float crossEntropyError(MiddleWareData const &y, MiddleWareData const &t)
	{
		if (t.size() == y.size()) {
			auto tArgMax = argMax(ROW, t);
			auto batchSize{ int(tArgMax.cols()) };
			auto targetArr{ Eigen::RowVectorXf(batchSize) };
			for (auto i{ int(0) }; i < batchSize; i++) {
				targetArr(i) = y(i, static_cast<int>(tArgMax(0, i)));
			}
			targetArr = (targetArr.array() + 1E-7f).log();
			auto targetNum = -targetArr.sum()/batchSize;
			return targetNum;
		} 
		else {
			auto batchSize{ int(t.cols()) };
			auto targetArr{ Eigen::RowVectorXf(batchSize) };
			for (auto i{ int(0) }; i < batchSize; i++) {
				targetArr(i) = y(i, static_cast<int>(t(0, i)));
			}
			targetArr = (targetArr.array() + 1E-7f).log();
			auto targetNum = -targetArr.sum() / batchSize;
			return targetNum;
		}
		throw std::invalid_argument("Wrong resultData£¡£¡");
	}

	Eigen::RowVectorXi MFUNC::argMax(AXIS const &a, MiddleWareData const &x)
	{
		if (a == ROW) {
			auto maxIndexsRows{ Eigen::RowVectorXi(x.rows()) } ;
			for (auto i{ int(0) }; i < x.rows(); i++) {
				x.row(i).maxCoeff(&maxIndexsRows(i));
			}
			return maxIndexsRows;
		}
		else if (a == COL) {
			auto maxIndexsCols{ Eigen::RowVectorXi(x.cols()) };
			for (auto i{ int(0) }; i < x.cols(); i++) {
				x.col(i).maxCoeff(&maxIndexsCols(i));
			}
			return maxIndexsCols;
		}
		else {
			throw std::invalid_argument("Unknown Params Axis!");
		}
	}

	RandomChoice::RandomChoice() : rd(), randGen(rd()), idcesSet() {}

	std::set<size_t> RandomChoice::operator()(size_t maxSize, size_t choiceNum) {
		std::uniform_int_distribution<> intDist(0, maxSize - 1);
		do {
			idcesSet.insert(intDist(randGen));
		} while (idcesSet.size() != choiceNum);
		auto randSet = idcesSet;
		this->idcesSet.clear();
		return randSet;
	}
}