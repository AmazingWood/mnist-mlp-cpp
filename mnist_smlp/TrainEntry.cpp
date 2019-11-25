#include "TrainEntry.h"
#include "TimeStamp.h"

void TrainEntry::entryFunc()
{
	auto minstPath{ fs::path("D://projects//mnist_smlp//dataset//") };
	auto [trainset,testset] = RLDNN::loadMnist(minstPath);

	auto netWork{ TwoLayerNet(784, 50, 10) };

	auto itersNum{ int(10000) }; //loop times
	auto trainSize{ int(static_cast<int>(trainset.first.rows())) };
	auto batchSize{ int(100) };
	auto learningRate{ float(0.1) };

	auto iterPerEpoch{ float(std::max(static_cast<float>(trainSize / batchSize),1.0f)) };

	MFUNC::RandomChoice randomChoice;

	//timestamp
	float time{ 0.f };
	RLVulkan::TimeStamp ts{};
	//end timestamp
	for (auto i{ int(0) }; i < itersNum; i++) {
		ts.setStart();
		auto xBatch{ Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(batchSize, 784) };
		auto tBatch{ Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>(batchSize, 10) };

		std::set<size_t> randSet(randomChoice(trainSize, batchSize));
		std::vector<int> indecs(randSet.begin(), randSet.end());
		for (auto j{ int(0) }; j < batchSize; j++) {
			xBatch.row(j) = trainset.first.row(indecs[j]);
			tBatch.row(j) = trainset.second.row(indecs[j]);
		}
		auto [w1, b1, w2, b2] = netWork.gradientFunc(xBatch, tBatch);
		netWork.updateNetParams(learningRate, w1, w2, b1, b2);
		ts.setEnd();
		time += ts.getElapsedTime<std::chrono::milliseconds>();
		if (i%static_cast<int>(iterPerEpoch) == 0) {
			auto trainLoss = netWork.lossFunc(xBatch, tBatch);
			auto trainAcc = netWork.accuracyFunc(trainset.first, trainset.second);
			auto testAcc = netWork.accuracyFunc(testset.first, testset.second);
			std::cout << "LOSS | " << trainLoss << std::endl;
			std::cout << "train acc, test acc | " << std::fixed << std::setprecision(6) << trainAcc << ",\t" << testAcc << std::endl;
		}
	}
	std::cout << std::endl << "Average time per iteration:" << time / itersNum << std::endl;
}
