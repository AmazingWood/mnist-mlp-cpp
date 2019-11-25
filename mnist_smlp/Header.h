#pragma once
//#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include <Eigen/Core>
#include <boost/endian/detail/endian_reverse.hpp>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <random>
#include <set>
//eigen
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/Geometry> 
namespace fs = std::filesystem;