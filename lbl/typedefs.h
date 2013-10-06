#ifndef _TYPEDEFS_H_
#define _TYPEDEFS_H_

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "corpus/corpus.h"
#include <vector>

namespace oxlm {

typedef float Real;
typedef std::vector<Real> Reals;
typedef Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> MatrixReal;
typedef Eigen::Matrix<Real, Eigen::Dynamic, 1>              VectorReal;
typedef Eigen::Array<Real, Eigen::Dynamic, 1>               ArrayReal;

typedef Eigen::SparseMatrix<Real> SparseMatrixInt;
typedef std::vector<WordId> WordIds;
typedef std::vector<WordIds> WordIdMap;

}

#endif
