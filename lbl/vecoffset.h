#ifndef _VECOFFSET_H
#define _VECOFFSET_H

#include <string>
#include "lbl/typedefs.h"
#include "corpus/corpus.h"

namespace oxlm {

  void task(const MatrixReal& R, const Dict& dict, WordId a, WordId b, WordId c, WordId d, WordId& ans, int& pos, bool verbose=false);
  void dotask(const std::string& taskfile, MatrixReal R, const Dict& dict, bool verbose=false);

}

#endif
