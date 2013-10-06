#include <iostream>
#include <fstream>
#include <algorithm>
#include "lbl/vecoffset.h"

using namespace oxlm;
using namespace std;

namespace mine {
  //less-than function. But want descending order so using >
  bool cmp(const std::pair<int, Real>& a, const std::pair<int,Real>& b) {
    return a.second > b.second;
  }
}

void oxlm::dotask(const std::string& taskfile, MatrixReal R, const Dict& dict, bool verbose) {
  R.rowwise().normalize();
  ifstream in(taskfile.c_str());
  assert(in.good() && "Can't open task file");
  int count=0, ranks=0, matches=0;
  string line,token;
  while (getline(in, line)) {
    string sa, sb, sc, sd;
    stringstream line_stream(line);
    line_stream >> sa >> sb >> sc >> sd;
    WordId a=dict.Lookup(sa); if (a<0 || a > R.rows()) continue;
    WordId b=dict.Lookup(sb); if (b<0 || b > R.rows()) continue;
    WordId c=dict.Lookup(sc); if (c<0 || c > R.rows()) continue;
    WordId d=dict.Lookup(sd); if (d<0 || d > R.rows()) continue;

    ++count;
    WordId ans; int pos;
    task(R, dict, a, b, c, d, ans, pos, verbose);
    ranks += pos;
    matches += (ans==d);
  }
  cout << "Accuracy: " << (float)matches/((float)count+1e-12) << endl;
  cout << "Average rank: " << (float)ranks/((float)count+1e-12) << endl;
}

void oxlm::task(const MatrixReal& R, const Dict& dict, WordId a, WordId b, WordId c, WordId d, WordId& ans, int& pos, bool verbose) {
  //cout << "R is " << R.rows() << "x" << R.cols() << endl;
  //cout << "a, b, c are " << a << ", " << b << ", " << c << endl;
  assert(a < R.rows());
  assert(b < R.rows());
  assert(c < R.rows());
  VectorReal xa=R.row(a);
  VectorReal xb=R.row(b);
  VectorReal xc=R.row(c);

  VectorReal y = xb - xa + xc;

  y.normalize();
  
  //cout << "dots.. y is " << y.rows() << "x" << y.cols() << endl;
  MatrixReal W = R * y;//.transpose();
  //cout << "product OK " << endl;
  
  vector<pair<int, Real> > v;
  v.resize(W.rows());
  for (int i=0;i<W.rows(); ++i)
    v[i] = make_pair(i, W(i,0));

  //cout << "sorting.." << endl;
  sort(v.begin(), v.end(), mine::cmp);

  ans = v[1].first;

  //cout << "searching.." << endl;
  for (int i=0;i<W.rows(); ++i)
    if (v[i].first == d) {
      pos = (int)i;
      break;
  }

  //cout << "done.." << endl;
  if (verbose) {
    cout << dict.Convert(a) << " "
         << dict.Convert(b) << " "
         << dict.Convert(c) << " "
         << dict.Convert(d) << "\t";
    for (int i=0;i<min((int)v.size(), 10); ++i) 
      cout << dict.Convert(v[i].first) << " " << v[i].second << " ";
    cout << endl;
  }

}

