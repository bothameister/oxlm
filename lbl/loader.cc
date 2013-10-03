///// Loads models trained with train_addfactored_sgd and does things with them -JAB
//
// STL
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iterator>
#include <iterator>
#include <cstring>
#include <functional>
#include <omp.h>
#include <time.h>
#include <math.h>
#include <float.h>
#include <set>

// Boost
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/shared_ptr.hpp>

// Eigen
#include <Eigen/Dense>
#include <Eigen/Core>

// Local
#include "lbl/nlm.h"
#include "lbl/log_add.h"
#include "corpus/corpus.h"


// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;


typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;

void eval_ppl(FactoredOutputNLM& model, const string& test_set);
Real perplexity(const FactoredOutputNLM& model, const Corpus& test_corpus, int stride=1);

int main(int argc, char **argv) {

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm; 

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
//    ("config,c", value<string>(), 
//        "config file specifying additional command line options")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("test-set", value<string>(), 
        "corpus of test sentences to be evaluated at each iteration")
    ("model,m", value<string>(), 
        "model to load")
    ("embeddings,e", value<string>(), 
        "file to write embeddings (R) to. Line format wordtype followed by values. If model has additive words, the factor vectors are output to the e-file suffixed .factors")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ;
  options_description config_options, cmdline_options;
//  config_options.add(generic);
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm); 
 // if (vm.count("config") > 0) {
 //   ifstream config(vm["config"].as<string>().c_str());
 //   store(parse_config_file(config, cmdline_options), vm); 
 // }
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  if (vm.count("help")) { 
    cout << cmdline_options << "\n"; 
    return 1; 
  }
  omp_set_num_threads(vm["threads"].as<int>());

  assert(vm.count("model") && "Must specify model to load");
  cerr << "# model = " << vm["model"].as<string>() << endl;

  boost::shared_ptr<FactoredOutputNLM> model = AdditiveFactoredOutputNLM::load_from_file(vm["model"].as<string>());

  if (vm.count("test-set"))
      eval_ppl(*model, vm["test-set"].as<string>());

  if (vm.count("embeddings")) {
    bool ok=model->write_embeddings(vm["embeddings"].as<string>());
    assert(ok && "Failed to write embeddings to file");
  }

  return 0;
}

void eval_ppl(FactoredOutputNLM& model, const string& test_set) {
  clock_t timer=clock();
  cout << "Evaluating ppl on " << test_set << endl;
  Corpus test_corpus;

  Dict& dict = model.label_set();

  ifstream test_in(test_set.c_str());
  WordId end_id = dict.Lookup("</s>");
  WordId unk = dict.Lookup("<unk>");
  assert(dict.valid(end_id));
  string line,token;
  while (getline(test_in, line)) {
    stringstream line_stream(line);
    Sentence tokens;
    while (line_stream >> token) {
      WordId w = dict.Convert(token, true);
      if (w < 0) {
        if (dict.valid(unk)) 
          w = unk;
        else {
          cerr << token << " " << w << endl;
          assert(!"Unknown word found in test corpus and training corpus does not include <unk>");
        }
      }
      test_corpus.push_back(w);
    }
    test_corpus.push_back(end_id);
  }
  test_in.close();

  Real pp = perplexity(model, test_corpus, 1);
  pp = exp(-pp/test_corpus.size());
  Real elapsed = (clock()-timer) / (Real)CLOCKS_PER_SEC;
  cerr << " | " << elapsed << " seconds | Test Perplexity = " << pp << endl;
}



Real perplexity(const FactoredOutputNLM& model, const Corpus& test_corpus, int stride) {
  Real p=0.0;

  int context_width = model.config.ngram_order-1;
  int tokens=0;
  WordId start_id = model.label_set().Lookup("<s>");
  WordId end_id = model.label_set().Lookup("</s>");

  #pragma omp master
  cerr << "Calculating perplexity for " << test_corpus.size()/stride << " tokens";

  std::vector<WordId> context(context_width);

  size_t thread_num = omp_get_thread_num();
  size_t num_threads = omp_get_num_threads();
  for (size_t s = (thread_num*stride); s < test_corpus.size(); s += (num_threads*stride)) {
    WordId w = test_corpus.at(s);
    int context_start = s - context_width;
    bool sentence_start = (s==0);

    for (int i=context_width-1; i>=0; --i) {
      int j=context_start+i;
      sentence_start = (sentence_start || j<0 || test_corpus.at(j) == end_id);
      int v_i = (sentence_start ? start_id : test_corpus.at(j));

      context.at(i) = v_i;
    }
    Real log_prob = model.log_prob(w, context, false);
    p += log_prob;

    #pragma omp master
    if (tokens % 1000 == 0) { cerr << "."; cerr.flush(); }

    tokens++;
  }
  #pragma omp master
  cerr << endl;

  return p;
}
