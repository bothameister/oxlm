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
#include "lbl/vecoffset.h"

// Namespaces
using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;


typedef vector<WordId> Sentence;
typedef vector<WordId> Corpus;

void eval_ppl(FactoredOutputNLM& model, const string& test_set);
void eval_ppl_ngrams(FactoredOutputNLM& model, const string& test_set, bool subtract_surface=false);
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
    ("test-set-ngrams", value<string>(), 
        "list of ngrams to score - no sentence padding applied")
    ("model,m", value<string>(), 
        "model to load")
    ("embeddings,e", value<string>(), 
        "file to write embeddings (R) to. Line format wordtype followed by values. If model has additive words, the factor vectors are output to the file suffixed .factors")
    ("ctx-embeddings,E", value<string>(), 
        "file to write embeddings (Q) to. Line format wordtype followed by values. If model has additive words, the factor vectors are output to the file suffixed .factors")
    ("analogy-task,a", value<string>(), 
        "Use vectors from model to do the analogy-task given in this file. Each line should be 'a b c d', posing the question 'a is to b as c is to ?'")
    ("threads", value<int>()->default_value(1), 
        "number of worker threads.")
    ("subtract-surface", "Subtract the surface factor vector from word vector of output words.")
    ("verbose,v", "echo task data and top 10 nearest answers")
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

  if (vm.count("subtract-surface")) 
    cerr << "Subtracting surface factors from output representation before computing probability" << endl;
  

  if (vm.count("test-set"))
      eval_ppl(*model, vm["test-set"].as<string>());
  if (vm.count("test-set-ngrams"))
      eval_ppl_ngrams(*model, vm["test-set-ngrams"].as<string>(), vm.count("subtract-surface"));

  if (vm.count("embeddings")) {
    bool ok=model->write_embeddings(vm["embeddings"].as<string>());
    assert(ok && "Failed to write output embeddings to file");
  }
  if (vm.count("ctx-embeddings")) {
    bool ok=model->write_embeddings(vm["ctx-embeddings"].as<string>(), false);
    assert(ok && "Failed to write context embeddings to file");
  }

  if (vm.count("analogy-task")) {
    //MatrixReal R = dynamic_cast<AdditiveFactoredOutputNLM&>(*model).Rp;
    dotask(vm["analogy-task"].as<string>(), dynamic_cast<AdditiveFactoredOutputNLM&>(*model).Rp, model->label_set(), vm.count("verbose"));
  }

  return 0;
}

void eval_ppl_ngrams(FactoredOutputNLM& model, const string& test_set, bool subtract_surface) {
  clock_t timer=clock();
  Dict& dict = model.label_set();
  Corpus test_event;
  ifstream test_in(test_set.c_str());
  string line,token;
  Real p=0.0;
  int tokcount=0;
  WordId unk = dict.Lookup("<unk>");
  bool use_cache=false;
  AdditiveFactoredOutputNLM& model_iface = dynamic_cast<AdditiveFactoredOutputNLM&>(model);
  while (getline(test_in, line)) {
    test_event.clear();
    stringstream line_stream(line);
    int i=0;
    while (i++ < model.config.ngram_order && line_stream >> token) {
      WordId w = dict.Convert(token, true);
      if (w < 0) {
        if (dict.valid(unk)) 
          w = unk;
        else {
          cerr << token << " " << w << endl;
          assert(!"Unknown word found in test data and training corpus does not include <unk>");
        }
      }
      test_event.push_back(w);
    }
    ++tokcount;

    assert((int)test_event.size() == model.config.ngram_order && "This scoring mode only handles full ngrams");
    WordId w = test_event.back();
    test_event.pop_back();

    if (subtract_surface) model_iface.toggle_surface_factor(w, false);

    Real prob=model.log_prob(w, test_event, use_cache);
    p += prob;
    cout << line << " " << prob << endl;

    if (subtract_surface) model_iface.toggle_surface_factor(w, true);
  }
  test_in.close();
  p = exp(-p/tokcount);
  Real elapsed = (clock()-timer) / (Real)CLOCKS_PER_SEC;
  cerr << " | " << elapsed << " seconds | Test Perplexity = " << p << endl;

}
void eval_ppl(FactoredOutputNLM& model, const string& test_set) {
  clock_t timer=clock();
  cerr << "Evaluating ppl on " << test_set << endl;
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
