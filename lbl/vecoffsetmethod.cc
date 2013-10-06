
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "lbl/vecoffset.h"
#include "lbl/typedefs.h"


using namespace boost;
using namespace boost::program_options;
using namespace std;
using namespace oxlm;

int main(int argc, char **argv) {

  ///////////////////////////////////////////////////////////////////////////////////////
  // Command line processing
  variables_map vm; 

  // Command line processing
  options_description cmdline_specific("Command line specific options");
  cmdline_specific.add_options()
    ("help,h", "print help message")
    ;
  options_description generic("Allowed options");
  generic.add_options()
    ("embeddings,e", value<string>(), 
        "file to write embeddings (R) to. Line format wordtype followed by values. If model has additive words, the factor vectors are output to the e-file suffixed .factors")
    ("analogy-task,a", value<string>(), 
        "Use vectors from model to do the analogy-task given in this file. Each line should be 'a b c d', posing the question 'a is to b as c is to ?'")
    ("verbose,v", "echo task data and top 10 nearest answers")
    ;
  options_description config_options, cmdline_options;
  cmdline_options.add(generic).add(cmdline_specific);

  store(parse_command_line(argc, argv, cmdline_options), vm); 
  notify(vm);
  ///////////////////////////////////////////////////////////////////////////////////////

  cerr << "Vector offset-method a la Mikolov & Zweig" << endl;
  if (vm.count("help")) { 
    cout << cmdline_options << "\n"; 
    return 1; 
  }

  vector<string> stash;
  stash.reserve(100000);
  int d=0;
  {
    ifstream in(vm["embeddings"].as<string>());
    assert(in.good());
    int i=0;
    string line,token;
    while (getline(in, line)) {
      if (i++ == 0) {
        stringstream line_stream(line);
        line_stream >> token; //first token is word
        while (line_stream >> token)
          ++d;
      }
      stash.push_back(line);
    }
  } 
//  cout << "Inferred dimensions " << stash.size() << "x" << d << endl;
  assert (d>0);
  MatrixReal R = MatrixReal::Zero(stash.size(), d);
  Dict dict;
  string token;
  Real val;
  for (int i=0; i<R.rows(); ++i) {
    stringstream line_stream(stash.at(i));
    line_stream >> token; //first token is word
    WordId w=dict.Convert(token);
    assert(w==i);
    int j=0;
    while (line_stream >> val && j < d) {
      R(i,j) = val;
      ++j;
    }
  } 
  assert(vm.count("analogy-task")); 
  dotask(vm["analogy-task"].as<string>(), R, dict, vm.count("verbose"));
}


