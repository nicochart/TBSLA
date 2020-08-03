#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/mm.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <vector>
#include <iostream>
#include <tbsla/cpp/utils/InputParser.hpp>

#include <algorithm>
#include <chrono>
#include <random>
#include <map>

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
}

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  int n = 4;
  double epsilon = 0.00001;
  double beta = 1;
  int max_iterations = 100;

  if(input.has_opt("--beta")) {
    std::string beta_string = input.get_opt("--beta", "1");
    beta = std::stod(beta_string);
  }
  if(input.has_opt("--epsilon")) {
    std::string epsilon_string = input.get_opt("--epsilon", "1");
    epsilon = std::stod(epsilon_string);
  }

  if(input.has_opt("--max-iterations")) { 
    std::string max_iterations_string = input.get_opt("--max-iterations", "1");
    max_iterations = std::stoi(max_iterations_string);
  }
  
  std::string matrix_input = input.get_opt("--matrix_input");
  if(matrix_input == "") {
    std::cerr << "A matrix file has to be given with the parameter --matrix_input file" << std::endl;
    exit(1);
  }
  auto t_read_start = now();
  tbsla::cpp::MatrixCOO m = tbsla::utils::io::readMM(matrix_input);
  auto t_read_end = now();
  
  std::vector<double> b(n);
  int nb_iterations_done;

  auto t_page_rank_start = now();
  b = m.page_rank(epsilon, beta, max_iterations, nb_iterations_done);
  auto t_page_rank_end = now();

  std::cout <<  "SOLUTION [";
  for(int  i = 0; i < n; i++){
    std::cout << b[i] << ", " ;
  } 
  std::cout << "]"<< std::endl;

  auto t_app_end = now();

  std::map<std::string, std::string> outmap;
  outmap["test"] = "page_rank";
  outmap["matrix_format"] = "COO";
  outmap["n"] = std::to_string(m.get_n_col());
  outmap["nnz"] = std::to_string(m.get_nnz());
  outmap["nb_iterations"] = std::to_string(nb_iterations_done);
  outmap["time_read_m"] = std::to_string((t_read_end - t_read_start) / 1e9);
  outmap["time_page_rank"] = std::to_string((t_page_rank_end - t_page_rank_start) / 1e9);
  outmap["time_app"] = std::to_string((t_app_end - t_read_start) / 1e9);
  outmap["lang"] = "C++";
  outmap["matrix_input"] = std::string(argv[1]);

  std::map<std::string, std::string>::iterator it=outmap.begin();
  std::cout << "{\"" << it->first << "\":\"" << it->second << "\"";
  it++;
  for (; it != outmap.end(); ++it) {
    std::cout << ",\"" << it->first << "\":\"" << it->second << "\"";
  }
  std::cout << "}\n";  
}