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

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  int n = 4;
	double epsilon = 0.00001;
	double beta = 1;
 	int max_iterations = 100;
  int size_personalized_nodes = 3;

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
  tbsla::cpp::MatrixCOO m = tbsla::utils::io::readMM(matrix_input);

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_real_distribution<double> dist {-1, 1};
  auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };

	std::vector<int> personalized_nodes(size_personalized_nodes);
  std::generate(personalized_nodes.begin(), personalized_nodes.end(), gen);

  std::vector<double> b(n);
  b = m.personalized_page_rank(epsilon, beta, max_iterations, personalized_nodes);
	
  std::cout << 	"SOLUTION [";
  for(int  i = 0; i < n; i++){
    std::cout << b[i] << ", " ;
  } 
  std::cout << "]"<< std::endl;

}