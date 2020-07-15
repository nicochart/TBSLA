#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/mm.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <vector>
#include <iostream>
#include <tbsla/cpp/utils/InputParser.hpp>

int main(int argc, char** argv) {
  	InputParser input(argc, argv);
  	int n = 4;
	double epsilon = 0.00001;
	double beta = 1;
  	int max_iterations = 2;

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
	
	std::vector<double> b(n);
	b = m.page_rank(epsilon, beta, max_iterations);
	
	std::cout << 	"SOLUTION [";
	for(int  i = 0; i < n; i++){
		std::cout << b[i] << ", " ;
	} 
	std::cout << "]"<< std::endl;
}