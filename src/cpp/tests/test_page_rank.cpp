#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <vector>
#include <iostream>
#include <tbsla/cpp/utils/InputParser.hpp>

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
  	
  	
	std::vector<double> values{11.0, 12.0, 2.0, 6.0, 8.0,3.0, 13.0, 14.0, 16.0};
	std::vector<int> columns{0, 1, 3, 1, 3, 3, 0, 2, 3};
	std::vector<int> rows{0, 3, 5, 6, 9}; 

	tbsla::cpp::Matrix * A = new tbsla::cpp::MatrixCSR(n, n, values, rows, columns);
	A->print_as_dense(std::cout); 
	
	std::vector<double> b(n); 
	b = A->page_rank(epsilon, beta, max_iterations);
	
	std::cout << 	"SOLUTION ["; 
	for(int  i = 0; i < n; i++){
		std::cout << b[i] << ", " ; 
	} 
	std::cout << "]"<< std::endl ; 
}