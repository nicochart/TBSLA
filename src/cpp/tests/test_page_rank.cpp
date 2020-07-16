#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <vector>
#include <iostream>
#include <tbsla/cpp/utils/InputParser.hpp>

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  int n = 4; 
  double epsilon = 0.0000001; 
  double beta = 1; 
  int max_iterations = 100; 

  	
  std::vector<double> values{11.0, 12.0, 2.0, 6.0, 8.0,3.0, 13.0, 14.0, 16.0};
  std::vector<int> columns{0, 1, 3, 1, 3, 3, 0, 2, 3};
  std::vector<int> rows{0, 3, 5, 6, 9}; 
  std::vector<double> solution{0.5246, 0.4342, 0.1231, 1.0};
  tbsla::cpp::Matrix * A = new tbsla::cpp::MatrixCSR(n, n, values, rows, columns);
  A->print_as_dense(std::cout); 

	std::vector<double> b(n); 
	b = A->page_rank(epsilon, beta, max_iterations);

  double error = 0.0;
  for(int i = 0 ; i < n; i++) {
    error += std::abs(b[i] - solution[i]);
  }
  if(error > 0.01){
    std::cout << "Test failed" << std::endl;
    return 1; 
  }
  std::cout << "Test OK" << std::endl;
  return 0; 
}