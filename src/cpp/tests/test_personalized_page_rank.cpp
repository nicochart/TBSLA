#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <vector>
#include <iostream>
#include <tbsla/cpp/utils/InputParser.hpp>

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  int n = 4; 
  double epsilon = 0.0000001; 
  double beta = 0.8; 
  int max_iterations = 100; 

    
  std::vector<double> values{1.0, 0.5, 0.5, 1.0, 1.0};
  std::vector<int> columns{1, 0, 0, 3, 2};
  std::vector<int> rows{0, 1, 2, 4, 5}; 
  std::vector<double> solution{ 0.294, 0.118, 0.327, 0.261};
  tbsla::cpp::Matrix * A = new tbsla::cpp::MatrixCSR(n, n, values, rows, columns);
  A->print_as_dense(std::cout); 

  std::vector<double> b(n); 
  std::vector<int> personalized_nodes = {0};

  b = A->personalized_page_rank(epsilon, beta, max_iterations, personalized_nodes);

  double error = 0.0;
  for(int i = 0 ; i < n; i++) {
    error += std::abs(b[i] - solution[i]);
  }
  if(error > 0.1){
    std::cout << "Test failed" << error << std::endl;
    return 1; 
  }
  std::cout << "Test OK" << std::endl;

  return 0; 

}