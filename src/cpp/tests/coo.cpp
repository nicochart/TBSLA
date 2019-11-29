#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {

  std::vector<double> values{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  std::vector<int> row{ 0, 0, 0, 1, 2, 3, 3, 4, 4, 4 };
  std::vector<int> col{ 0, 1, 4, 2, 1, 1, 3, 0, 2, 4 };

  std::vector<double> x{ 1, 2, 3, 4, 5 };

  tbsla::cpp::MatrixCOO m(5, 5, values, row, col);
  std::cout << m << std::endl;

  tbsla::utils::vector::streamvector<double>(std::cout, "x", x);
  std::cout << std::endl;

  std::vector<double> r = m.spmv(x);

  tbsla::utils::vector::streamvector<double>(std::cout, "r", r);
  std::cout << std::endl;

  std::vector<double> exp_res{20, 12, 10, 40, 85};
  if(exp_res == r) {
    return 0;
  } else {
    std::cerr << "Error : r has not the right values !" << std::endl;
    return 1;
  }
}
