#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <vector>
#include <iostream>

int main(int argc, char** argv) {

  std::vector<double> values{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  std::vector<int> rowptr{ 0, 3, 4, 5, 7, 10 };
  std::vector<int> colidx{ 0, 1, 4, 2, 1, 1, 3, 0, 2, 4 };

  std::vector<double> x{ 1, 2, 3, 4, 5 };

  tbsla::cpp::MatrixCSR m(5, 5, values, rowptr, colidx);
  std::cout << m << std::endl;

  tbsla::utils::vector::streamvector<double>(std::cout, "x", x);
  std::cout << std::endl;

  std::vector<double> r = m.spmv(x);

  tbsla::utils::vector::streamvector<double>(std::cout, "r", r);
  std::cout << std::endl;
}
