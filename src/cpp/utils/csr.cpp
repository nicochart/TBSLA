#include <tbsla/cpp/utils/csr.hpp>

bool tbsla::cpp::utils::csr::compare_row(std::vector<int> row, std::vector<int> col, unsigned i, unsigned j) {
  if (row[i] == row[j]) {
    return col[i] < col[j];
  }
  return row[i] < row[j];
}
