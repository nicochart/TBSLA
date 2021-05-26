#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/values_generation.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/cpp/utils/split.hpp>
#include <numeric>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>

void tbsla::cpp::MatrixCOO::init(int n_row, int n_col, long int n_values) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
  this->values.reserve(n_values);
  this->row.reserve(n_values);
  this->col.reserve(n_values);
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col, std::vector<double> & values, std::vector<int> & row,  std::vector<int> & col) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->values = values;
  this->row = row;
  this->col = col;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col, long int n_values) {
  this->init(n_row, n_col, n_values);
}

tbsla::cpp::MatrixCOO::MatrixCOO(int n_row, int n_col) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->ln_row = n_row;
  this->ln_col = n_col;
  this->f_row = 0;
  this->f_col = 0;
  this->pr = 0;
  this->pc = 0;
  this->NR = 1;
  this->NC = 1;
}

std::ostream& tbsla::cpp::MatrixCOO::print_as_dense(std::ostream& os) {
  std::vector<double> d(this->n_row * this->n_col, 0);
  for(int i = 0; i < this->row.size(); i++) {
    d[row[i] * this->n_col + col[i]] += this->values[i];
  }
  tbsla::utils::vector::print_dense_matrix(this->n_row, this->n_col, d, os);
  return os;
}

std::ostream& tbsla::cpp::MatrixCOO::print(std::ostream& os) const {
  os << "-----------------" << std::endl;
  os << "------ COO ------" << std::endl;
  os << "-----------------" << std::endl;
  os << "number of rows             -  n_row   : " << this->n_row << std::endl;
  os << "number of columns          -  n_col   : " << this->n_col << std::endl;
  os << "local number of rows       -  ln_row  : " << this->ln_row << std::endl;
  os << "local number of columns    -  ln_col  : " << this->ln_col << std::endl;
  os << "first row                  -  f_row   : " << this->f_row << std::endl;
  os << "first column               -  f_col   : " << this->f_col << std::endl;
  os << "number of non-zero elts    -  nnz     : " << this->nnz << std::endl;
  os << "block position (row)       -  pr      : " << this->pr << std::endl;
  os << "block position (column)    -  pc      : " << this->pc << std::endl;
  os << "number of blocks (row)     -  NR      : " << this->NR << std::endl;
  os << "number of blocks (column)  -  NC      : " << this->NC << std::endl;
  tbsla::utils::vector::streamvector<double>(os, "val", this->values);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "row", this->row);
  os << std::endl;
  tbsla::utils::vector::streamvector<int>(os, "col", this->col);
  os << std::endl;
  os << "-----------------" << std::endl << std::flush;
  return os;
}

std::ostream & tbsla::cpp::operator<<( std::ostream &os, const tbsla::cpp::MatrixCOO &m) {
  return m.print(os);
}

std::vector<double> tbsla::cpp::MatrixCOO::spmv(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r (this->n_row, 0);
  this->Ax(r, v, vect_incr);
  return r;
}

inline void tbsla::cpp::MatrixCOO::Ax(std::vector<double> &r, const std::vector<double> &v, int vect_incr) const {
  // https://stackoverflow.com/questions/43168661/openmp-and-reduction-on-stdvector
  #pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))
  #pragma omp parallel for reduction(vec_double_plus: r)
  for (std::size_t i = 0; i < this->values.size(); i++) {
     r[this->row[i] + vect_incr] += this->values[i] * v[this->col[i]];
  }
}

void tbsla::cpp::MatrixCOO::push_back(int r, int c, double v) {
  if(r >= this->n_row || r < 0)
    return;
  if(c >= this->n_col || c < 0)
    return;
  this->values.push_back(v);
  this->row.push_back(r);
  this->col.push_back(c);
}

void tbsla::cpp::MatrixCOO::update_nnz() {
  this->values.shrink_to_fit();
  this->col.shrink_to_fit();
  this->row.shrink_to_fit();
  this->nnz = values.size();
}

std::ostream & tbsla::cpp::MatrixCOO::print_infos(std::ostream &os) {
  os << "-----------------" << std::endl;
  os << "------ COO ------" << std::endl;
  os << "--- general   ---" << std::endl;
  os << "n_row : " << n_row << std::endl;
  os << "n_col : " << n_col << std::endl;
  os << "--- capacity  ---" << std::endl;
  os << "values : " << values.capacity() << std::endl;
  os << "row : " << row.capacity() << std::endl;
  os << "col : " << col.capacity() << std::endl;
  os << "--- size      ---" << std::endl;
  os << "values : " << values.size() << std::endl;
  os << "row : " << row.size() << std::endl;
  os << "col : " << col.size() << std::endl;
  os << "-----------------" << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixCOO::print_stats(std::ostream &os) {
  int s = 0, u = 0, d = 0;
  if(row.size() != col.size()) {
    std::cerr << "Err \n";
    return os;
  }
  for(int i = 0; i < row.size(); i++) {
    if(row[i] < col[i]) {
      s++;
    } else if(row[i] > col[i]) {
      u++;
    } else {
      d++;
    }
  }
  os << "upper values : " << u << std::endl;
  os << "lower values : " << s << std::endl;
  os << "diag  values : " << d << std::endl;
  return os;
}

std::ostream & tbsla::cpp::MatrixCOO::write(std::ostream &os) {
  os.write(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  os.write(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  os.write(reinterpret_cast<char*>(&this->ln_row), sizeof(this->ln_row));
  os.write(reinterpret_cast<char*>(&this->ln_col), sizeof(this->ln_col));
  os.write(reinterpret_cast<char*>(&this->f_row), sizeof(this->f_row));
  os.write(reinterpret_cast<char*>(&this->f_col), sizeof(this->f_col));
  os.write(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
  os.write(reinterpret_cast<char*>(&this->pr), sizeof(this->pr));
  os.write(reinterpret_cast<char*>(&this->pc), sizeof(this->pc));
  os.write(reinterpret_cast<char*>(&this->NR), sizeof(this->NR));
  os.write(reinterpret_cast<char*>(&this->NC), sizeof(this->NC));

  size_t size_v = this->values.size();
  os.write(reinterpret_cast<char*>(&size_v), sizeof(size_v));
  os.write(reinterpret_cast<char*>(this->values.data()), this->values.size() * sizeof(double));

  size_t size_r = this->row.size();
  os.write(reinterpret_cast<char*>(&size_r), sizeof(size_r));
  os.write(reinterpret_cast<char*>(this->row.data()), this->row.size() * sizeof(int));

  size_t size_c = this->col.size();
  os.write(reinterpret_cast<char*>(&size_c), sizeof(size_c));
  os.write(reinterpret_cast<char*>(this->col.data()), this->col.size() * sizeof(int));
  return os;
}

std::istream & tbsla::cpp::MatrixCOO::read(std::istream &is, std::size_t pos, std::size_t n) {
  is.read(reinterpret_cast<char*>(&this->n_row), sizeof(this->n_row));
  is.read(reinterpret_cast<char*>(&this->n_col), sizeof(this->n_col));
  is.read(reinterpret_cast<char*>(&this->ln_row), sizeof(this->ln_row));
  is.read(reinterpret_cast<char*>(&this->ln_col), sizeof(this->ln_col));
  is.read(reinterpret_cast<char*>(&this->f_row), sizeof(this->f_row));
  is.read(reinterpret_cast<char*>(&this->f_col), sizeof(this->f_col));
  is.read(reinterpret_cast<char*>(&this->nnz), sizeof(this->nnz));
  is.read(reinterpret_cast<char*>(&this->pr), sizeof(this->pr));
  is.read(reinterpret_cast<char*>(&this->pc), sizeof(this->pc));
  is.read(reinterpret_cast<char*>(&this->NR), sizeof(this->NR));
  is.read(reinterpret_cast<char*>(&this->NC), sizeof(this->NC));


  size_t size;
  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->values.resize(size);
  is.read(reinterpret_cast<char*>(this->values.data()), size * sizeof(double));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->row.resize(size);
  is.read(reinterpret_cast<char*>(this->row.data()), size * sizeof(int));

  is.read(reinterpret_cast<char*>(&size), sizeof(size_t));
  this->col.resize(size);
  is.read(reinterpret_cast<char*>(this->col.data()), size * sizeof(int));
  return is;
}

void tbsla::cpp::MatrixCOO::readMM(std::string fname) {
  std::ifstream is(fname);
  std::string line;
  std::string delim(" ");

  if (is.is_open()) {
    getline(is, line);
    std::vector<std::string> splits = tbsla::utils::io::split(line, delim);
    std::cout << line << "\n";
    if(splits[0].compare(std::string("%%MatrixMarket")) == 0
       && splits[1].compare(std::string("matrix")) == 0
       && splits[2].compare(std::string("coordinate")) == 0
       && splits[3].compare(std::string("real")) == 0 ) {
      if(splits[4].compare(std::string("general")) == 0) {
        while(line.rfind("%", 0) == 0) {
          getline(is, line);
        }
        std::cout << line << "\n";
        std::stringstream ss(line);
        int nc, nr, nv;
        ss >> nr;
        ss >> nc;
        ss >> nv;
        this->init(nr, nc, nv);
        int r, c;
        double v;
        for(int i = 0; i < nv && !is.eof(); i++) {
          is >> r;
          is >> c;
          is >> v;
          this->push_back(r - 1, c - 1, v);
        }
        this->update_nnz();
      } else if(splits[4].compare(std::string("symmetric")) == 0) {
        while(line.rfind("%", 0) == 0) {
          getline(is, line);
        }
        std::cout << line << "\n";
        std::stringstream ss(line);
        int nc, nr, nv;
        ss >> nr;
        ss >> nc;
        ss >> nv;
        this->init(nr, nc, nv * 2);
        int r, c;
        double v;
        for(int i = 0; i < nv && !is.eof(); i++) {
          is >> r;
          is >> c;
          is >> v;
          this->push_back(r - 1, c - 1, v);
          if(r != c)
            this->push_back(c - 1, r - 1, v);
        }
        this->update_nnz();
      } else {
        throw tbsla::cpp::MatrixFormatReadException();
      }
    } else {
      throw tbsla::cpp::MatrixFormatReadException();
    }
  }
}

void tbsla::cpp::MatrixCOO::fill_cdiag(int n_row, int n_col, int cdiag, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->values.clear();
  this->col.clear();
  this->row.clear();

  this->ln_row = this->NR;
  this->ln_col = this->NC;
  this->f_row = 0;
  this->f_col = 0;

  long int gnv = std::max(std::min(n_row, n_col - cdiag), 0) + std::max(std::min(n_row - cdiag, n_col), 0);
  if(cdiag == 0)
    gnv /= 2;
  this->nnz = 0;
  if(gnv == 0)
    return;

  long int s = tbsla::utils::range::pflv(gnv, pr * NC + pc, NR * NC);
  long int n = tbsla::utils::range::lnv(gnv, pr * NC + pc, NR * NC);
  this->nnz = n;

  this->values.reserve(n);
  this->col.reserve(n);
  this->row.reserve(n);

  for(int i = s; i < s + n; i++) {
    auto tuple = tbsla::utils::values_generation::cdiag_value(i, gnv, n_row, n_col, cdiag);
    this->push_back(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
  }
}

void tbsla::cpp::MatrixCOO::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->n_row = n_row;
  this->n_col = n_col;
  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->values.clear();
  this->col.clear();
  this->row.clear();

  this->ln_row = this->NR;
  this->ln_col = this->NC;
  this->f_row = 0;
  this->f_col = 0;

  long int gnv = 0;
  for(long int i = 0; i < std::min(n_col - std::min(c, n_col) + 1, n_row); i++) {
    gnv += std::min(c, n_col);
  }
  for(long int i = 0; i < std::min(n_row, n_col) - std::min(n_col - std::min(c, n_col) + 1, n_row); i++) {
    gnv += std::min(c, n_col) - i - 1;
  }
  if(gnv == 0)
    return;

  long int s = tbsla::utils::range::pflv(gnv, pr * NC + pc, NR * NC);
  long int n = tbsla::utils::range::lnv(gnv, pr * NC + pc, NR * NC);
  this->nnz = n;

  this->values.reserve(n);
  this->col.reserve(n);
  this->row.reserve(n);

  for(long int i = s; i < s + n; i++) {
    auto tuple = tbsla::utils::values_generation::cqmat_value(i, n_row, n_col, c, q, seed_mult);
    this->push_back(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
  }

  this->values.shrink_to_fit();
  this->col.shrink_to_fit();
  this->row.shrink_to_fit();
}

void tbsla::cpp::MatrixCOO::fill_cqmat_stochastic(int n_row, int n_col, int c, double q, unsigned int seed_mult, int pr, int pc, int NR, int NC) {
  this->fill_cqmat(n_row, n_col, c, q, seed_mult, pr, pc, NR, NC);
  std::vector<double> sum = tbsla::utils::values_generation::cqmat_sum_columns(n_row, n_col, c, q, seed_mult);
  for(long int i = 0; i < this->values.size(); i++) {
    this->values[i] /= sum[this->col[i]];
  }
}

void tbsla::cpp::MatrixCOO::normalize_columns() {
  std::vector<double> sum(this->n_col, 0);
  for(long int i = 0; i < this->values.size(); i++) {
    sum[this->col[i]] += this->values[i];
  }
  for(long int i = 0; i < this->values.size(); i++) {
    this->values[i] /= sum[this->col[i]];
  }
}
