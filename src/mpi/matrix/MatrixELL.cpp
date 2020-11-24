#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <vector>
#include <mpi.h>

int tbsla::mpi::MatrixELL::read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);
  MPI_File_read_at_all(fh, 10 * sizeof(int) + sizeof(long int), &this->max_col, 1, MPI_INT, &status);

  size_t vec_size, depla_general, values_start, columns_start;
  depla_general = 11 * sizeof(int) + sizeof(long int);

  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;

  this->ln_row = tbsla::utils::range::lnv(n_row, pr, NR);
  this->f_row = tbsla::utils::range::pflv(n_row, pr, NR);
  this->ln_col = tbsla::utils::range::lnv(n_col, pc, NC);
  this->f_col = tbsla::utils::range::pflv(n_col, pc, NC);

  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  this->gnnz = vec_size;
  this->nnz = 0;
  values_start = depla_general;
  depla_general += vec_size * sizeof(double);

  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  columns_start = depla_general;

  this->values.reserve(this->ln_row * this->max_col);
  this->columns.reserve(this->ln_row * this->max_col);
  int idx;
  double val;

  for(int i = 0; i < this->ln_row; i++) {
    int incr = 0;
    for(int j = 0; j < this->max_col; j++) {
      MPI_File_read_at(fh, columns_start + ((this->f_row + i) * this->max_col + j) * sizeof(int), &idx, 1, MPI_INT, &status);
      if(idx >= this->f_col && idx < this->f_col + this->ln_col) {
        MPI_File_read_at(fh, values_start + ((this->f_row + i) * this->max_col + j) * sizeof(double), &val, 1, MPI_DOUBLE, &status);
        this->columns.push_back(idx);
        this->values.push_back(val);
        incr++;
      }
    }
    this->nnz += incr;
    for(int j = incr; j < this->max_col; j++) {
      this->columns.push_back(0);
      this->values.push_back(0);
    }
  }

  MPI_File_close(&fh);
  return 0;
}

std::vector<double> tbsla::mpi::MatrixELL::spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> send = this->spmv(v, vect_incr);
  if(this->NC == 1 && this->NR == 1) {
    return send;
  } else if(this->NC == 1 && this->NR > 1) {
    std::vector<int> recvcounts(this->NR);
    std::vector<int> displs(this->NR, 0);
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    std::vector<double> recv(this->get_n_row());
    MPI_Allgatherv(send.data(), send.size(), MPI_DOUBLE, recv.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, comm);
    return recv;
  } else if(this->NC > 1 && this->NR == 1) {
    std::vector<double> recv(this->get_n_row());
    MPI_Allreduce(send.data(), recv.data(), send.size(), MPI_DOUBLE, MPI_SUM, comm);
    return recv;
  } else {
    MPI_Comm row_comm;
    MPI_Comm_split(comm, this->pr, this->pc, &row_comm);
    std::vector<double> recv(send.size());
    MPI_Allreduce(send.data(), recv.data(), send.size(), MPI_DOUBLE, MPI_SUM, row_comm);

    std::vector<double> recv2(this->get_n_row());
    std::vector<int> recvcounts(this->NR);
    std::vector<int> displs(this->NR, 0);
    for(int i = 0; i < this->NR; i++) {
      recvcounts[i] = tbsla::utils::range::lnv(this->get_n_row(), i, this->NR);
    }
    for(int i = 1; i < this->NR; i++) {
      displs[i] = displs[i - 1] + recvcounts[i - 1];
    }
    MPI_Comm col_comm;
    MPI_Comm_split(comm, this->pc, this->pr, &col_comm);
    MPI_Allgatherv(recv.data(), recv.size(), MPI_DOUBLE, recv2.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, col_comm);
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
    return recv2;
  }
}

std::vector<double> tbsla::mpi::MatrixELL::a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> vs(v.begin() + this->f_col, v.begin() + this->f_col + this->ln_col);
  std::vector<double> r = this->spmv(comm, vs, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  std::vector<double> l(r.begin() + this->f_col, r.begin() + this->f_col + this->ln_col);
  return this->spmv(comm, l, vect_incr);
}
