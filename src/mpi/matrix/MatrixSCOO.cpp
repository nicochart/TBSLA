#include <tbsla/cpp/utils/range.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <algorithm>
#include <vector>
#include <mpi.h>

int tbsla::mpi::MatrixSCOO::read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);

  size_t vec_size, depla_general, values_start, row_start, col_start;
  depla_general = 10 * sizeof(int) + sizeof(long int);

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
  row_start = values_start + sizeof(size_t) + this->gnnz * sizeof(double);
  col_start = row_start + sizeof(size_t) + this->gnnz * sizeof(int);
  int c, r;
  double v;

  this->values.reserve(this->gnnz / NR / NC);
  this->col.reserve(this->gnnz / NR / NC);
  this->row.reserve(this->gnnz / NR / NC);

  for(int i = 0; i < this->gnnz; i++) {
    MPI_File_read_at(fh, row_start + i * sizeof(int), &r, 1, MPI_INT, &status);
    MPI_File_read_at(fh, col_start + i * sizeof(int), &c, 1, MPI_INT, &status);
    if(r >= f_row && r < f_row + ln_row && c >= f_col && c < f_col + ln_col) {
      MPI_File_read_at(fh, values_start + i * sizeof(double), &v, 1, MPI_DOUBLE, &status);
      this->row.push_back(r);
      this->col.push_back(c);
      this->values.push_back(v);
      this->nnz++;
    }
  }
  this->values.shrink_to_fit();
  this->col.shrink_to_fit();
  this->row.shrink_to_fit();

  return 0;
}

std::vector<double> tbsla::mpi::MatrixSCOO::spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
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

std::vector<double> tbsla::mpi::MatrixSCOO::a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> vs(v.begin() + this->f_col, v.begin() + this->f_col + this->ln_col);
  std::vector<double> r = this->spmv(comm, vs, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  std::vector<double> l(r.begin() + this->f_col, r.begin() + this->f_col + this->ln_col);
  return this->spmv(comm, l, vect_incr);
}
