#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <vector>
#include <algorithm>
#include <mpi.h>

int tbsla::mpi::MatrixCSR::read_bin_mpiio(MPI_Comm comm, std::string filename) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);

  size_t vec_size, depla_general, depla_local, vstart;
  depla_general = 2 * sizeof(int);

  /*
  *
  * skip values vector for now
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  this->gnnz = vec_size;
  vstart = depla_general;
  depla_general += vec_size * sizeof(double);

  /*
  *
  * read rowptr vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  int n_read = this->n_row / world;
  int mod = this->n_row % world;

  if (rank < mod)
    n_read++;

  this->rowptr.resize(n_read + 1);
  if (rank < mod) {
    this->f_row = rank * n_read;
  } else {
    this->f_row = rank * n_read + mod;
  }
  depla_local = depla_general + this->f_row * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->rowptr.data(), n_read + 1, MPI_INT, &status);
  depla_general += vec_size * sizeof(int);

  /*
  *
  * read colidx vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  n_read = this->rowptr.back() - this->rowptr.front();

  this->colidx.resize(n_read);
  depla_local = depla_general + this->rowptr.front() * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->colidx.data(), n_read, MPI_INT, &status);

  /*
  *
  * read values vector
  * 
  */
  this->values.resize(n_read);
  depla_local = vstart + this->rowptr.front() * sizeof(double);
  MPI_File_read_at_all(fh, depla_local, this->values.data(), n_read, MPI_DOUBLE, &status);

  MPI_File_close(&fh);
  return 0;
}

std::vector<double> tbsla::mpi::MatrixCSR::spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
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

std::vector<double> tbsla::mpi::MatrixCSR::a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> vs(v.begin() + this->f_col, v.begin() + this->f_col + this->ln_col);
  std::vector<double> r = this->spmv(comm, vs, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  std::vector<double> l(r.begin() + this->f_col, r.begin() + this->f_col + this->ln_col);
  return this->spmv(comm, l, vect_incr);
}
