#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <mpi.h>

int tbsla::mpi::MatrixCOO::read_bin_mpiio(MPI_Comm comm, std::string filename, int pr, int pc, int NR, int NC) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);
  MPI_File_read_at_all(fh, 6 * sizeof(int), &this->gnnz, 1, MPI_LONG, &status);

  this->pr = pr;
  this->pc = pc;
  this->NR = NR;
  this->NC = NC;
  this->ln_row = this->n_row;
  this->ln_col = this->n_col;
  this->f_row = 0;
  this->f_col = 0;

  size_t vec_size, depla_general, depla_local;
  depla_general = 10 * sizeof(int) + sizeof(long int);

  /*
  *
  * read values vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);
  int s = tbsla::utils::range::pflv(vec_size, pr * NC + pc, NR * NC);
  this->nnz = tbsla::utils::range::lnv(vec_size, pr * NC + pc, NR * NC);

  if (this->values)
    delete[] this->values;
  this->values = new double[this->nnz]();
  depla_local = depla_general + s * sizeof(double);
  MPI_File_read_at_all(fh, depla_local, this->values, this->nnz, MPI_DOUBLE, &status);
  depla_general += vec_size * sizeof(double);

  /*
  *
  * read row vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  if (this->row)
    delete[] this->row;
  this->row = new int[this->nnz]();
  depla_local = depla_general + s * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->row, this->nnz, MPI_INT, &status);
  depla_general += vec_size * sizeof(int);

  /*
  *
  * read col vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  if (this->col)
    delete[] this->col;
  this->col = new int[this->nnz]();
  depla_local = depla_general + s * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->col, this->nnz, MPI_INT, &status);

  MPI_File_close(&fh);
  return 0;
}

double* tbsla::mpi::MatrixCOO::spmv(MPI_Comm comm, const double* v, int vect_incr) {
  double* send = this->spmv(v, vect_incr);
  double* recv = new double[this->n_row]();
  MPI_Allreduce(send, recv, this->n_row, MPI_DOUBLE, MPI_SUM, comm);
  delete[] send;
  return recv;
}

double* tbsla::mpi::MatrixCOO::a_axpx_(MPI_Comm comm, const double* v, int vect_incr) {
  double* r = this->spmv(comm, v, vect_incr);
  std::transform (r, r + this->n_row, v, r, std::plus<double>());
  return this->spmv(comm, r, vect_incr);
}

