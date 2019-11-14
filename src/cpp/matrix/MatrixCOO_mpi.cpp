#include <tbsla/cpp/MatrixCOO.hpp>
#include <vector>
#include <mpi.h>

int MatrixCOO::read_bin_mpiio(MPI_Comm comm, std::string filename) {
  int world, rank;
  MPI_Comm_size(comm, &world);
  MPI_Comm_rank(comm, &rank);

  MPI_File fh;
  MPI_Status status;
  MPI_File_open(comm, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  MPI_File_read_all(fh, &this->n_row, 1, MPI_INT, &status);
  MPI_File_read_all(fh, &this->n_col, 1, MPI_INT, &status);

  size_t vec_size, depla_general, depla_local;
  depla_general = 2 * sizeof(int);

  /*
  *
  * read values vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  int n_read = vec_size / world;
  int mod = vec_size % world;

  if (rank < mod)
    n_read++;

  this->values.resize(n_read);
  if (rank < mod) {
    depla_local = depla_general + rank * n_read * sizeof(double);
  } else {
    depla_local = depla_general + (rank * n_read + mod) * sizeof(double);
  }
  MPI_File_read_at_all(fh, depla_local, this->values.data(), n_read, MPI_DOUBLE, &status);
  depla_general += vec_size * sizeof(double);

  /*
  *
  * read row vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  n_read = vec_size / world;
  mod = vec_size % world;

  if (rank < mod)
    n_read++;

  this->row.resize(n_read);
  if (rank < mod) {
    depla_local = depla_general + rank * n_read * sizeof(int);
  } else {
    depla_local = depla_general + (rank * n_read + mod) * sizeof(int);
  }
  MPI_File_read_at_all(fh, depla_local, this->row.data(), n_read, MPI_INT, &status);
  depla_general += vec_size * sizeof(int);

  /*
  *
  * read col vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  n_read = vec_size / world;
  mod = vec_size % world;

  if (rank < mod)
    n_read++;

  this->col.resize(n_read);
  if (rank < mod) {
    depla_local = depla_general + rank * n_read * sizeof(int);
  } else {
    depla_local = depla_general + (rank * n_read + mod) * sizeof(int);
  }
  MPI_File_read_at_all(fh, depla_local, this->col.data(), n_read, MPI_INT, &status);

  MPI_File_close(&fh);
  return 0;
}

std::vector<double> MatrixCOO::spmv(MPI_Comm comm, const std::vector<double> &v) {
  std::vector<double> send = this->spmv(v);
  std::vector<double> recv(v.size());
  MPI_Allreduce(send.data(), recv.data(), send.size(), MPI_DOUBLE, MPI_SUM, comm);
  return recv;
}
