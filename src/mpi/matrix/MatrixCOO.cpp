#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <vector>
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

  this->values.resize(this->nnz);
  depla_local = depla_general + s * sizeof(double);
  MPI_File_read_at_all(fh, depla_local, this->values.data(), this->nnz, MPI_DOUBLE, &status);
  depla_general += vec_size * sizeof(double);

  /*
  *
  * read row vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  this->row.resize(this->nnz);
  depla_local = depla_general + s * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->row.data(), this->nnz, MPI_INT, &status);
  depla_general += vec_size * sizeof(int);

  /*
  *
  * read col vector
  * 
  */
  MPI_File_read_at_all(fh, depla_general, &vec_size, 1, MPI_UNSIGNED_LONG, &status);
  depla_general += sizeof(size_t);

  this->col.resize(this->nnz);
  depla_local = depla_general + s * sizeof(int);
  MPI_File_read_at_all(fh, depla_local, this->col.data(), this->nnz, MPI_INT, &status);

  MPI_File_close(&fh);
  return 0;
}

std::vector<double> tbsla::mpi::MatrixCOO::spmv(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> send = this->spmv(v, vect_incr);
  std::vector<double> recv(send.size());
  MPI_Allreduce(send.data(), recv.data(), send.size(), MPI_DOUBLE, MPI_SUM, comm);
  return recv;
}

std::vector<double> tbsla::mpi::MatrixCOO::a_axpx_(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  std::vector<double> r = this->spmv(comm, v, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  return this->spmv(comm, r, vect_incr);
}

std::vector<double> tbsla::mpi::MatrixCOO::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done){
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  std::vector<double> b(n_col, 1.0);
  bool converge = false;
  int nb_iterations = 0;
  std::vector<double> b_t(n_col);
  double max, error, teleportation_sum;

  while(!converge && nb_iterations <= max_iterations){
    b_t = b;

    b = this->spmv(comm, b_t);
    max = b[0];
    teleportation_sum = b_t[0];
    for(int i = 1; i < n_col; i++){
      if(max < b[i])
        max = b[i];
      teleportation_sum += b_t[i];
    }
     
    teleportation_sum *= (1-beta)/n_col;
    max = beta*max + teleportation_sum;
    error = 0.0;

    for(int  i = 0 ; i < n_col; i++){
      b[i] = (beta*b[i] + teleportation_sum)/max;
      error += std::abs(b[i] - b_t[i]);
    }

    if(error < epsilon)
      converge = true;
    nb_iterations++;
  }
  
  nb_iterations_done = nb_iterations; 

  double sum = b[0];
  for(int i = 1; i < n_col; i++) {
    sum += b[i];
  }

  for(int i = 0 ; i < n_col; i++) {
    b[i] = b[i]/sum;
  }
  return b;
}

std::vector<double> tbsla::mpi::MatrixCOO::personalized_page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, std::vector<int> personalized_nodes, int &nb_iterations_done){
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  std::vector<double> b(n_col, 0.25);
  bool converge = false;
  int nb_iterations = 0;
  std::vector<double> b_t(n_col);
  double max, error, teleportation_sum;
  while(!converge && nb_iterations <= max_iterations){
    b_t = b;

    b = this->spmv(comm, b_t);
    teleportation_sum = b_t[0];
    for(int i = 1; i < n_col; i++){
      teleportation_sum += b_t[i];
    }
    teleportation_sum *= (1-beta)/personalized_nodes.size(); 

    max = 0.0;
    for(int  i = 0; i < n_col; i++){
      b[i] = beta*b[i];
      if(std::find(personalized_nodes.begin(), personalized_nodes.end(), i) != personalized_nodes.end()){
        b[i] += teleportation_sum;
      }
      if(max < b[i])
      max = b[i]; 
    }

    error = 0.0;
    for(int i = 0; i < n_col; i++){
      b[i] = b[i]/max;
      error += std::abs(b[i] - b_t[i]);
    }
    if(error < epsilon)
      converge = true;
    nb_iterations++;
  }
  
  nb_iterations_done = nb_iterations; 

  double sum = b[0];
  for(int i = 1; i < n_col; i++) {
    sum += b[i];
  }

  for(int i = 0 ; i < n_col; i++) {
    b[i] = b[i]/sum;
  }
  return b;
}
