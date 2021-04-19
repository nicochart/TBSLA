#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <iostream>

long int const tbsla::mpi::Matrix::compute_sum_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
  return nnz;
}

long int const tbsla::mpi::Matrix::compute_min_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
  return nnz;
}

long int const tbsla::mpi::Matrix::compute_max_nnz(MPI_Comm comm) {
  long int lnnz = this->get_nnz();
  long int nnz;
  MPI_Reduce(&lnnz, &nnz, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
  return nnz;
}

std::vector<double> tbsla::mpi::Matrix::spmv_no_redist(MPI_Comm comm, const std::vector<double> &v, int vect_incr) {
  return this->spmv(v, vect_incr);
}

std::vector<double> tbsla::mpi::Matrix::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, int &nb_iterations_done){
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  std::vector<double> b(n_col, 1.0);
  bool converge = false;
  int nb_iterations = 0;
  std::vector<double> b_t(n_col);
  double max, error, teleportation_sum;

  while(!converge && nb_iterations < max_iterations){
    b_t = b;
    
    auto begin = b_t.begin() + f_col; 
    auto end = begin + ln_col; 
    std::vector<double> tmp(begin, end); 
    b = this->spmv(comm, tmp);
 
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

std::vector<double> tbsla::mpi::Matrix::personalized_page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations, std::vector<int> personalized_nodes, int &nb_iterations_done){
  int proc_rank;
  MPI_Comm_rank(comm, &proc_rank);
  std::vector<double> b(n_col, 0.25);
  bool converge = false;
  int nb_iterations = 0;
  std::vector<double> b_t(n_col);
  double max, error, teleportation_sum;
  while(!converge && nb_iterations < max_iterations){
    b_t = b;
    auto begin = b_t.begin() + f_col; 
    auto end = begin + ln_col; 
    std::vector<double> tmp(begin, end); 
    b = this->spmv(comm, tmp);

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
