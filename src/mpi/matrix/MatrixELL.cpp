#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <algorithm>
#include <vector>
#include <mpi.h>

int tbsla::mpi::MatrixELL::read_bin_mpiio(MPI_Comm comm, std::string filename) {
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
