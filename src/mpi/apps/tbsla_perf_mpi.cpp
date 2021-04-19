#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/mpi/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>

#if TBSLA_COMPILED_WITH_OMP
#include <omp.h>
#endif

#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <string>

#include <mpi.h>

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
}

int main(int argc, char** argv) {
  InputParser input(argc, argv);
  MPI_Init(&argc, &argv);
  int world, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::string format = input.get_opt("--format");
  if(format == "") {
    std::cerr << "A file format has to be given with the parameter --format format" << std::endl;
    exit(1);
  }

  std::string nr_string = input.get_opt("--NR", "1024");
  std::string nc_string = input.get_opt("--NC", "1024");

  std::string gr_string = input.get_opt("--GR", "1");
  std::string gc_string = input.get_opt("--GC", "1");

  int NR = std::stoi(nr_string);
  int NC = std::stoi(nc_string);
  int GR = std::stoi(gr_string);
  int GC = std::stoi(gc_string);
  int C = -1;
  double Q = -1;
  int S = -1;

  if(world != GR * GC) {
    printf("The number of processes (%d) does not match the grid dimensions (%d x %d = %d).\n", world, GR, GC, GR * GC);
    exit(99);
  }

  if(input.has_opt("--cdiag")) {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
  } else if(input.has_opt("--cqmat")) {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
    std::string q_string = input.get_opt("--Q", "0.1");
    Q = std::stod(q_string);
    std::string s_string = input.get_opt("--S", "0");
    S = std::stoi(s_string);
  } else {
    if(rank == 0) {
      std::cerr << "No matrix type has been given (--cdiag or --cqmat)." << std::endl;
    }
    exit(1);
  }

  std::string op = input.get_opt("--op");
  if(op == "") {
    if(rank == 0) {
      std::cerr << "An operation (spmv, a_axpx) has to be given with the parameter --op op" << std::endl;
    }
    exit(1);
  }
  if(op != "spmv" && op != "a_axpx" && op != "spmv_no_redist") {
    if(rank == 0) {
      std::cerr << "OP : " << op << " unrecognized!" << std::endl;
    }
    exit(1);
  }

  tbsla::mpi::Matrix *m;

  if(format == "COO" | format == "coo") {
    m = new tbsla::mpi::MatrixCOO();
  } else if(format == "SCOO" | format == "scoo") {
    m = new tbsla::mpi::MatrixSCOO();
  } else if(format == "CSR" | format == "csr") {
    m = new tbsla::mpi::MatrixCSR();
  } else if(format == "ELL" | format == "ell") {
    m = new tbsla::mpi::MatrixELL();
  } else if(format == "DENSE" | format == "dense") {
    m = new tbsla::mpi::MatrixDENSE();
  } else {
    if(rank == 0) {
      std::cerr << format << " unrecognized!" << std::endl;
    }
    exit(1);
  }
  auto t_app_start = now();


  if(input.has_opt("--cdiag")) {
    m->fill_cdiag(NR, NC, C, rank / GC, rank % GC, GR, GC);
  } else if(input.has_opt("--cqmat")) {
    m->fill_cqmat(NR, NC, C, Q, S, rank / GC, rank % GC, GR, GC);
  }

  if(input.has_opt("--print-infos")) {
    m->print_stats(std::cout);
    m->print_infos(std::cout);
  }

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_real_distribution<double> dist {-1, 1};
  auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
  std::vector<double> vec(m->get_n_col());
  std::generate(begin(vec), end(vec), gen);

  MPI_Barrier(MPI_COMM_WORLD);
  auto t_op_start = now();
  if(op == "spmv") {
    std::vector<double> res = m->spmv(MPI_COMM_WORLD, vec);
  } else if(op == "spmv_no_redist") {
    std::vector<double> res = m->spmv_no_redist(MPI_COMM_WORLD, vec);
  } else if(op == "a_axpx") {
    std::vector<double> res = m->a_axpx_(MPI_COMM_WORLD, vec);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  auto t_op_end = now();

  long int sum_nnz = m->compute_sum_nnz(MPI_COMM_WORLD);
  long int min_nnz = m->compute_min_nnz(MPI_COMM_WORLD);
  long int max_nnz = m->compute_max_nnz(MPI_COMM_WORLD);

  if(rank == 0) {
    auto t_app_end = now();

    std::map<std::string, std::string> outmap;
    outmap["test"] = op;
    outmap["matrix_format"] = format;
    outmap["n_row"] = std::to_string(m->get_n_row());
    outmap["n_col"] = std::to_string(m->get_n_col());
    outmap["g_row"] = gr_string;
    outmap["g_col"] = gc_string;
    outmap["nnz"] = std::to_string(sum_nnz);
    outmap["nnz_min"] = std::to_string(min_nnz);
    outmap["nnz_max"] = std::to_string(max_nnz);
    outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
    outmap["time_op"] = std::to_string((t_op_end - t_op_start) / 1e9);
    outmap["processes"] = std::to_string(world);
#if TBSLA_COMPILED_WITH_OMP
    outmap["lang"] = "MPIOMP";
    outmap["omp_threads"] = std::to_string(omp_get_max_threads());
#else
    outmap["lang"] = "MPI";
#endif
    if(input.has_opt("--cdiag")) {
      outmap["matrix_type"] = "cdiag";
      outmap["cdiag_c"] = std::to_string(C);
    } else if(input.has_opt("--cqmat")) {
      outmap["matrix_type"] = "cqmat";
      outmap["cqmat_c"] = std::to_string(C);
      outmap["cqmat_q"] = std::to_string(Q);
      outmap["cqmat_s"] = std::to_string(S);
    }

    std::map<std::string, std::string>::iterator it=outmap.begin();
    std::cout << "{\"" << it->first << "\":\"" << it->second << "\"";
    it++;
    for (; it != outmap.end(); ++it) {
      std::cout << ",\"" << it->first << "\":\"" << it->second << "\"";
    }
    std::cout << "}\n";
  }

  MPI_Finalize();
}
