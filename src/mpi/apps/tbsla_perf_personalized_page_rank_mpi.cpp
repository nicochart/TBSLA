#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/mpi/MatrixDENSE.hpp>
#include <vector> 
#include <tbsla/cpp/utils/split.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>
#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <string>
#include <mpi.h>
#include <iostream>

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

  std::string matrix_dim_string = input.get_opt("--matrix_dim", "1024");

  std::string gr_string = input.get_opt("--GR", "1");
  std::string gc_string = input.get_opt("--GC", "1");

  std::string beta_string = input.get_opt("--beta", "0.85");
  std::string epsilon_string = input.get_opt("--epsilon", "0.00001");
  std::string max_iterations_string = input.get_opt("--max-iterations", "10000");

  double epsilon = std::stod(epsilon_string);
  double beta = std::stod(beta_string);
  int max_iterations = std::stoi(max_iterations_string);;
  int nb_iterations_done;
  int matrix_dim = std::stoi(matrix_dim_string);
  int GR = std::stoi(gr_string);
  int GC = std::stoi(gc_string);
  int C = -1;
  double Q = -1;
  int S = -1;

  if(world != GR * GC) {
    printf("The number of processes (%d) does not match the grid dimensions (%d x %d = %d).\n", world, GR, GC, GR * GC);
    exit(99);
  }

  std::vector<int> personalized_nodes;
  std::string str_personalized_nodes = input.get_opt("--personalized_nodes"); 
  if(str_personalized_nodes == ""){
    std::cerr << "A list of personalized nodes is needed" << std::endl; 
    exit(1);
  }
  else{
    std::string delim(" ");
    std::vector<std::string> list_nodes = tbsla::utils::io::split(str_personalized_nodes, delim); 
    for(int i = 0; i < list_nodes.size(); i++){
      double to_insert = std::stod(list_nodes[i]);
      if (std::find(personalized_nodes.begin(), personalized_nodes.end(), to_insert) == personalized_nodes.end()){
        personalized_nodes.push_back(to_insert);
      }
    }
  }

  std::string c_string = input.get_opt("--C", "8");
  C = std::stoi(c_string);
  std::string q_string = input.get_opt("--Q", "0.1");
  Q = std::stod(q_string);
  std::string s_string = input.get_opt("--S", "0");
  S = std::stoi(s_string);

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

  m->fill_cqmat_stochastic(matrix_dim, matrix_dim, C, Q, S, rank / GC, rank % GC, GR, GC);

  if(input.has_opt("--print-infos")) {
    m->print_stats(std::cout);
    m->print_infos(std::cout);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  auto t_op_start = now();
  std::vector<double> res = m->personalized_page_rank(MPI_COMM_WORLD, beta, epsilon, max_iterations, personalized_nodes, nb_iterations_done);
  MPI_Barrier(MPI_COMM_WORLD);
  auto t_op_end = now();

  if(rank == 0) {
    auto t_app_end = now();

    std::map<std::string, std::string> outmap;
    outmap["test"] = "page_rank";
    outmap["matrix_format"] = format;
    outmap["n"] = std::to_string(m->get_n_col());
    outmap["g_row"] = gr_string;
    outmap["g_col"] = gc_string;
    outmap["nnz"] = std::to_string(m->get_gnnz());
    outmap["nb_iterations"] = std::to_string(nb_iterations_done);    
    outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
    outmap["time_op"] = std::to_string((t_op_end - t_op_start) / 1e9);
    outmap["lang"] = "MPI";
    outmap["matrix_type"] = "cqmat stochastic";
    outmap["cqmat_c"] = std::to_string(C);
    outmap["cqmat_q"] = std::to_string(Q);
    outmap["cqmat_s"] = std::to_string(S);
  

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
