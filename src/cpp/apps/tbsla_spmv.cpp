#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/mm.hpp>

#include <algorithm>
#include <chrono>
#include <random>
#include <map>

#ifdef TBSLA_HAS_MPI
#include <mpi.h>
#endif

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
}

int main(int argc, char** argv) {

  if(argc == 2) {
#ifdef TBSLA_HAS_MPI
    MPI_Init(&argc, &argv);
    int world, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    auto t_read_start = now();
#ifdef TBSLA_HAS_MPI
    MatrixCOO m;
    m.read_bin_mpiio(MPI_COMM_WORLD, std::string(argv[1]));
#else
    MatrixCOO m = tbsla::utils::io::readMM(std::string(argv[1]));
#endif

    auto t_read_end = now();
    m.print_stats(std::cout);
    m.print_infos(std::cout);

    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_real_distribution<double> dist {-1, 1};
    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    std::vector<double> vec(m.get_n_col());
    std::generate(begin(vec), end(vec), gen);

    //auto t_spmv_start = Clock::now().time_since_epoch().count();
    auto t_spmv_start = now();
#ifdef TBSLA_HAS_MPI
    std::vector<double> res = m.spmv(MPI_COMM_WORLD, vec);
#else
    std::vector<double> res = m.spmv(vec);
#endif
    //auto t_spmv_end = Clock::now().time_since_epoch().count();
    auto t_spmv_end = now();


#ifdef TBSLA_HAS_MPI
    if(rank == 0) {
#endif
    auto t_app_end = now();

    std::map<std::string, std::string> outmap;
    outmap["Machine"] = "my_machine";
    outmap["test"] = "spmv";
    outmap["matrix_format"] = "COO";
    // outmap[""] = "";
    outmap["n_row"] = std::to_string(m.get_n_row());
    outmap["n_col"] = std::to_string(m.get_n_col());
#ifdef TBSLA_HAS_MPI
    outmap["nnz"] = std::to_string(m.get_gnnz());
#else
    outmap["nnz"] = std::to_string(m.get_nnz());
#endif
    outmap["time_read_m"] = std::to_string((t_read_end - t_read_start) / 1e9);
    outmap["time_spmv"] = std::to_string((t_spmv_end - t_spmv_start) / 1e9);
    outmap["time_app"] = std::to_string((t_app_end - t_read_start) / 1e9);
    outmap["lang"] = "C++";
#ifdef TBSLA_HAS_MPI
    outmap["lang"] += "_MPI";
#endif
    outmap["matrix_input"] = std::string(argv[1]);

    std::map<std::string, std::string>::iterator it=outmap.begin();
    std::cout << "{\"" << it->first << "\":\"" << it->second << "\"";
    it++;
    for (; it != outmap.end(); ++it) {
      std::cout << ",\"" << it->first << "\":\"" << it->second << "\"";
    }
    std::cout << "}\n";
#ifdef TBSLA_HAS_MPI
    }
#endif

#ifdef TBSLA_HAS_MPI
    MPI_Finalize();
#endif
  }
}
