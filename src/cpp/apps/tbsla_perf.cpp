#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/MatrixSCOO.hpp>
#include <tbsla/cpp/MatrixCSR.hpp>
#include <tbsla/cpp/MatrixELL.hpp>
#include <tbsla/cpp/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/InputParser.hpp>
#include <tbsla/Configs.h>

#if TBSLA_COMPILED_WITH_OMP
#include <omp.h>
#include <tbsla/cpp/utils/cpuset_to_cstr.hpp>
#include <cstring>
#include <unistd.h>
#endif

#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <string>

static std::uint64_t now() {
  std::chrono::nanoseconds ns = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<std::uint64_t>(ns.count());
}

int main(int argc, char** argv) {
  InputParser input(argc, argv);

#if TBSLA_COMPILED_WITH_OMP
  //https://www.nics.tennessee.edu/files/tutorials/HybridHello/HybridHello.c
  if(input.has_opt("--report-affinity")) {
    int thread;
    cpu_set_t coremask;
    char clbuf[7 * CPU_SETSIZE], hnbuf[64];
    memset(clbuf, 0, sizeof(clbuf));
    memset(hnbuf, 0, sizeof(hnbuf));
    (void)gethostname(hnbuf, sizeof(hnbuf));
    #pragma omp parallel private(thread, coremask, clbuf)
    {
      thread = omp_get_thread_num();
      (void)sched_getaffinity(0, sizeof(coremask), &coremask);
      tbsla::cpp::utils::cpuset_to_cstr(&coremask, clbuf);
      #pragma omp barrier
      printf("Hello from thread %d, on %s. (core affinity = %s)\n", thread, hnbuf, clbuf);
    }
  }
#endif

  std::string format = input.get_opt("--format");
  if(format == "") {
    std::cerr << "A file format has to be given with the parameter --format format" << std::endl;
    exit(1);
  }

  std::string nr_string = input.get_opt("--NR", "1024");
  std::string nc_string = input.get_opt("--NC", "1024");

  int NR = std::stoi(nr_string);
  int NC = std::stoi(nc_string);
  int C = -1;
  double Q = -1;
  int S = -1;

  std::string matrix = input.get_opt("--matrix");
  std::string matrix_folder = input.get_opt("--matrix_folder", ".");
  if(matrix == "cdiag") {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
  } else if(matrix == "cqmat") {
    std::string c_string = input.get_opt("--C", "8");
    C = std::stoi(c_string);
    std::string q_string = input.get_opt("--Q", "0.1");
    Q = std::stod(q_string);
    std::string s_string = input.get_opt("--S", "0");
    S = std::stoi(s_string);
  } else if (matrix == "") {
    std::cerr << "No matrix has been given with the parameter --matrix matrix." << std::endl;
    exit(1);
  }

  std::string op = input.get_opt("--op");
  if(op == "") {
    std::cerr << "An operation (spmv, a_axpx) has to be given with the parameter --op op" << std::endl;
    exit(1);
  }
  if(op != "spmv" && op != "a_axpx" && op != "spmv_no_redist" && op != "Ax" && op != "Ax_") {
    std::cerr << "OP : " << op << " unrecognized!" << std::endl;
    exit(1);
  }

  tbsla::cpp::Matrix *m;

  if(format == "COO" | format == "coo") {
    m = new tbsla::cpp::MatrixCOO();
  } else if(format == "SCOO" | format == "scoo") {
    m = new tbsla::cpp::MatrixSCOO();
  } else if(format == "CSR" | format == "csr") {
    m = new tbsla::cpp::MatrixCSR();
  } else if(format == "ELL" | format == "ell") {
    m = new tbsla::cpp::MatrixELL();
  } else if(format == "DENSE" | format == "dense") {
    m = new tbsla::cpp::MatrixDENSE();
  } else {
    std::cerr << format << " unrecognized!" << std::endl;
    exit(1);
  }
  auto t_app_start = now();


  if(matrix == "cdiag") {
    m->fill_cdiag(NR, NC, C, 0, 0, 1, 1);
  } else if(matrix == "cqmat") {
    m->fill_cqmat(NR, NC, C, Q, S, 0, 0, 1, 1);
  } else {
    std::ifstream is(matrix_folder + "/" + matrix + "." + format, std::ifstream::binary);
    m->read(is);
    is.close();
  }

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
  std::uniform_real_distribution<double> dist {-1, 1};
  auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
  std::vector<double> vec(m->get_n_col());
  std::vector<double> res(m->get_n_row());
  std::generate(begin(vec), end(vec), gen);

  auto t_op_start = now();
  if(op == "spmv" or op == "spmv_no_redist") {
    double* res = m->spmv(vec.data());
  } else if(op == "Ax" or op == "Ax_") {
    m->Ax(res.data(), vec.data());
  } else if(op == "a_axpx") {
    double* res = m->a_axpx_(vec.data());
  }
  auto t_op_end = now();
  auto t_app_end = now();

  std::map<std::string, std::string> outmap;
  outmap["test"] = op;
  outmap["matrix_format"] = format;
  outmap["n_row"] = std::to_string(m->get_n_row());
  outmap["n_col"] = std::to_string(m->get_n_col());
  outmap["nnz"] = std::to_string(m->get_nnz());
  outmap["time_app_in"] = std::to_string((t_app_end - t_app_start) / 1e9);
  outmap["time_op"] = std::to_string((t_op_end - t_op_start) / 1e9);
  if(op == "spmv" or op == "spmv_no_redist" or op == "Ax" or op == "Ax_") {
    outmap["gflops"] = std::to_string(2.0 * m->get_nnz() / (t_op_end - t_op_start));
  } else if(op == "a_axpx") {
    outmap["gflops"] = std::to_string(4.0 * m->get_nnz() / (t_op_end - t_op_start));
  }
#if TBSLA_COMPILED_WITH_OMP
  outmap["lang"] = "CPPOMP";
  outmap["omp_threads"] = std::to_string(omp_get_max_threads());
#else
  outmap["lang"] = "CPP";
#endif
  outmap["matrix_type"] = matrix;
  outmap["compiler"] = std::string(CMAKE_CXX_COMPILER_ID) + " " + std::string(CMAKE_CXX_COMPILER_VERSION);
  outmap["compile_options"] = std::string(CMAKE_BUILD_TYPE);
  if (std::string(CMAKE_CXX_FLAGS).length() > 0) {
    outmap["compile_options"] += " " + std::string(CMAKE_CXX_FLAGS);
  }
  if (std::string(CMAKE_BUILD_TYPE) == "Release" && std::string(CMAKE_CXX_FLAGS_RELEASE).length() > 0) {
    outmap["compile_options"] += " " + std::string(CMAKE_CXX_FLAGS_RELEASE);
  } else if (std::string(CMAKE_BUILD_TYPE) == "Debug" && std::string(CMAKE_CXX_FLAGS_DEBUG).length() > 0) {
    outmap["compile_options"] += " " + std::string(CMAKE_CXX_FLAGS_DEBUG);
  }
#if TBSLA_COMPILED_WITH_OMP
  outmap["compile_options"] += " " + std::string(OpenMP_CXX_FLAGS);
#endif
  if(matrix == "cdiag") {
    outmap["cdiag_c"] = std::to_string(C);
  } else if(matrix == "cqmat") {
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
  return 0;
}
