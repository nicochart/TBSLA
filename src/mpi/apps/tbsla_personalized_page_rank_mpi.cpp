#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/mpi/MatrixCOO.hpp>
#include <tbsla/mpi/MatrixSCOO.hpp>
#include <tbsla/mpi/MatrixCSR.hpp>
#include <tbsla/mpi/MatrixELL.hpp>
#include <tbsla/mpi/MatrixDENSE.hpp>
#include <tbsla/cpp/utils/vector.hpp>
#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <map>
#include <numeric>
#include <iostream>
#include <tbsla/cpp/utils/InputParser.hpp>

int main(int argc, char **argv){
	MPI_Init(NULL, NULL);

	int world, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
		
  InputParser input(argc, argv);
  double epsilon = 0.00001;
  double beta = 1;
  int max_iterations = 100;
  int size_personalized_nodes = 5;

  if(input.has_opt("--beta")) {
   	std::string beta_string = input.get_opt("--beta", "1");
   	beta = std::stod(beta_string);
  }
	if(input.has_opt("--epsilon")) {
    	std::string epsilon_string = input.get_opt("--epsilon", "1");
    	epsilon = std::stod(epsilon_string);
  }

  if(input.has_opt("--max-iterations")) {	
   	std::string max_iterations_string = input.get_opt("--max-iterations", "1");
   	max_iterations = std::stoi(max_iterations_string);
 	}
	
	std::string matrix_input = input.get_opt("--matrix_input");
  if(matrix_input == "") {
    std::cerr << "A matrix file has to be given with the parameter --matrix_input file" << std::endl;
   	exit(1);
  }

	std::string format = input.get_opt("--format");
  if(format == "") {
   	std::cerr << "A file format has to be given with the parameter --format format" << std::endl;
   	exit(1);
  }
    
	std::string gr_string = input.get_opt("--GR", "1");
  std::string gc_string = input.get_opt("--GC", "1");
  int GR = std::stoi(gr_string);
  int GC = std::stoi(gc_string);
  if(world != GR * GC) {
   	printf("The number of processes (%d) does not match the grid dimensions (%d x %d = %d).\n", world, GR, GC, GR * GC);
   	exit(99);
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
  m->read_bin_mpiio(MPI_COMM_WORLD, matrix_input, rank / GC, rank % GC, GR, GC);

  std::random_device rnd_device;
  std::mt19937 mersenne_engine {rnd_device()};// Generates random integers
  std::uniform_real_distribution<double> dist {-1, 1};
  auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine);};

  std::vector<int> personalized_nodes(size_personalized_nodes);
  std::generate(personalized_nodes.begin(), personalized_nodes.end(), gen);

	std::vector<double> b(m->get_n_col());

	b = m->personalized_page_rank(MPI_COMM_WORLD, beta, epsilon, max_iterations, personalized_nodes);
 	
  if(rank == 0){
    std::cout <<  "SOLUTION ["; 
    for(int  i = 0;i < m->get_n_col();i++){
      std::cout << b[i] << ", " ;
    } 
    std::cout << "]"<< std::endl ;    
  }
  MPI_Finalize();

}