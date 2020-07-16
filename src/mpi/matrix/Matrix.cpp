#include <tbsla/mpi/Matrix.hpp>
#include <tbsla/cpp/utils/range.hpp>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <iostream>

std::vector<double> tbsla::mpi::Matrix::page_rank(MPI_Comm comm, double beta, double epsilon, int max_iterations){
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
 
		max = b[0] ; 
		teleportation_sum = b_t[0]; 
		for(int i = 1; i < n_col; i++){
			if(max < b[i])
				max = b[i];
			teleportation_sum += b_t[i];
		}
		 
		teleportation_sum *= (1-beta)/n_col; 

		for(int  i = 0 ; i < n_col; i++){
			b[i] = beta*b[i] + teleportation_sum; 
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
	return b; 
}