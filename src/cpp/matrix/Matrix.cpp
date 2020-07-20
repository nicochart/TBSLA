#include <tbsla/cpp/Matrix.hpp>
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
std::vector<double> tbsla::cpp::Matrix::a_axpx_(const std::vector<double> &v, int vect_incr) const {
  std::vector<double> r = this->spmv(v, vect_incr);
  std::transform (r.begin(), r.end(), v.begin(), r.begin(), std::plus<double>());
  r = this->spmv(r, vect_incr);
  return r;
}

std::vector<double> tbsla::cpp::Matrix::page_rank(double epsilon, double beta, int max_iteration){
	std::vector<double> b(n_col, 1.0);
	bool converge = false;
	double max, error;
	std::vector<double> b_t(n_col);
	int nb_iterations  = 0;
	while(!converge && nb_iterations < max_iteration){
		double teleportation_sum = 0.0;
		b_t = b;
		b = this->spmv(b_t);
		
		for(int i = 0 ; i < n_col; i++){
			teleportation_sum += b_t[i];
		}
		teleportation_sum *= (1-beta)/n_col ;
		
		for(int  i = 0 ; i < n_col;i++){
			b[i] = beta*b[i] + teleportation_sum;
		}

		max  = b[0];
		for(int i =  0; i < n_col;i++){
			if(max < b[i])
				max = b[i];
		}
		error = 0.0;
		for (int i = 0;i< n_col;i++){
			b[i] = b[i]/max;
			error += std::abs(b[i]- b_t[i]);
		}
		if(error < epsilon){
			converge = true;
		}

		nb_iterations++;
	}
	double sum = b[0];
	for(int i = 1; i < n_col; i++) {
		sum += b[i];
	}

	for(int i = 0 ; i < n_col; i++) {
		b[i] = b[i]/sum; 
	}
	return b;
}

std::vector<double> tbsla::cpp::Matrix::personalized_page_rank(double epsilon, double beta, int max_iteration, std::vector<int> personalized_nodes){
	std::vector<double> b(n_col, 0.25);
	bool converge = false;
	double max, error;
	std::vector<double> b_t(n_col);
	int nb_iterations  = 0;
	while(!converge && nb_iterations < max_iteration){
		double teleportation_sum = 0.0;
		b_t = b;
		b = this->spmv(b_t);

		for(int i = 0 ; i < n_col; i++){
			teleportation_sum += b_t[i];
		}
		teleportation_sum *= (1-beta)/personalized_nodes.size() ;
		
		for(int  i = 0 ; i < n_col;i++){
			b[i] = beta*b[i];
			if(std::find(personalized_nodes.begin(), personalized_nodes.end(), i) != personalized_nodes.end()){
				b[i] += teleportation_sum;
			}
		}

		max  = b[0];
		for(int i =  1; i < n_col;i++){
			if(max < b[i])
				max = b[i];
		}
		error = 0.0;
		for (int i = 0;i< n_col;i++){
			b[i] = b[i]/max;
			error += std::abs(b[i]- b_t[i]);
		}
		if(error < epsilon){
			converge = true;
		}

		nb_iterations++;
	}
	double sum = b[0];
	for(int i = 1; i < n_col; i++) {
		sum += b[i];
	}

	for(int i = 0 ; i < n_col; i++) {
		b[i] = b[i]/sum; 
	}
	return b;
}