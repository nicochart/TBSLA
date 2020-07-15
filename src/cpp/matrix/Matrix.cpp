#include <tbsla/cpp/Matrix.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

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
		double telportation_sum = 0.0;
		b_t = b;
		b = this->spmv(b_t);
		
		for(int i = 0 ; i < n_col; i++){
			telportation_sum += b_t[i];
		}
		telportation_sum *= (1-beta)/n_col ;
		
		for(int  i = 0 ; i < n_col;i++){
			b[i] = beta*b[i] + telportation_sum;
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
	return b;
}