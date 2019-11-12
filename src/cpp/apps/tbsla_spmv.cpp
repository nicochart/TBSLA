#include <tbsla/cpp/MatrixCOO.hpp>
#include <tbsla/cpp/utils/mm.hpp>

#include <random>

int main(int argc, char** argv) {

  if(argc == 2) {
    MatrixCOO m = tbsla::utils::io::readMM(std::string(argv[1]));

    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};  // Generates random integers
    std::uniform_real_distribution<double> dist {-1, 1};
    auto gen = [&dist, &mersenne_engine](){ return dist(mersenne_engine); };
    std::vector<double> vec(m.get_n_col());
    generate(begin(vec), end(vec), gen);

    std::vector<double> res = m.spmv(vec);

  }

}
