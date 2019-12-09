#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>
#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/hpx/MatrixELL.hpp>
#include <tbsla/cpp/utils/vector.hpp>

void test_cqmat(int N, int nr, int nc, int c, double q, unsigned int seed) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  Vector_client v(localities[0], nc);
  std::vector<double> v_data = v.get_data().get().get_vect();

  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << "; q : " << q << "; s : " << seed << " ---- N : " << N << std::endl;
  Vector_client r = do_spmv_coo_cqmat(v, N, nr, nc, c, q, seed);
  std::vector<double> rcoo = r.get_data().get().get_vect();

  r = do_spmv_csr_cqmat(v, N, nr, nc, c, q, seed);
  std::vector<double> rcsr = r.get_data().get().get_vect();
  if(rcsr != rcoo) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v ", v_data);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcsr ", rcsr);
    std::cout << std::endl;
    throw "Result vector does not correspond to the expected results !";
  }

  r = do_spmv_ell_cqmat(v, N, nr, nc, c, q, seed);
  std::vector<double> rell = r.get_data().get().get_vect();
  if(rell != rcoo) {
    tbsla::utils::vector::streamvector<double>(std::cout, "v ", v_data);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rcoo ", rcoo);
    std::cout << std::endl;
    tbsla::utils::vector::streamvector<double>(std::cout, "rell ", rell);
    std::cout << std::endl;
    throw "Result vector does not correspond to the expected results !";
  }
}

void test_mat(int N, int nr, int nc, int c) {
  for(double s = 0; s < 2; s++) {
    for(double q = 0; q <= 1; q += 0.2) {
      test_cqmat(N, nr, nc, c, q, s);
    }
  }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 4; nt++) {
      test_mat(nt, 30, 30, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_mat(nt * 10, 30, 30, 2 * i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 4; nt++) {
      test_mat(nt, 20, 30, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_mat(nt * 10, 20, 30, 2 * i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 4; nt++) {
      test_mat(nt, 30, 20, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_mat(nt * 10, 30, 20, 2 * i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 4; nt++) {
      test_mat(nt, 100, 100, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_mat(nt * 10, 100, 100, 2 * i);
    }
  }
  std::cout << "=== finished without error === " << std::endl;
  return hpx::finalize();
}

int main(int argc, char* argv[])
{
  using namespace hpx::program_options;
  options_description desc_commandline;

  // Initialize and run HPX
  return hpx::init(desc_commandline, argc, argv);
}
