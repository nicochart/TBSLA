#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>
#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/hpx/MatrixELL.hpp>
#include <tbsla/cpp/utils/vector.hpp>

void test_cdiag(int N, int nr, int nc, int c) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << " ---- N : " << N << std::endl;
  Vector_client r = do_spmv_coo_cdiag(N, nr, nc, c);
  std::vector<double> r_data = r.get_data().get().get_vect();
  int res = tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r_data, false);
  std::cout << "return : " << res << std::endl;
  if(res) {
    tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r_data, true);
    throw "Result vector does not correspond to the expected results !";
  }

  r = do_spmv_csr_cdiag(N, nr, nc, c);
  r_data = r.get_data().get().get_vect();
  res = tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r_data, false);
  std::cout << "return : " << res << std::endl;
  if(res) {
    tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r_data, true);
    throw "Result vector does not correspond to the expected results !";
  }

  r = do_spmv_ell_cdiag(N, nr, nc, c);
  r_data = r.get_data().get().get_vect();
  res = tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r_data, false);
  std::cout << "return : " << res << std::endl;
  if(res) {
    tbsla::utils::vector::test_vres_cdiag(nr, nc, c, r_data, true);
    throw "Result vector does not correspond to the expected results !";
  }
}

int hpx_main(hpx::program_options::variables_map& vm)
{
  int t = 0;
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(nt, 10, 10, i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(nt, 10, 5, i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(nt, 5, 10, i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(nt, 30, 30, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_cdiag(nt * 10, 30, 30, 2 * i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(nt, 20, 30, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_cdiag(nt * 10, 20, 30, 2 * i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(nt, 30, 20, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_cdiag(nt * 10, 30, 20, 2 * i);
    }
  }
  for(int i = 0; i <= 12; i++) {
    std::cout << "=== test " << t++ << " ===" << std::endl;
    for(int nt = 1; nt <= 5; nt++) {
      test_cdiag(nt, 100, 100, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_cdiag(nt * 10, 100, 100, 2 * i);
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
