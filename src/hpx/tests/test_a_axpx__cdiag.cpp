#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>

#include <tbsla/hpx/MatrixCOO.hpp>
#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/hpx/MatrixELL.hpp>
#include <tbsla/cpp/utils/vector.hpp>

void test_mat(tbsla::hpx_::Matrix & m, int N, int nr, int nc, int c) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  Vector_client v(localities[0], nc);
  std::vector<double> v_data = v.get_data().get().get_vect();

  m.fill_cdiag(localities, nr, nc, c, N);
  m.wait();
  Vector_client r = m.a_axpx_(v);
  std::vector<double> r_data = r.get_data().get().get_vect();
  int res = tbsla::utils::vector::test_a_axpx__cdiag(nr, nc, c, v_data, r_data, false);
  std::cout << "return : " << res << std::endl;
  if(res) {
    tbsla::utils::vector::test_a_axpx__cdiag(nr, nc, c, v_data, r_data, true);
    throw "Result vector does not correspond to the expected results !";
  }
}

void test_cdiag(int N, int nr, int nc, int c) {
  std::cout << "---- nr : " << nr << "; nc : " << nc << "; c : " << c << " ---- N : " << N << std::endl;
  tbsla::hpx_::MatrixCOO mcoo;
  test_mat(mcoo, N, nr, nc, c);

  tbsla::hpx_::MatrixCSR mcsr;
  test_mat(mcsr, N, nr, nc, c);

  tbsla::hpx_::MatrixELL mell;
  test_mat(mell, N, nr, nc, c);
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
      test_cdiag(nt, 30, 30, 2 * i);
    }
    for(int nt = 1; nt <= 3; nt++) {
      test_cdiag(nt * 10, 30, 30, 2 * i);
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
