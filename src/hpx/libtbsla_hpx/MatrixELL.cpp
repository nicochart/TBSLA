#include <tbsla/hpx/MatrixELL.hpp>
#include <tbsla/cpp/utils/range.hpp>

typedef hpx::components::component<MatrixELL_server> MatrixELL_server_type;
HPX_REGISTER_COMPONENT(MatrixELL_server_type, MatrixELL_server);

typedef MatrixELL_server::get_data_action get_MatrixELL_data_action;
HPX_REGISTER_ACTION(get_MatrixELL_data_action);

void tbsla::hpx::MatrixELL::fill_cdiag(int n_row, int n_col, int cdiag, int rp, int RN){
  this->tbsla::cpp::MatrixELL::fill_cdiag(n_row, n_col, cdiag, rp, RN);
  this->row_incr = tbsla::utils::range::pflv(n_row, rp, RN);
}

std::vector<double> tbsla::hpx::MatrixELL::spmv(const std::vector<double> &v, int vect_incr) const {
  return this->tbsla::cpp::MatrixELL::spmv(v, this->row_incr + vect_incr);
}

static Vector_client spmv_ell_part(MatrixELL_client const& A_p, Vector_client const& v_p, Vector_client const& r_p) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx::MatrixELL> A_data = A_p.get_data();
  hpx::shared_future<Vector_data> v_data = v_p.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([r_p](tbsla::hpx::MatrixELL const& A, Vector_data const& v) -> Vector_client {
      Vector_data r(A.spmv(v.get_vect()));
      return Vector_client(r_p.get_id(), r);
    }),
    A_data, v_data);
}

HPX_PLAIN_ACTION(spmv_ell_part, spmv_ell_part_action);



Vector_client spmv_ell(std::vector<hpx::id_type> localities, std::vector<MatrixELL_client> & tiles, Vector_client v) {
  std::vector<Vector_client> spmv_res;
  spmv_res.resize(tiles.size());
  for (std::size_t i = 0; i != tiles.size(); ++i) {
    spmv_res[i] = Vector_client(localities[i % localities.size()]);
  }

  spmv_ell_part_action act_spmv;

  using hpx::dataflow;
  for (std::size_t i = 0; i != tiles.size(); ++i) {
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;
    auto Op = hpx::util::bind(act_spmv, localities[i % localities.size()], _1, _2, _3);
    spmv_res[i] = dataflow(hpx::launch::async, Op, tiles[i], v, spmv_res[i]);
  }

  return reduce(spmv_res);
}



Vector_client do_spmv_ell(std::size_t N, std::string matrix_file) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = localities.size();    // Number of localities

  std::vector<MatrixELL_client> tiles;
  tiles.resize(N);

  for (std::size_t i = 0; i != N; ++i) {
    tiles[i] = MatrixELL_client(localities[i % nl], i, N, matrix_file);
  }

  tiles[0].get_data().wait();
  tbsla::hpx::MatrixELL m = tiles[0].get_data().get();
  int n_col = m.get_n_col();


  Vector_client v(localities[0], n_col);
  return spmv_ell(localities, tiles, v);
}

Vector_client do_spmv_ell_cdiag(Vector_client v, std::size_t N, int nr, int nc, int cdiag) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = localities.size();    // Number of localities

  std::vector<MatrixELL_client> tiles;
  tiles.resize(N);

  for (std::size_t i = 0; i != N; ++i) {
    tiles[i] = MatrixELL_client(localities[i % nl], nr, nc, cdiag, i, N);
  }

  return spmv_ell(localities, tiles, v);
}

Vector_client a_axpx__ell(std::vector<hpx::id_type> localities, std::vector<MatrixELL_client> & tiles, Vector_client v) {
  Vector_client r = spmv_ell(localities, tiles, v);
  r = add_vectors(localities[0], r, v);
  return spmv_ell(localities, tiles, r);
}

Vector_client do_a_axpx__ell_cdiag(Vector_client v, std::size_t N, int nr, int nc, int cdiag) {
  std::vector<hpx::id_type> localities = hpx::find_all_localities();
  std::size_t nl = localities.size();    // Number of localities

  std::vector<MatrixELL_client> tiles;
  tiles.resize(N);

  for (std::size_t i = 0; i != N; ++i) {
    tiles[i] = MatrixELL_client(localities[i % nl], nr, nc, cdiag, i, N);
  }

  return a_axpx__ell(localities, tiles, v);
}
