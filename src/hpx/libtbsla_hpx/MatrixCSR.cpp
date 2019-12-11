#include <tbsla/hpx/MatrixCSR.hpp>
#include <tbsla/cpp/utils/range.hpp>

void tbsla::hpx_::detail::MatrixCSR::fill_cdiag(int n_row, int n_col, int cdiag, int rp, int RN){
  this->tbsla::cpp::MatrixCSR::fill_cdiag(n_row, n_col, cdiag, rp, RN);
  this->row_incr = tbsla::utils::range::pflv(n_row, rp, RN);
}

void tbsla::hpx_::detail::MatrixCSR::fill_cqmat(int n_row, int n_col, int c, double q, unsigned int seed, int rp, int RN){
  this->tbsla::cpp::MatrixCSR::fill_cqmat(n_row, n_col, c, q, seed, rp, RN);
  this->row_incr = tbsla::utils::range::pflv(n_row, rp, RN);
}

std::vector<double> tbsla::hpx_::detail::MatrixCSR::spmv(const std::vector<double> &v, int vect_incr) const {
  return this->tbsla::cpp::MatrixCSR::spmv(v, this->row_incr + vect_incr);
}

typedef hpx::components::component<tbsla::hpx_::server::MatrixCSR> MatrixCSR_server_type;
HPX_REGISTER_COMPONENT(MatrixCSR_server_type, MatrixCSR_server);

typedef tbsla::hpx_::server::MatrixCSR::get_data_action get_MatrixCSR_data_action;
HPX_REGISTER_ACTION(get_MatrixCSR_data_action);

std::size_t tbsla::hpx_::MatrixCSR::get_n_col() {
  if(this->tiles.size() == 0)
    return 0;
  return tiles[0].get_data().get().get_n_col();
}

std::size_t tbsla::hpx_::MatrixCSR::get_n_row() {
  if(this->tiles.size() == 0)
    return 0;
  return tiles[0].get_data().get().get_n_row();
}

void tbsla::hpx_::MatrixCSR::fill_cdiag(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t nt) {
  this->tiles.resize(nt);
  this->localities = localities;
  int nl = this->localities.size();

  for (std::size_t i = 0; i != nt; ++i) {
    this->tiles[i] = tbsla::hpx_::client::MatrixCSR(this->localities[i % nl], nr, nc, cdiag, i, nt);
  }
}

void tbsla::hpx_::MatrixCSR::fill_cqmat(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t nt) {
  this->tiles.resize(nt);
  this->localities = localities;
  int nl = this->localities.size();

  for (std::size_t i = 0; i != nt; ++i) {
    this->tiles[i] = tbsla::hpx_::client::MatrixCSR(this->localities[i % nl], nr, nc, c, q, seed, i, nt);
  }
}

void tbsla::hpx_::MatrixCSR::read(std::vector<hpx::id_type> localities, std::string matrix_file, std::size_t nt) {
  this->tiles.resize(nt);
  this->localities = localities;
  int nl = this->localities.size();

  for (std::size_t i = 0; i != nt; ++i) {
    this->tiles[i] = tbsla::hpx_::client::MatrixCSR(this->localities[i % nl], i, nt, matrix_file);
  }
}


static Vector_client tbsla::hpx_::detail::spmv_part(tbsla::hpx_::client::MatrixCSR const& A_p, Vector_client const& v_p, Vector_client const& r_p) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx_::detail::MatrixCSR> A_data = A_p.get_data();
  hpx::shared_future<Vector_data> v_data = v_p.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([r_p](tbsla::hpx_::detail::MatrixCSR const& A, Vector_data const& v) -> Vector_client {
      Vector_data r(A.spmv(v.get_vect(), 0));
      return Vector_client(r_p.get_id(), r);
    }),
    A_data, v_data);
}

HPX_PLAIN_ACTION(tbsla::hpx_::detail::spmv_part, spmv_csr_part_action);


Vector_client tbsla::hpx_::MatrixCSR::spmv(Vector_client v) {
  std::vector<Vector_client> spmv_res;
  spmv_res.resize(this->tiles.size());
  for (std::size_t i = 0; i != this->tiles.size(); ++i) {
    spmv_res[i] = Vector_client(this->localities[i % this->localities.size()]);
  }

  spmv_csr_part_action act_spmv;

  using hpx::dataflow;
  for (std::size_t i = 0; i != this->tiles.size(); ++i) {
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;
    auto Op = hpx::util::bind(act_spmv, this->localities[i % this->localities.size()], _1, _2, _3);
    spmv_res[i] = dataflow(hpx::launch::async, Op, this->tiles[i], v, spmv_res[i]);
  }

  return reduce(spmv_res);
}

Vector_client tbsla::hpx_::MatrixCSR::a_axpx_(Vector_client v) {
  Vector_client r = this->spmv(v);
  r = add_vectors(this->localities[0], r, v);
  return this->spmv(r);
}

void tbsla::hpx_::MatrixCSR::wait() {
  for (std::size_t i = 0; i < this->tiles.size(); ++i) {
    tiles[i].get_data().wait();
  }
}

