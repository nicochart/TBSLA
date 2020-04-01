#include <tbsla/hpx/MatrixCOO.hpp>

typedef hpx::components::component<tbsla::hpx_::server::MatrixCOO> MatrixCOO_server_type;
HPX_REGISTER_COMPONENT(MatrixCOO_server_type, MatrixCOO_server);

typedef tbsla::hpx_::server::MatrixCOO::get_data_action get_MatrixCOO_data_action;
HPX_REGISTER_ACTION(get_MatrixCOO_data_action);

std::size_t tbsla::hpx_::MatrixCOO::get_n_col() {
  if(this->tiles.size() == 0)
    return 0;
  return tiles[0].get_data().get().get_n_col();
}

std::size_t tbsla::hpx_::MatrixCOO::get_n_row() {
  if(this->tiles.size() == 0)
    return 0;
  return tiles[0].get_data().get().get_n_row();
}

void tbsla::hpx_::MatrixCOO::fill_cdiag(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t cdiag, std::size_t gr, std::size_t gc) {
  this->tiles.resize(gr * gc);
  this->localities = localities;
  std::size_t nl = this->localities.size();
  std::size_t nt = this->tiles.size();

  for (std::size_t pr = 0; pr < gr; pr++) {
    for (std::size_t pc = 0; pc < gc; pc++) {
      this->tiles[pr * gc + pc] = tbsla::hpx_::client::MatrixCOO(this->localities[(pr * gc + pc) * nl / nt], nr, nc, cdiag, pr, pc, gr, gc);
    }
  }
}

void tbsla::hpx_::MatrixCOO::fill_cqmat(std::vector<hpx::id_type> localities, std::size_t nr, std::size_t nc, std::size_t c, double q, unsigned int seed, std::size_t gr, std::size_t gc) {
  this->tiles.resize(gr * gc);
  this->localities = localities;
  std::size_t nl = this->localities.size();
  std::size_t nt = this->tiles.size();

  for (std::size_t pr = 0; pr < gr; pr++) {
    for (std::size_t pc = 0; pc < gc; pc++) {
      this->tiles[pr * gc + pc] = tbsla::hpx_::client::MatrixCOO(this->localities[(pr * gc + pc) * nl / nt], nr, nc, c, q, seed, pr, pc, gr, gc);
    }
  }
}

static Vector_client tbsla::hpx_::detail::spmv_part(tbsla::hpx_::client::MatrixCOO const& A_p, Vector_client const& v_p, Vector_client const& r_p) {
  using hpx::dataflow;
  using hpx::util::unwrapping;

  hpx::shared_future<tbsla::hpx_::detail::MatrixCOO> A_data = A_p.get_data();
  hpx::shared_future<Vector_data> v_data = v_p.get_data();
  return dataflow(hpx::launch::async,
    unwrapping([r_p](tbsla::hpx_::detail::MatrixCOO const& A, Vector_data const& v) -> Vector_client {
      Vector_data r(A.spmv(v.get_vect(), 0));
      return Vector_client(r_p.get_id(), r);
    }),
    A_data, v_data);
}

HPX_PLAIN_ACTION(tbsla::hpx_::detail::spmv_part, spmv_coo_part_action);


Vector_client tbsla::hpx_::MatrixCOO::spmv(Vector_client v) {
  std::vector<Vector_client> spmv_res;
  spmv_res.resize(this->tiles.size());
  for (std::size_t i = 0; i != this->tiles.size(); ++i) {
    spmv_res[i] = Vector_client(this->localities[i * this->localities.size() / this->tiles.size()]);
  }

  spmv_coo_part_action act_spmv;

  using hpx::dataflow;
  for (std::size_t i = 0; i != this->tiles.size(); ++i) {
    using hpx::util::placeholders::_1;
    using hpx::util::placeholders::_2;
    using hpx::util::placeholders::_3;
    auto Op = hpx::util::bind(act_spmv, this->localities[i * this->localities.size() / this->tiles.size()], _1, _2, _3);
    spmv_res[i] = dataflow(hpx::launch::async, Op, this->tiles[i], v, spmv_res[i]);
  }

  return reduce(spmv_res);
}

Vector_client tbsla::hpx_::MatrixCOO::a_axpx_(Vector_client v) {
  Vector_client r = this->spmv(v);
  r = add_vectors(this->localities[0], r, v);
  return this->spmv(r);
}

void tbsla::hpx_::MatrixCOO::wait() {
  for (std::size_t i = 0; i < this->tiles.size(); ++i) {
    tiles[i].get_data().wait();
  }
}

